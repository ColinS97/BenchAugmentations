import os
import time

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import medmnist
from medmnist import INFO, Evaluator

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

# from torchmetrics.functional import accuracy, auc
import torchmetrics.functional

import aug_lib

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

seed_everything(7)

print("CPU Count:", os.cpu_count())

PATH_DATASETS = "./data"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

parser = argparse.ArgumentParser(description="PyTorch Lightning CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--epochs", default=10, type=int, help="how many epochs should the net train for"
)
parser.add_argument(
    "--slurm_id", default=00000000, type=int, help="slurm id for job array"
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--deepaugment",
    "-da",
    action="store_true",
    help="use deepaugment policy for data augmentation place policy.txt next to this file",
)
parser.add_argument(
    "--baseline",
    "-ba",
    action="store_true",
    help="use baseline transforms for data augmentation",
)
parser.add_argument(
    "--randaugment",
    "-ra",
    action="store_true",
    help="use randaugment transformer for data augmentation",
)
parser.add_argument(
    "--trivialaugment",
    "-ta",
    action="store_true",
    help="use trivialaugment transformer for data augmentation",
)
parser.add_argument(
    "--noaugment",
    "-na",
    action="store_true",
    help="use no augmentations",
)


args = parser.parse_args()

epochs = args.epochs
slurm_id = args.slurm_id


def validate_args(args):
    bools = [
        args.randaugment,
        args.deepaugment,
        args.trivialaugment,
        args.baseline,
        args.noaugment,
        args.resume,
    ]
    print(sum(bools))
    if sum(bools) > 1:
        raise ValueError(
            "Only one of --randaugment, --deepaugment, --baseline, --trivialaugment, --resume can be used"
        )


validate_args(args)

train_transforms_list = []
aug_type = ""

if args.noaugment:
    aug_type = "noaugment"
    print("Using no augmentation")

if args.baseline:
    # WARNING baseline is still adjusted to cifar10
    aug_type = "baseline"
    train_transforms_list.extend(
        [torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
    )

if args.randaugment:
    aug_type = "randaugment"
    train_transforms_list.append(aug_lib.RandAugment(1, 30))

if args.trivialaugment:
    aug_type = "trivialaugment"
    train_transforms_list.append(aug_lib.TrivialAugment())


if args.deepaugment:
    aug_type = "deepaugment"
    raise ValueError("deepaugment not implemented yet")


train_transforms_list.extend(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


train_transforms = torchvision.transforms.Compose(train_transforms_list)

test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data_flag = "pathmnist"
info = INFO[data_flag]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])
download = True

DataClass = getattr(medmnist, info["python_class"])


# load the data
train_dataset = DataClass(split="train", transform=train_transforms, download=download)
test_dataset = DataClass(split="test", transform=test_transforms, download=download)
val_dataset = DataClass(split="val", transform=test_transforms, download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=n_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.001, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self(x)
        targets = torch.squeeze(targets, 1).long()
        loss = F.cross_entropy(logits, targets)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):

        x, targets = batch
        logits = self(x)
        targets = torch.squeeze(targets, 1).long()
        loss = F.cross_entropy(logits, targets)
        preds = F.softmax(logits, dim=1)
        # targets = targets.float().resize_(len(targets), 1)
        acc = torchmetrics.functional.accuracy(preds, targets)
        auroc = torchmetrics.functional.auroc(num_classes=n_classes)
        auc = auroc(preds, targets)
        if stage:
            self.log(f"{stage}_loss", loss, sync_dist=True)
            self.log(f"{stage}_acc", acc, sync_dist=True)
            self.log(f"{stage}_auc", auc, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma
            )
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


model = LitResnet(lr=0.001, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1)

trainer = Trainer(
    max_epochs=args.epochs,
    strategy="ddp",
    accelerator="gpu",
    devices="auto",
    logger=CSVLogger(save_dir="logs/pyjob_" + str(slurm_id) + "_" + aug_type + "/"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ],
)
start = time.time()
print("Start:" + str(start))
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

end = time.time()


print("End:" + str(end))
print("Duration:" + str(end - start))
