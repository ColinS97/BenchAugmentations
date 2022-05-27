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

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

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


args = parser.parse_args()


def validate_args(args):
    bools = [
        args.randaugment,
        args.deepaugment,
        args.trivialaugment,
        args.baseline,
        args.resume,
    ]
    print(sum(bools))
    if sum(bools) > 1:
        raise ValueError(
            "Only one of --randaugment, --deepaugment, --baseline, --trivialaugment, --resume can be used"
        )


validate_args(args)

train_transforms_list = []

if args.baseline:
    train_transforms_list.extend(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    )

if args.randaugment:
    train_transforms_list.append(aug_lib.RandAugment(1, 30))

if args.trivialaugment:
    train_transforms_list.append(aug_lib.TrivialAugment())


if args.deepaugment:
    raise ValueError("deepaugment not implemented yet")


train_transforms_list.extend(
    [torchvision.transforms.ToTensor(), cifar10_normalization()]
)


train_transforms = torchvision.transforms.Compose(train_transforms_list)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)


info = INFO["pathmnist"]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])
download = True

DataClass = getattr(medmnist, info["python_class"])


# load the data
train_dataset = DataClass(split="train", transform=train_transforms, download=download)
test_dataset = DataClass(split="test", transform=test_transforms, download=download)

pil_dataset = DataClass(split="train", download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
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
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()
        # define loss function and optimizer
        if task == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if task == "multi-label, binary-class":
            targets = y.to(torch.float32)
            loss = self.criterion(logits, targets)
        else:
            targets = y.squeeze().long()
            loss = self.criterion(logits, targets)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)

        if task == "multi-label, binary-class":
            targets = y.to(torch.float32)
            loss = self.criterion(logits, targets)
        else:
            targets = y.squeeze().long()
            loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


model = LitResnet(lr=0.05)

trainer = Trainer(
    max_epochs=args.epochs,
    strategy="ddp",
    accelerator="gpu",
    devices="auto",
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ],
)
start = time.time()
print("Start:" + str(start))
trainer.fit(model, train_loader, train_loader_at_eval)
trainer.test(model, test_loader)

end = time.time()


print("End:" + str(end))
print("Duration:" + str(end - start))
