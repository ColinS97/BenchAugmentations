import numpy as np
import pandas as pd
from lib.cutout import cutout_numpy
from augmenter import augment_by_policy

import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

AUG_TYPES = [
    "crop",
    "gaussian-blur",
    "rotate",
    "shear",
    "translate-x",
    "translate-y",
    "sharpen",
    "emboss",
    "additive-gaussian-noise",
    "dropout",
    "coarse-dropout",
    "gamma-contrast",
    "brighten",
    "invert",
    "fog",
    "clouds",
    "add-to-hue-and-saturation",
    "coarse-salt-pepper",
    "horizontal-flip",
    "vertical-flip",
]


def augment_type_chooser():
    """A random function to choose among augmentation types

    Returns:
        function object: np.random.choice function with AUG_TYPES input
    """
    return np.random.choice(AUG_TYPES)


def random_flip(x):
    """Flip the input x horizontally with 50% probability."""
    if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
    return x


def zero_pad_and_crop(img, amount=4):
    """Zero pad by `amount` zero pixels on each side then take a random crop.
  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.
  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
    padded_img = np.zeros(
        (img.shape[0] + amount * 2, img.shape[1] + amount * 2, img.shape[2])
    )
    padded_img[amount: img.shape[0] + amount, amount: img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    new_img = padded_img[top: top + img.shape[0], left: left + img.shape[1], :]
    return new_img


def apply_default_transformations(X):
    # apply cutout
    X_aug = []
    for img in X:
        img_aug = zero_pad_and_crop(img, amount=4)
        img_aug = cutout_numpy(img_aug, size=6)
        X_aug.append(img_aug)
    return X_aug


def deepaugment_image_generator(X, y, policy, batch_size=64, augment_chance=0.5, show_policy=False):
    """Yields batch of images after applying random augmentations from the policy

    Each image is augmented by 50% chance. If augmented, one of the augment-chain in the policy is applied.
    Which augment-chain to apply is chosen randomly.

    Args:
        X (numpy.array):
        labels (numpy.array):
        policy (pd.DF): Dataframe of policies

    Returns:
        tuple: (X_batch, y_batch) X_batch is a numpy array of images as type uint8 from 0 to 255 and y_batch is a batch of labels
    """
    if(show_policy):
        print("Policies are:")
        print(policy)
        print()

    while True:
        ix = np.arange(len(X))
        np.random.shuffle(ix)
        for i in range(len(X) // batch_size):
            _ix = ix[i * batch_size: (i + 1) * batch_size]
            _X = X[_ix]
            _y = y[_ix]

            tiny_batch_size = 4
            aug_X = _X[0:tiny_batch_size]
            aug_y = _y[0:tiny_batch_size]
            # images are given to the augenter in batches of 5
            for j in range(1, len(_X) // tiny_batch_size):
                tiny_X = _X[j * tiny_batch_size: (j + 1) * tiny_batch_size]
                tiny_y = _y[j * tiny_batch_size: (j + 1) * tiny_batch_size]
                # if the random number is smaller than the augment_chance, augment the tiny batch of images
                if np.random.rand() <= augment_chance:
                    # select a random policy from usually the top 20 policies
                    aug_chain = np.random.choice(policy)
                    aug_chain[
                        "portion"
                    ] = 1.0  # last element is portion, which we want to be 1
                    hyperparams = list(aug_chain.values())

                    aug_data = augment_by_policy(tiny_X, tiny_y, *hyperparams)

                    # commented out because it was causing errors
                    aug_data["X_train"] = apply_default_transformations(
                        aug_data["X_train"]
                     )

                    aug_X = np.concatenate([aug_X, aug_data["X_train"]])
                    aug_y = np.concatenate([aug_y, aug_data["y_train"]])
                else:
                    aug_X = np.concatenate([aug_X, tiny_X])
                    aug_y = np.concatenate([aug_y, tiny_y])
            yield aug_X.astype(np.uint8), aug_y


X = np.random.rand(200, 32, 32, 3)
y = np.random.randint(10, size=200)

batch_size = 64

policy = [
    {'A_aug1_type': 'additive-gaussian-noise',
     'A_aug1_magnitude': 0.272,
     'A_aug2_type': 'fog',
     'A_aug2_magnitude': 0.628,
     'B_aug1_type': 'translate-x',
     'B_aug1_magnitude': 0.681,
     'B_aug2_type': 'coarse-salt-pepper',
     'B_aug2_magnitude': 0.056,
     'C_aug1_type': 'sharpen',
     'C_aug1_magnitude': 0.765,
     'C_aug2_type': 'sharpen',
     'C_aug2_magnitude': 0.839,
     'D_aug1_type': 'vertical-flip',
     'D_aug1_magnitude': 0.92,
     'D_aug2_type': 'invert',
     'D_aug2_magnitude': 0.431,
     'E_aug1_type': 'shear',
     'E_aug1_magnitude': 0.028,
     'E_aug2_type': 'sharpen',
     'E_aug2_magnitude': 0.443},
    {'A_aug1_type': 'coarse-dropout',
     'A_aug1_magnitude': 0.256,
     'A_aug2_type': 'fog',
     'A_aug2_magnitude': 0.534,
     'B_aug1_type': 'sharpen',
     'B_aug1_magnitude': 0.662,
     'B_aug2_type': 'horizontal-flip',
     'B_aug2_magnitude': 0.978,
     'C_aug1_type': 'brighten',
     'C_aug1_magnitude': 0.151,
     'C_aug2_type': 'dropout',
     'C_aug2_magnitude': 0.164,
     'D_aug1_type': 'vertical-flip',
     'D_aug1_magnitude': 0.004,
     'D_aug2_type': 'add-to-hue-and-saturation',
     'D_aug2_magnitude': 0.955,
     'E_aug1_type': 'shear',
     'E_aug1_magnitude': 0.202,
     'E_aug2_type': 'crop',
     'E_aug2_magnitude': 0.264},
]


def load_k_policies_from_csv(nb_df, k=20):
    """loading the top policies from dataframe into parsable format for image generator

    Args:
        nb_df (pd.DF):
        k (int, optional): Defaults to 20.

    Returns:
        list: list of dictionaries
        """
    trial_avg_val_acc_df = (
        nb_df.drop_duplicates(["trial_no", "sample_no"])
            .groupby("trial_no")
            .mean()["mean_late_val_acc"]
            .reset_index()
    )[["trial_no", "mean_late_val_acc"]]

    x_df = pd.merge(
        nb_df.drop(columns=["mean_late_val_acc"]),
        trial_avg_val_acc_df,
        on="trial_no",
        how="left",
    )

    x_df = x_df.sort_values("mean_late_val_acc", ascending=False)

    baseline_val_acc = x_df[x_df["trial_no"] == 0]["mean_late_val_acc"].values[0]

    x_df["expected_accuracy_increase(%)"] = (
                                                    x_df["mean_late_val_acc"] - baseline_val_acc
                                            ) * 100

    top_df = x_df.drop_duplicates(["trial_no"]).sort_values(
        "mean_late_val_acc", ascending=False
    )[:k]

    SELECT = [
        "trial_no",
        'A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
        'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
        'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
        'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
        'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude',
        "mean_late_val_acc", "expected_accuracy_increase(%)"
    ]
    top_df = top_df[SELECT]

    top_policies_list = top_df[
        ['A_aug1_type', 'A_aug1_magnitude', 'A_aug2_type', 'A_aug2_magnitude',
         'B_aug1_type', 'B_aug1_magnitude', 'B_aug2_type', 'B_aug2_magnitude',
         'C_aug1_type', 'C_aug1_magnitude', 'C_aug2_type', 'C_aug2_magnitude',
         'D_aug1_type', 'D_aug1_magnitude', 'D_aug2_type', 'D_aug2_magnitude',
         'E_aug1_type', 'E_aug1_magnitude', 'E_aug2_type', 'E_aug2_magnitude']
    ].to_dict(orient="records")

    return top_policies_list


def test_deepaugment_image_generator(X=X, y=y, batchsize=batch_size, policy=policy):
    gen = deepaugment_image_generator(X, y, policy, batch_size=batch_size)

    a = next(gen)
    b = next(gen)
    c = next(gen)
    # if no error happened during next()'s, it is good


def get_deepaugment_image_generator(X=X, y=y, batchsize=batch_size, policy=policy):
    gen = deepaugment_image_generator(X, y, policy, batch_size=batch_size)

    return gen


if __name__ == "__main__":
    test_deepaugment_image_generator()
