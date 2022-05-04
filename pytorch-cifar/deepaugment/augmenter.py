# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import numpy as np
from imgaug import augmenters as iaa


def normalize(X):
    return (X / 255.0).copy()


def denormalize(X):
    X_dn = (X * 255).astype(np.uint8)
    return X_dn


def transform(aug_type, magnitude, X):
    if aug_type == "crop":
        X_aug = iaa.Crop(px=(0, int(magnitude * 32))).augment_images(X)
    elif aug_type == "gaussian-blur":
        X_aug = iaa.GaussianBlur(sigma=(0, magnitude * 25.0)).augment_images(X)
    elif aug_type == "rotate":
        X_aug = iaa.Affine(rotate=(-180 * magnitude, 180 * magnitude)).augment_images(X)
    elif aug_type == "shear":
        X_aug = iaa.Affine(shear=(-90 * magnitude, 90 * magnitude)).augment_images(X)
    elif aug_type == "translate-x":
        X_aug = iaa.Affine(
            translate_percent={"x": (-magnitude, magnitude), "y": (0, 0)}
        ).augment_images(X)
    elif aug_type == "translate-y":
        X_aug = iaa.Affine(
            translate_percent={"x": (0, 0), "y": (-magnitude, magnitude)}
        ).augment_images(X)
    elif aug_type == "horizontal-flip":
        X_aug = iaa.Fliplr(magnitude).augment_images(X)
    elif aug_type == "vertical-flip":
        X_aug = iaa.Flipud(magnitude).augment_images(X)
    elif aug_type == "sharpen":
        X_aug = iaa.Sharpen(
            alpha=(0, 1.0), lightness=(0.50, 5 * magnitude)
        ).augment_images(X)
    elif aug_type == "emboss":
        X_aug = iaa.Emboss(
            alpha=(0, 1.0), strength=(0.0, 20.0 * magnitude)
        ).augment_images(X)
    elif aug_type == "additive-gaussian-noise":
        X_aug = iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, magnitude * 255), per_channel=0.5
        ).augment_images(X)
    elif aug_type == "dropout":
        X_aug = iaa.Dropout(
            (0.01, max(0.011, magnitude)), per_channel=0.5
        ).augment_images(
            X
        )  # Dropout first argument should be smaller than second one
    elif aug_type == "coarse-dropout":
        X_aug = iaa.CoarseDropout(
            (0.03, 0.15), size_percent=(0.30, np.log10(magnitude * 3)), per_channel=0.2
        ).augment_images(X)
    elif aug_type == "gamma-contrast":
        X_norm = normalize(X)
        X_aug_norm = iaa.GammaContrast(magnitude * 1.75).augment_images(
            X_norm
        )  # needs 0-1 values
        X_aug = denormalize(X_aug_norm)
    elif aug_type == "brighten":
        # coarse salt and pepper for testing
        # X_aug = iaa.CoarseSaltAndPepper(p=0.2, size_percent=magnitude).augment_images(X)

        # brighten didn't work so I replaced it with CoarseSaltanPepper
        X_aug = iaa.Add(
            (int(-40 * magnitude), int(40 * magnitude)), per_channel=0.5
        ).augment_images(
            X
        )  # brighten
    elif aug_type == "invert":
        X_aug = iaa.Invert(1.0).augment_images(X)  # magnitude not used
    elif aug_type == "fog":
        X_aug = iaa.Fog().augment_images(X)  # magnitude not used
    elif aug_type == "clouds":
        X_aug = iaa.Clouds().augment_images(X)  # magnitude not used
    elif aug_type == "histogram-equalize":
        X_aug = iaa.AllChannelsHistogramEqualization().augment_images(
            X
        )  # magnitude not used
    elif aug_type == "super-pixels":  # deprecated
        X_norm = normalize(X)
        X_norm2 = (X_norm * 2) - 1
        X_aug_norm2 = iaa.Superpixels(
            p_replace=(0, magnitude), n_segments=(100, 100)
        ).augment_images(X_norm2)
        X_aug_norm = (X_aug_norm2 + 1) / 2
        X_aug = denormalize(X_aug_norm)
    elif aug_type == "perspective-transform":
        X_norm = normalize(X)
        X_aug_norm = iaa.PerspectiveTransform(
            scale=(0.01, max(0.02, magnitude))
        ).augment_images(
            X_norm
        )  # first scale param must be larger
        np.clip(X_aug_norm, 0.0, 1.0, out=X_aug_norm)
        X_aug = denormalize(X_aug_norm)
    elif aug_type == "elastic-transform":  # deprecated
        X_norm = normalize(X)
        X_norm2 = (X_norm * 2) - 1
        X_aug_norm2 = iaa.ElasticTransformation(
            alpha=(0.0, max(0.5, magnitude * 300)), sigma=5.0
        ).augment_images(X_norm2)
        X_aug_norm = (X_aug_norm2 + 1) / 2
        X_aug = denormalize(X_aug_norm)
    elif aug_type == "add-to-hue-and-saturation":
        X_aug = iaa.AddToHueAndSaturation(
            (int(-45 * magnitude), int(45 * magnitude))
        ).augment_images(X)
    elif aug_type == "coarse-salt-pepper":
        X_aug = iaa.CoarseSaltAndPepper(p=0.2, size_percent=magnitude).augment_images(X)
    elif aug_type == "grayscale":
        X_aug = iaa.Grayscale(alpha=(0.0, magnitude)).augment_images(X)
    else:
        raise ValueError
    return X_aug


def augment_by_policy(
        X, y, *hyperparams
):
    """
    Augment data by applying a set of 5 policies.
    input:
        X: numpy array of images as integers from 0 to 255
        y: numpy array of labels
        hyperparams: list of tuples of (hyperparameter, value)
    """
    X_portion = X
    y_portion = y

    all_X_portion_aug = None
    all_y_portion = None

    for i in range(0, len(hyperparams) - 1, 4):

        X_portion_aug = transform(hyperparams[i], hyperparams[i + 1], X_portion)  # first transform

        assert (
                X_portion_aug.min() >= -0.1 and X_portion_aug.max() <= 255.1
        ), "first transform is unvalid"
        np.clip(X_portion_aug, 0, 255, out=X_portion_aug)

        X_portion_aug = transform(
            hyperparams[i + 2], hyperparams[i + 3], X_portion_aug
        )  # second transform
        assert (
                X_portion_aug.min() >= -0.1 and X_portion_aug.max() <= 255.1
        ), "second transform is unvalid"
        np.clip(X_portion_aug, 0, 255, out=X_portion_aug)

        if all_X_portion_aug is None:
            all_X_portion_aug = X_portion_aug
            all_y_portion = y_portion
        else:
            all_X_portion_aug = np.concatenate([all_X_portion_aug, X_portion_aug])
            all_y_portion = np.concatenate([all_y_portion, y_portion])

    augmented_data = {
        "X_train": all_X_portion_aug,
        "y_train": all_y_portion,
    }  # back to normalization

    return augmented_data  # augmenteed data is mostly smaller than whole data
