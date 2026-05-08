"""
data/transforms.py
Standard torchvision transforms for training and evaluation.
ImageNet mean/std is used because DINOv2 was pretrained on it.
"""

import torchvision.transforms as T

# ImageNet statistics (DINOv2 backbone normalisation)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size=(512, 1024)):
    """Augmentation pipeline for Cityscapes training images."""
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size=(512, 1024)):
    """Deterministic pipeline for validation / evaluation."""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_mask_transform(image_size=(512, 1024)):
    """Nearest-neighbour resize for segmentation masks (no interpolation artefacts)."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
    ])
