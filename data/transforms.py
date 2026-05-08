"""
data/transforms.py
Joint transforms for images and segmentation masks.

Key idea:
- Geometric transforms (resize, flip, crop) must be applied to image AND mask
  with the same random parameters.
- Photometric transforms (color jitter, normalization) are applied ONLY to the image.
"""
# NOTE:
# Previous implementation applied random augmentations (e.g. horizontal flip)
# only to the input image, while the segmentation mask was transformed separately.
# This causes image/label misalignment, which is a critical bug for semantic segmentation.
#
# The current implementation uses joint geometric transforms so that image and mask
# always share the same resize/flip parameters. Photometric transforms
# (color jitter, normalization) are still applied only to the image.

from typing import Tuple, Callable

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F


# ImageNet statistics (DINOv2 backbone normalisation)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class JointResize:
    """Resize image and mask to the same size.

    Image: bilinear interpolation.
    Mask : nearest-neighbour to avoid label interpolation.
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(mask,  self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

# Important: the same random flip must be applied to both image and mask.
# Otherwise the model sees pixels and labels that no longer correspond.
class JointRandomHorizontalFlip:
    """Random horizontal flip for image and mask with the same coin flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        import random
        # We sample one shared random decision and apply it to both image and mask.
        # Using independent random flips would break pixel-label alignment.
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class JointCompose:
    """Compose a list of joint transforms (image, mask) -> (image, mask)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_train_joint_transform(image_size=(512, 1024)) -> Callable:
    """Geometric + photometric pipeline for training.

    Returns a function (image, mask) -> (image_tensor, mask_pil).
    Mask stays as PIL so it can be encoded to trainId later.
    """
    joint_geo = JointCompose([
        JointResize(image_size),
        JointRandomHorizontalFlip(p=0.5),
    ])

    color_jitter = T.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
    )
    to_tensor_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def _transform(image: Image.Image, mask: Image.Image):
        # geometric transforms on both
        image, mask = joint_geo(image, mask)
        # photometric transforms only on image
        image = color_jitter(image)
        image = to_tensor_norm(image)
        return image, mask  # mask still PIL (labelIds)

    return _transform


def get_val_joint_transform(image_size=(512, 1024)) -> Callable:
    """Deterministic pipeline for validation / evaluation."""
    joint_geo = JointCompose([
        JointResize(image_size),
    ])
    to_tensor_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def _transform(image: Image.Image, mask: Image.Image):
        image, mask = joint_geo(image, mask)
        image = to_tensor_norm(image)
        return image, mask

    return _transform