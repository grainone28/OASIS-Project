import torchvision.transforms as T
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size=(512, 1024)):
    # Augmentation pipeline for Cityscapes training images.
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size=(512, 1024)):
    # Deterministic pipeline for validation / evaluation.
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])



def get_mask_transform(image_size=(512, 1024)):

    def _resize_mask(mask: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),  
            size=image_size,
            mode="nearest"
        ).squeeze().long()                        
    return _resize_mask



def get_val_transform_erfnet(image_size=(512, 1024)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])


def get_train_transform_erfnet(image_size=(512, 1024)):
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])