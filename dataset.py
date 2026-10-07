import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 380):
    return A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.7, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.7),
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0), p=1.0),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 30), hole_width_range=(8, 30), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int = 380):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root: str, alb_transform=None, val_transform=None):
        super().__init__(root)
        self.alb_transform = alb_transform
        self.val_transform = val_transform

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)

        if self.alb_transform:
            image = self.alb_transform(image=image_np)['image']
        elif self.val_transform:
            image = self.val_transform(image)

        return image, label