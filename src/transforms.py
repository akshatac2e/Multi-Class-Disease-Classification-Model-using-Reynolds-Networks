import albumentations as A

def train_transforms(size: int):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.8,1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
    ])

def val_transforms(size: int):
    return A.Compose([
        A.Resize(size, size),
    ])
