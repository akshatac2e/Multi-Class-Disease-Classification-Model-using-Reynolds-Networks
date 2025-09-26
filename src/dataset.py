from __future__ import annotations
import os, glob
from typing import Tuple, List
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, img_size: int, classes: List[str], transform=None):
        self.samples = []
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for cls in classes:
            for p in glob.glob(os.path.join(root, cls, "*")):
                if os.path.isfile(p):
                    self.samples.append((p, self.class_to_idx[cls]))
        self.img_size = img_size
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, y = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.transform: img = self.transform(image=img)["image"]
        img = (img.astype(np.float32) / 255.0).transpose(2,0,1)
        return torch.from_numpy(img), y
