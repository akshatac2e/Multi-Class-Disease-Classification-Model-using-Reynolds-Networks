from __future__ import annotations
import os, torch, yaml
from torch.utils.data import DataLoader
from .dataset import ImageFolderDataset
from .transforms import val_transforms
from .utils import save_json
from .models.reynoldsnet import ReynoldsNetTiny

def evaluate(ckpt_path: str, data_root: str, img_size: int = 224):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes", [])
    model = ReynoldsNetTiny(num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ImageFolderDataset(data_root, img_size, classes, transform=val_transforms(img_size))
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    correct, total = 0, 0
    with torch.no_grad():
        for x,y in dl:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred==y).sum().item()
            total += y.numel()
    acc = correct/total if total else 0.0
    save_json(os.path.join("reports","eval.json"), {"accuracy": acc, "n": total})
    print(f"[eval] accuracy={acc:.4f} n={total}")
