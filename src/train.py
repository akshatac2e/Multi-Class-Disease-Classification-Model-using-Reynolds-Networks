from __future__ import annotations
import os, yaml
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from .config import load_config
from .utils import set_seed, device_select, save_json
from .transforms import train_transforms, val_transforms
from .dataset import ImageFolderDataset
from .models.reynoldsnet import ReynoldsNetTiny

def train_loop(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    device = device_select(cfg.device)

    with open("config/classes.yaml","r") as f:
        classes = yaml.safe_load(f)["classes"]
    num_classes = len(classes)

    train_ds = ImageFolderDataset(cfg.data["train_root"], cfg.data["img_size"], classes, transform=train_transforms(cfg.data["img_size"]))
    val_ds   = ImageFolderDataset(cfg.data["val_root"],   cfg.data["img_size"], classes, transform=val_transforms(cfg.data["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=cfg.data["batch_size"], shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=cfg.data["batch_size"], shuffle=False, num_workers=2)

    model = ReynoldsNetTiny(num_classes=num_classes, dropout=cfg.train.get("dropout", 0.3)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim["lr"], weight_decay=cfg.optim["weight_decay"])

    best_acc, best_path = 0.0, None
    for epoch in range(cfg.optim["epochs"]):
        model.train()
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step(); opt.zero_grad()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct/total if total>0 else 0.0
        if acc > best_acc:
            best_acc = acc
            os.makedirs(cfg.save["ckpt_dir"], exist_ok=True)
            best_path = os.path.join(cfg.save["ckpt_dir"], "best.ckpt")
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
        print(f"[epoch {epoch+1}] val_acc={acc:.4f} best={best_acc:.4f}")

    save_json(os.path.join("reports","train_metrics.json"), {"best_val_accuracy": best_acc, "best_ckpt": best_path or ""})
    print(f"[done] best_acc={best_acc:.4f} ckpt={best_path}")
