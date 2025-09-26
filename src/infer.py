from __future__ import annotations
import os, glob, json
import cv2, torch
import numpy as np
from .transforms import val_transforms
from .models.reynoldsnet import ReynoldsNetTiny

def mc_dropout_enable(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def softmax(x): return np.exp(x)/np.exp(x).sum(-1, keepdims=True)

def infer_folder(ckpt_path: str, input_dir: str, out_dir: str, img_size: int = 224, mc_passes: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes", [])
    model = ReynoldsNetTiny(num_classes=len(classes), dropout=0.3)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mc_dropout_enable(model)

    tfm = val_transforms(img_size)
    results = {}
    for p in glob.glob(os.path.join(input_dir, "*")):
        if not os.path.isfile(p): continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tfm(image=img)["image"]
        x = (img.astype(np.float32)/255.0).transpose(2,0,1)[None]
        x = torch.from_numpy(x)

        logits_list = []
        with torch.no_grad():
            for _ in range(mc_passes):
                logits = model(x).cpu().numpy()[0]
                logits_list.append(logits)
        logits_arr = np.stack(logits_list, axis=0)
        probs = np.exp(logits_arr)/np.exp(logits_arr).sum(-1, keepdims=True)
        mean_probs = probs.mean(axis=0)
        entropy = float(-(mean_probs * np.log(mean_probs + 1e-9)).sum())

        top_idx = int(mean_probs.argmax())
        results[os.path.basename(p)] = {
            "pred_class": classes[top_idx] if classes else top_idx,
            "confidence": float(mean_probs[top_idx]),
            "uncertainty_entropy": entropy,
            "probs": { (classes[i] if classes else str(i)): float(mean_probs[i]) for i in range(len(mean_probs)) }
        }

    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[infer] wrote {os.path.join(out_dir, 'predictions.json')}")
