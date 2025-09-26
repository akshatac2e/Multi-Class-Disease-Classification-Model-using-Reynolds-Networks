# Automated Medical Microscopy Diagnostic Platform
AI-Powered Disease Detection (Jan'24–Apr'24)

End-to-end pipeline for cell microscopy: segmentation -> feature extraction (with Reynolds operators) -> multi-class classification with attention -> uncertainty & confidence reporting. Designed for rapid triage and clinic-ready batch inference.

## Highlights
- Model: "ReynoldsNet" backbone (custom PyTorch) + attention head for fast representation learning.
- Performance: 98.5% accuracy on ~20k images (project context). Rapid learning: ~94% by epoch 3, ~95% validation accuracy.
- Algorithms: Reynolds operators on high-level features to reduce search complexity (factorial -> polynomial scale).
- System: Raw image -> cell segmentation -> classification -> per-class counts, confidence & uncertainty (MC Dropout).
- Modular: Plug new disease classes in config/classes.yaml.

## Repo Layout
medical-microscopy/
- README.md, LICENSE, requirements.txt, .gitignore, .env.example
- config/ (train.yaml, classes.yaml)
- data/ (sample_structure.txt)
- models/, reports/
- src/ (dataset, transforms, segmentation U-Net, Reynolds operators, ReynoldsNet, attention head, train/eval/infer)
- cli.py

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

## Quickstart (toy run)
# 0) Organize data as data/train/<class>/*.png and data/val/<class>/*.png
# 1) Train
python -m cli train --config config/train.yaml
# 2) Evaluate
python -m cli evaluate --ckpt models/best.ckpt --data-root data/val
# 3) Inference on a folder; outputs JSON per image
python -m cli infer --ckpt models/best.ckpt --input-dir data/val --out reports/infer_json

Note: This repo is a template mirroring the original project. Results depend on data/labels & hardware.
