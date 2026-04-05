# Multimodal Code Learning — TinyCLIP

A from-scratch PyTorch exploration of CLIP (Contrastive Language-Image Pre-training), progressively refined across two experiments. Each experiment trains aligned image-text embeddings on a local COCO 2017 validation subset.

NOTE :  copy the -->  data_loader.py to code2 or code1 folder to run end to end

---

## Repository Structure

```
multimodal_code_learning/
├── coco_dataset_convert.py     # Converts COCO annotations to pipe-separated .txt format
├── data_loader.py              # Shared COCO-style dataset loader
├── requirements.txt
├── mycustomdata/               # Shared dataset (images + annotations)
│   ├── annotations/
│   │   ├── captions_val2017.json
│   │   └── coco_val_captions.txt   # format: filename|caption
│   └── images/
│       └── val2017/
├── code_exp_1/                 # Experiment 1 — minimal single-file CLIP
│   ├── tinyclip.py
│   └── README.md
└── code_exp_2/                 # Experiment 2 — modular, production-style CLIP
    ├── clip.py
    ├── data_loader.py
    ├── encoder_image.py
    ├── encoder_text.py
    ├── tokenizer.py
    ├── tinyclip_train.py
    ├── image_encoder.pth
    ├── text_encoder.pth
    └── README.md
```

---

## Experiments

### Experiment 1 — `code_exp_1/`

A minimal self-contained CLIP implementation in a single file (`tinyclip.py`). Good starting point for understanding the core mechanics:

- Lightweight CNN image encoder
- 2-layer Transformer text encoder
- Symmetric InfoNCE loss with learnable temperature
- Word-level tokenizer built from training captions

See [code_exp_1/README.md](code_exp_1/README.md) for details.

### Experiment 2 — `code_exp_2/`

A refactored, modular version with improved architecture and training setup:

- **Image Encoder**: `ResNetEncoder` — 3-stage residual network with BasicResNetBlock layers
- **Text Encoder**: `SimpleTextEncoder` — 4-layer, 8-head Transformer
- **Training**: Mixed-precision (AMP), gradient clipping, AdamW optimizer
- Each component is in its own module for clarity and reuse

See [code_exp_2/README.md](code_exp_2/README.md) for full architecture and hyperparameter details.

---

## Dataset Setup

Both experiments use the COCO 2017 validation split. Download the images and annotations, then convert to the pipe-separated format:

```bash
python coco_dataset_convert.py
```

This produces `mycustomdata/annotations/coco_val_captions.txt`:

```
000000000139.jpg|A person riding a motorcycle on a dirt road.
```

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision pillow tqdm

# Run Experiment 1 (single-file)
cd code_exp_1
python tinyclip.py

# Run Experiment 2 (modular, recommended)
cd code_exp_2
python tinyclip_train.py
```

---

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- tqdm
