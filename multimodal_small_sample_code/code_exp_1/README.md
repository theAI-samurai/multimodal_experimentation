# TinyCLIP

A minimal from-scratch implementation of the CLIP (Contrastive Language–Image Pre-training) model in PyTorch. Built for learning and experimentation with multimodal representation learning.

## Overview

TinyCLIP trains image and text encoders jointly using contrastive learning (InfoNCE loss), aligning image and caption embeddings into a shared vector space — the core idea behind OpenAI's CLIP.

## Architecture

| Component | Implementation |
|-----------|----------------|
| **Image Encoder** | Lightweight CNN (Conv → MaxPool × 3, AdaptiveAvgPool) + linear projection |
| **Text Encoder** | Token embedding + 2-layer Transformer encoder + mean pooling + linear projection |
| **Loss** | Symmetric InfoNCE / CLIP contrastive loss with learnable temperature |
| **Tokenizer** | Simple whitespace-based word tokenizer built from the training corpus |

Both encoders project into a shared `embed_dim=256` space. Embeddings are L2-normalized before computing cosine similarity.

## Requirements

```bash
pip install torch torchvision pillow tqdm numpy
```

You also need the `data_loader.py` module providing `CustomImageCaptionDataset`.

## Dataset Format

The script expects:
- **Images**: a directory of image files (e.g. COCO `val2017/`)
- **Captions**: a plain-text annotation file where each line maps an image filename to a caption

Update the paths in `tinyclip.py` before training:

```python
image_dir       = "mycustomdata/images/val2017"
annotation_file = "mycustomdata/annotations/coco_val_captions.txt"
```

## Usage

```bash
python tinyclip.py
```

Training runs for `num_epochs=5` by default (configurable in the script). Progress is displayed per batch via `tqdm`.

After training, model weights are saved to:
- `tiny_clip_image_encoder.pth`
- `tiny_clip_text_encoder.pth`

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 256 | Shared embedding dimension |
| `max_length` | 32 | Max token length for captions |
| `temperature` | 0.07 | Contrastive loss temperature |
| `batch_size` | 32 | Training batch size |
| `lr` | 1e-4 | Adam learning rate |
| `num_epochs` | 5 | Training epochs |

## File Structure

```
code_exp_1/
├── tinyclip.py       # Model definitions + training script
├── data_loader.py    # CustomImageCaptionDataset (required)
└── README.md
```
