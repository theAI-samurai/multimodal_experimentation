# TinyClip

A lightweight implementation of OpenAI's [CLIP](https://openai.com/research/clip) (Contrastive Language-Image Pre-training) that learns joint image-text embeddings using contrastive learning. Trained on a local COCO 2017 validation subset.

---

## Project Structure

```
code_exp_2/
├── clip.py               # Symmetric contrastive (InfoNCE) loss
├── data_loader.py        # COCO-style dataset and DataLoader
├── encoder_image.py      # CNN image encoders (SimpleCNN & ResNetEncoder)
├── encoder_text.py       # Transformer-based text encoder
├── tokenizer.py          # Simple word-level tokenizer
├── tinyclip_train.py     # Main training script
├── image_encoder.pth     # Saved image encoder weights
├── text_encoder.pth      # Saved text encoder weights
└── mycustomdata/
    ├── annotations/
    │   ├── captions_val2017.json
    │   └── coco_val_captions.txt   # format: filename|caption
    └── images/
        └── val2017/
```

---

## Architecture

### Image Encoder — `ResNetEncoder`

| Layer | Details |
|-------|---------|
| Stem | Conv2d(3→64, 7×7, stride=2) + BN + ReLU + MaxPool |
| Layer 1 | 2× BasicResNetBlock (64→64) |
| Layer 2 | 2× BasicResNetBlock (64→128, stride=2) |
| Layer 3 | 2× BasicResNetBlock (128→256, stride=2) |
| Head | AdaptiveAvgPool → Linear(256→256) |

### Text Encoder — `SimpleTextEncoder`

| Component | Details |
|-----------|---------|
| Embedding | vocab_size → 256 |
| Transformer | 4 layers, 8 heads, FFN dim 512, dropout 0.1 |
| Pooling | Mean over sequence length |
| Head | Linear(256→256) |
| Max tokens | 32 (padded / truncated) |

### Loss — `CLIPLoss`

Symmetric InfoNCE loss over a batch similarity matrix:

$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{\text{img}\to\text{txt}} + \mathcal{L}_{\text{txt}\to\text{img}}\right)$$

Temperature is learnable, initialized at **0.07**.

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Epochs | 10 |
| Gradient clipping | max\_norm = 1.0 |
| Mixed precision | Enabled (AMP) |
| Embedding dim | 256 |
| Image input size | 224×224 RGB |

### Dataset

Images and captions are loaded from the COCO 2017 validation split stored locally under `mycustomdata/`. The annotation file uses a simple pipe-separated format:

```
000000000139.jpg|A person riding a motorcycle on a dirt road.
```

When an image has multiple captions, one is chosen randomly per epoch.

Image preprocessing follows ImageNet conventions:
- Resize to 224×224
- Normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

### Running Training

```bash
python tinyclip_train.py
```

Trained weights are saved to `image_encoder.pth` and `text_encoder.pth`.

---

## Tokenizer

`SimpleTokenizer` builds a word-level vocabulary from training captions. Special tokens:

| Token | ID |
|-------|----|
| `<PAD>` | 0 |
| `<UNK>` | 1 |

Sequences are truncated or zero-padded to a maximum length of **32** tokens.

---

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- Pillow

Install with:

```bash
pip install torch torchvision pillow
```
