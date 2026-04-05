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

---

## Improvements suggestions

### Better Text Encoder Options

The current `SimpleTextEncoder` can be replaced with stronger pre-trained encoders. The table below ranks alternatives by capability:

| Rank | Text Encoder | Type | Params (approx) | Why Better than SimpleTextEncoder | Best For | Difficulty to Integrate | Hugging Face Example |
|------|-------------|------|-----------------|-----------------------------------|----------|------------------------|----------------------|
| 1 | ModernBERT (Base/Large) | Encoder-only Transformer | 149M / 395M | Drop-in BERT replacement with modern training (longer context, better efficiency, RoPE, GeGLU, etc.). Outperforms old BERT on embeddings & retrieval. | Contrastive learning, retrieval | Medium (use HF) | `answerdotai/ModernBERT-base` |
| 2 | SigLIP text tower (or CLIP text encoder) | Transformer | ~100–300M | Trained with sigmoid loss on massive image-text data. Excellent for contrastive alignment. | Pure CLIP-style training | Easy–Medium | OpenCLIP or `google/siglip-base` variants |
| 3 | Gemma-2B / Gemma-3 (text part) | Decoder-only LLM | 2B+ | Very strong contextual understanding. Many 2025–2026 VLMs use Gemma-style decoders. | When you want richer semantics | Medium-High | `google/gemma-2-2b` |
| 4 | Qwen2.5 / Qwen3 Embedding | LLM-based embedding model | 0.6B–7B | Instruction-aware, multilingual, excellent for retrieval & multimodal. | Multilingual captions | Easy (via sentence-transformers or HF) | `Qwen/Qwen3-Embedding-0.6B` |
| 5 | RoBERTa or DeBERTa-v3 | Encoder-only | 125M–300M | Still strong baselines with better training than original BERT. | Quick upgrade | Easy | `roberta-base`, `microsoft/deberta-v3-base` |
| 6 | DistilBERT or MiniLM | Distilled encoder | 66M / 33M | Much faster & lighter than your current encoder while being stronger. | Speed + small size | Very Easy | `distilbert-base-uncased` |
| 7 | Jina Embeddings v4 or EmbeddingGemma | Multimodal-aware embedding | 300M+ | Supports text + vision in some variants; great for unified retrieval. | Future-proof multimodal | Medium | `jinaai/jina-embeddings-v4` |
