# POLY-SIM — Project Info & FAQ

---

## FAQ

### Q: What does "UNEXPECTED" and "MISSING" mean when loading a HuggingFace model?

When loading `ViTModel` from `google/vit-base-patch16-224` you may see:

```
ViTModel LOAD REPORT from: google/vit-base-patch16-224
Key                 | Status
--------------------+------------
classifier.weight   | UNEXPECTED
classifier.bias     | UNEXPECTED
pooler.dense.weight | MISSING
pooler.dense.bias   | MISSING
```

**UNEXPECTED — `classifier.weight`, `classifier.bias`**

The HuggingFace checkpoint was saved as a `ViTForImageClassification` model (trained on ImageNet-21k with a 1000-class head). You are loading it as a bare `ViTModel` (encoder only, no classification head). The classifier weights exist in the file but your architecture has no place for them — they are silently dropped.

→ **Safe to ignore.** The ImageNet classifier is not needed.

**MISSING — `pooler.dense.weight`, `pooler.dense.bias`**

`ViTModel` defines an optional pooler layer (a linear layer over the `[CLS]` token) but the original checkpoint was not saved with one. These weights are randomly initialised.

→ **Also safe to ignore.** `FaceEncoder` in `scripts/model.py` reads the `[CLS]` token directly from `last_hidden_state[:, 0]` — the pooler is never called.

**Summary**

| Key | Reason | Impact |
|---|---|---|
| `classifier.*` UNEXPECTED | checkpoint has ImageNet head, model does not | None — dropped |
| `pooler.dense.*` MISSING | model has pooler, checkpoint does not | None — pooler is unused |

The full encoder (all transformer layers, patch embeddings, positional embeddings) loads correctly — that is what matters.

---

### Q: Why does training start from `start_epoch` instead of 1 sometimes?

This happens when `RESUME_FROM` is set in `.env`. The training script resumes from the saved epoch and continues until `EPOCHS` is reached. Set `RESUME_FROM=` (empty) to train from scratch.

---

### Q: What checkpoints are saved?

| File | When saved |
|---|---|
| `checkpoints/best.pt` | Every time a new best accuracy is reached |
| `checkpoints/epoch_XXXX.pt` | Every `CHECKPOINT_INTERVAL` epochs (default: 10) |
| `checkpoints/last.pt` | At the end of training |

All checkpoints include model weights, optimizer state, scheduler state, best accuracy, and full config — sufficient to resume training exactly.

---

### Q: How do I resume training from a checkpoint?

Set in `.env`:
```ini
RESUME_FROM=checkpoints/epoch_0030.pt
EPOCHS=60   # must be greater than the resumed epoch
```
Then run:
```bash
python scripts/train.py
```

---

### Q: Which model combination should I use?

| `FACE_ENCODER` | `AUDIO_ENCODER` | Best for | VRAM |
|---|---|---|---|
| `resnet50` | `unispeech_sv` | P3/P4 fast baseline | ~4 GB |
| `vit_base` | `wavlm_base` | P3–P6 balanced | ~8 GB |
| `vit_large` | `wavlm_large` | P5/P6 best cross-lingual | ~16 GB |

---

### Q: What is modality dropout (`MASK_PROB`)?

During training, the face embedding is replaced with a learnable `[MASK]` token with probability `MASK_PROB`. This forces the audio branch to learn speaker identity independently — directly simulating the test condition where no face is available (protocols P4 and P6).

```
MASK_PROB=0.0  →  always face + audio    (P3/P5 only)
MASK_PROB=0.3  →  30% audio-only steps   (recommended)
MASK_PROB=1.0  →  always audio-only      (pure audio baseline)
```

---

### Q: Exactly where is P4 (audio-only) implemented in code?

P4 is implemented by forcing the face branch to be masked at inference.

In `scripts/eval_p3.py`, the P4 pass does:

```python
mask_all = torch.ones(1, dtype=torch.bool, device=device)
_, logits_p4 = model(face_batch, wave_batch, mask_faces=mask_all)
```

This means `mask_faces=True` for every sample in that pass.

In `scripts/model.py`, `FusionModel.forward()` handles this by replacing the
real face embedding with a learned token:

```python
mask_token = self.face_mask_token.unsqueeze(0).expand(B, -1)
f_emb = torch.where(mask_faces.unsqueeze(1), mask_token, f_emb)
```

So P4 is effectively:

`fused = [face_mask_token || audio_embedding]`

instead of:

`fused = [real_face_embedding || audio_embedding]`

---

### Q: How is this missing-face case learned during training?

In training (`scripts/train.py`), the model is called as:

```python
embeddings, logits = model(face_frames, waveforms)
```

No `mask_faces` tensor is passed manually. Inside `FusionModel.forward()`, when
the model is in training mode, it samples face masks automatically:

```python
if mask_faces is None and self.training:
	mask_faces = torch.rand(B, device=device) < self.mask_prob
```

`self.mask_prob` comes from `.env` as `MASK_PROB` (default `0.3`).

Result: during training, a fraction of samples are audio-only (face replaced by
`face_mask_token`), so P4/P6 behavior is learned before evaluation.

Note: the current eval script still loads face files for batching/shape
consistency, then masks face embeddings for P4. So it is "face ignored" rather
than "face files not required".

---

### Q: How do I train a heavy model (vit_large + wavlm_large) without running out of GPU RAM?

Three complementary techniques are enabled via `.env`:

| Setting | What it does | GPU RAM saved | Cost |
|---|---|---|---|
| `USE_AMP=true` | fp16 mixed precision (activations + buffers) | ~50% | negligible |
| `GRAD_CHECKPOINT=true` | recompute activations during backward | ~50% of activations | ~25% extra compute |
| `CPU_OFFLOAD_OPTIM=true` | Adam m/v states live in CPU RAM | ~2× model size | H2D/D2H per step |
| `GRAD_ACCUM_STEPS=4` | 4 micro-steps per weight update | proportional to step count | none |

**Recommended config for `vit_large + wavlm_large` on a 16 GB GPU:**

```ini
FACE_ENCODER=vit_large
AUDIO_ENCODER=wavlm_large
BATCH_SIZE=8
GRAD_ACCUM_STEPS=4    # effective batch = 32
USE_AMP=true
GRAD_CHECKPOINT=true
CPU_OFFLOAD_OPTIM=false   # enable if still OOM
```

**For 8 GB GPU, also set:**
```ini
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
CPU_OFFLOAD_OPTIM=true
```

**How it works internally:**

- **AMP (`USE_AMP`)** — wraps the forward pass in `torch.autocast(device_type="cuda")`. PyTorch automatically casts eligible ops to `float16`, reducing activation memory ~50%. A `GradScaler` multiplies the loss before backward to prevent fp16 underflow, then unscales before the gradient clip.

- **Gradient checkpointing (`GRAD_CHECKPOINT`)** — calls `backbone.gradient_checkpointing_enable()` on both ViT and WavLM. Instead of storing all intermediate activations for backward, only the layer inputs (checkpoints) are stored; activations are recomputed on demand. For 24-layer ViT-Large this frees hundreds of MB.

- **CPU optimizer offload (`CPU_OFFLOAD_OPTIM`)** — after every weight update, Adam's first-moment (m) and second-moment (v) tensors are moved from GPU to CPU. Before the next update they are moved back. For `vit_large + wavlm_large` (~1.3 B params total), this frees ~10 GB of optimizer state.

- **Gradient accumulation (`GRAD_ACCUM_STEPS`)** — the loss is divided by `N` and `backward()` is called `N` times before a single `optimizer.step()`. This keeps effective batch size constant while using only `1/N` of the per-step activation memory.
