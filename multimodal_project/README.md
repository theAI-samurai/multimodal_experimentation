# POLY-SIM: Polyglot Speaker Identification with Missing Modality

Grand Challenge 2026 — Multimodal speaker identification under missing-modality and cross-lingual conditions.

---

## Dataset: MAV-Celeb

Raw data is located under `Dataset/mavceleb_v1_train/` with the following structure:

```
Dataset/mavceleb_v1_train/
├── faces/
│   └── <speaker_id>/
│       ├── English/
│       │   └── <video_folder>/
│       │       └── *.jpg
│       └── Urdu/
│           └── <video_folder>/
│               └── *.jpg
└── voices/
    └── <speaker_id>/
        ├── English/
        │   └── <video_folder>/
        │       └── *.wav
        └── Urdu/
            └── <video_folder>/
                └── *.wav
```

---

## Prepared English-Only Flat Split

Script: `scripts/prepare_english_flat.py`

Produces a flat directory under `Dataset/mavceleb_v1_train_english_flat/` containing only English faces and voices, with filenames that encode the speaker ID, language, video folder, and original filename:

```
Dataset/mavceleb_v1_train_english_flat/
└── train/
    ├── faces/
    │   └── <speaker_id>_eng_<video_folder>_<image_filename>
    │       e.g. id0001_eng_A8Ch7gJalls_0000100.jpg
    └── voices/
        └── <speaker_id>_eng_<video_folder>_<wav_filename>
            e.g. id0001_eng_A8Ch7gJalls_00000.wav
```

**Stats:** 64 speakers · 89,621 face images · 5,786 audio clips

### Usage

```bash
# Default: create symbolic links (fast, no extra disk usage)
python scripts/prepare_english_flat.py

# Custom paths
python scripts/prepare_english_flat.py \
    --source-root Dataset/mavceleb_v1_train \
    --output-root Dataset/mavceleb_v1_train_english_flat

# Copy files instead of symlinking
python scripts/prepare_english_flat.py --copy
```

---

## English-Only Training Data (Data/)

A clean copy of only the English faces and voices, ready for training.
Created directly from `Dataset/mavceleb_v1_train/` — Urdu data excluded, no symlinks.

```
Data/
├── faces/
│   └── <speaker_id>/
│       └── <video_folder>/
│           └── *.jpg
└── voices/
    └── <speaker_id>/
        └── <video_folder>/
            └── *.wav
```

**Stats:** 64 speakers · English faces + English audio only

To recreate:

```bash
for speaker in Dataset/mavceleb_v1_train/faces/*/; do
    speaker_id=$(basename "$speaker")
    mkdir -p "Data/faces/${speaker_id}"
    cp -r "${speaker}English/." "Data/faces/${speaker_id}/"
done

for speaker in Dataset/mavceleb_v1_train/voices/*/; do
    speaker_id=$(basename "$speaker")
    mkdir -p "Data/voices/${speaker_id}"
    cp -r "${speaker}English/." "Data/voices/${speaker_id}/"
done
```

---

## Dataset Analysis (English split)

### Scale & Structure

| | Count |
|---|---|
| Speakers | 64 |
| Face images | 89,621 |
| Audio clips | 5,786 |
| Avg face frames per audio clip | 15.5× |

Each video folder is shared between `faces/` and `voices/` — face frames and audio clips are naturally paired at the **video level**.

### Per-Speaker Distribution

```
Faces:  min=94  (id0031)   max=8,196  (id0039)   mean=1,400
Audio:  min=9   (id0031)   max=373    (id0039)    mean=90
```

~6–7× imbalance between the smallest and largest speakers.

### Face-to-Audio Ratio

```
Per speaker: min=5.9×   max=29.7×   mean=15.7×
```

Face frames are sampled at ~25 fps; each `.wav` covers ~1–2 s. A single audio clip has 6–30 matchable face frames from the same video.

---

## Recommended Training Mechanism

### 1. Pair Construction (video-level)

For each audio clip `voices/<id>/<video>/<clip>.wav`:
- **Positive face**: randomly sample K frames from `faces/<id>/<video>/` (same video = same scene)
- **Negative face**: sample from a different speaker's video

### 2. Encoder Architecture

```
Face branch:  FaceNet / ViT-S       → 512-d embedding
Audio branch: ECAPA-TDNN / WavLM    → 512-d embedding
Both projected to a shared 256-d embedding space
```

### 3. Training Objectives

```
L_total = L_CE + λ1·L_contrastive + λ2·L_ortho
```

| Loss | Purpose |
|---|---|
| `L_CE` | Softmax cross-entropy over 64 speakers — direct ID supervision |
| `L_contrastive` | NT-Xent / SupCon — pull same-speaker face+audio embeddings together |
| `L_ortho` | Orthogonality penalty on different-speaker embeddings (FOP-style) |

### 4. Modality Dropout

At training time, replace the face embedding with a learnable `[MASK]` token with probability **p=0.3**.
This forces the audio branch to produce independently identifiable embeddings, simulating the test condition (P4/P6 protocols — audio only, no face).

```
p=0.0  →  always face+audio   (P3/P5)
p=0.3  →  30% audio-only      (trains P4/P6 robustness)
```

### 5. Class-Balanced Sampling

Use a `WeightedRandomSampler` with per-speaker weight `w_i = 1 / n_audio_clips_i` to counter the 6–7× speaker imbalance. Sample at the audio clip level so every epoch sees each speaker roughly equally.

### 6. Face Sampling Strategy

Per training step, for a given audio clip:
- Sample **K=4 face frames** from the same video (random temporal jitter)
- Average-pool the K face embeddings → single robust face representation

### Full Pipeline Summary

```
For each batch:
  1. Sample balanced audio clips (WeightedRandomSampler)
  2. For each clip → sample K=4 face frames from same video
  3. Encode faces → avg-pool → f_emb (256-d)
  4. Encode audio → a_emb (256-d)
  5. With p=0.3, replace f_emb with learned [MASK] token
  6. Fuse: z = proj(concat(f_emb, a_emb))  OR  z = a_emb  (if masked)
  7. Compute L_CE + L_contrastive + L_ortho
  8. Backprop through both branches

At test time (audio-only, Urdu):
  wav → WavLM encoder → a_emb → project → nearest-neighbour to speaker prototypes
```

### Priority Improvements for Challenge Protocols P4 / P6

| Issue | Fix |
|---|---|
| Audio-only at test (P4/P6) | Modality dropout during training |
| Urdu at test (P5/P6) | Replace ECAPA-TDNN with WavLM-Large (multilingual) |
| Speaker imbalance | Weighted sampler per speaker |
| Noisy face frames | K=4 frame average pooling |
| Face/audio domain gap | Orthogonality loss + SupCon across modalities |

---

## Training Scripts

### File Overview

| File | Purpose |
|---|---|
| [scripts/dataset.py](scripts/dataset.py) | `MAVCelebDataset` — video-level pairing, K-frame face sampling, audio crop/pad |
| [scripts/model.py](scripts/model.py) | `FaceEncoder`, `AudioEncoder`, `FusionModel` with modality dropout |
| [scripts/losses.py](scripts/losses.py) | `SupConLoss` (supervised contrastive), `OrthogonalityLoss` (FOP-style) |
| [scripts/train.py](scripts/train.py) | Training loop with balanced sampler, backbone freeze warmup, cosine LR |
| [requirements.txt](requirements.txt) | All Python dependencies |

### Install

```bash
pip install -r requirements.txt
```

### Model Choices

All backbones are loaded automatically from HuggingFace Hub via `transformers`.

**Face encoder** (`FACE_ENCODER`):

| Value | HuggingFace model | Notes |
|---|---|---|
| `vit_base` *(default)* | `google/vit-base-patch16-224` | [CLS] token, strong general features |
| `vit_large` | `google/vit-large-patch16-224` | Higher capacity |
| `resnet50` | `microsoft/resnet-50` | Lightweight, fast baseline |

**Audio encoder** (`AUDIO_ENCODER`):

| Value | HuggingFace model | Notes |
|---|---|---|
| `wavlm_base` *(default)* | `microsoft/wavlm-base` | Balanced, ~95M params |
| `wavlm_large` | `microsoft/wavlm-large` | Best cross-lingual (P5/P6), ~317M params |
| `wav2vec2` | `facebook/wav2vec2-base` | Strong English baseline |
| `unispeech_sv` | `microsoft/unispeech-sat-base-plus-sv` | Pretrained for speaker verification |

**Recommended combinations**:

| `FACE_ENCODER` | `AUDIO_ENCODER` | Best for | VRAM |
|---|---|---|---|
| `resnet50` | `unispeech_sv` | P3/P4 fast baseline | ~4 GB |
| `vit_base` | `wavlm_base` | P3–P6 balanced | ~8 GB |
| `vit_large` | `wavlm_large` | P5/P6 best cross-lingual | ~16 GB |

### Configure

All hyperparameters are set in `.env` at the project root (already provided).  
Key settings:

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `Data` | Path to prepared English split |
| `FACE_ENCODER` | `vit_base` | `vit_base` \| `vit_large` \| `resnet50` |
| `AUDIO_ENCODER` | `wavlm_base` | `wavlm_base` \| `wavlm_large` \| `wav2vec2` \| `unispeech_sv` |
| `EMBED_DIM` | `256` | Shared embedding dimension |
| `MASK_PROB` | `0.3` | Face modality dropout probability |
| `FREEZE_ENCODERS_EPOCHS` | `5` | Freeze backbones for warmup then unfreeze |
| `LAMBDA_CON` | `0.5` | Weight for SupCon loss |
| `LAMBDA_ORTH` | `0.1` | Weight for orthogonality loss |
| `EPOCHS` | `100` | Total training epochs |
| `BATCH_SIZE` | `16` | Batch size |
| `TRAIN_SPLIT_CSV` | *(empty)* | Optional CSV manifest for fixed training split (overrides random split when set) |
| `VAL_SPLIT_CSV` | *(empty)* | Optional CSV manifest for fixed validation split |
| `VAL_SPLIT` | `0.1` | Fraction of training data held out for validation (0 = off) |
| `OUTPUT_DIR` | `checkpoints` | Where `best.pt` and `last.pt` are saved |
| `RESUME_FROM` | `checkpoints/best.pt` | Resume from checkpoint (leave empty for scratch) |
| `TENSORBOARD_DIR` | `runs` | TensorBoard event file directory |
| `TB_LOG_EVERY` | `10` | Write batch-level TensorBoard scalars every N training steps |
| `USE_AMP` | `true` | fp16 mixed-precision |
| `GRAD_CHECKPOINT` | `true` | Gradient checkpointing (~50% activation RAM, ~25% slower) |
| `CPU_OFFLOAD_OPTIM` | `true` | Keep Adam states in CPU RAM |

### Fixed 70/20/10 Split CSVs (Train/Test/Val)

Generate split manifests from `Data/`:

```bash
python scripts/create_data_splits.py \
    --data-root Data \
    --out-dir splits \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --val-ratio 0.1
```

Output files:

- `splits/train_split.csv`
- `splits/test_split.csv`
- `splits/val_split.csv`
- `splits/all_splits.csv`

Each row includes:

- `image_path` (representative frame path)
- `voice_path` (audio clip path)
- `face_dir` (folder of all frames used for K-frame sampling)

To train with fixed splits, set in `.env`:

```bash
TRAIN_SPLIT_CSV=splits/train_split.csv
VAL_SPLIT_CSV=splits/val_split.csv
```

`test_split.csv` is held out for your later test-only evaluation.

### Run Training

```bash
# Edit .env to configure, then:
python scripts/train.py
```

The script will print a startup banner showing all active settings, then train  
 while logging to both the console and TensorBoard.

**Resume an interrupted run** — set `RESUME_FROM=checkpoints/last.pt` in `.env` (already the default).

### View TensorBoard

Open a second terminal and run:

```bash
tensorboard --logdir runs
```

Then open **http://localhost:6006** in your browser.

The following scalar groups are logged every epoch:

| TensorBoard tag | Series | Description |
|---|---|---|
| `Loss` | `train` / `val` | Total weighted loss |
| `CE` | `train` / `val` | Cross-entropy loss |
| `ConLoss` | `train` / `val` | Supervised contrastive loss |
| `OrthLoss` | `train` / `val` | Orthogonality loss |
| `Accuracy` | `train` / `val` | Speaker identification accuracy (%) |
| `LearningRate` | — | Current LR (cosine schedule) |

The following training-only tags are logged during the epoch every `TB_LOG_EVERY` steps:

| TensorBoard tag | Description |
|---|---|
| `BatchLoss/train` | Total weighted loss for the current batch |
| `BatchCE/train` | Cross-entropy for the current batch |
| `BatchConLoss/train` | Supervised contrastive loss for the current batch |
| `BatchOrthLoss/train` | Orthogonality loss for the current batch |
| `BatchAccuracy/train` | Accuracy for the current batch |

> **Tip:** set `TENSORBOARD_DIR=runs/exp1` for one experiment and  
> `TENSORBOARD_DIR=runs/exp2` for another, then run  
> `tensorboard --logdir runs` to compare all experiments side-by-side.

### Unified Evaluation (P3/P4/P5/P6)

Run all four protocols in one command:

```bash
python scripts/eval_all.py
```

This single script generates challenge-format outputs:

- EN-EN CSV: `key,p3,p4`
- EN-UR CSV: `key,p5,p6`

### Unified Evaluation Environment Variables

`scripts/eval_all.py` reads these from `.env`:

| Variable | Default | Description |
|---|---|---|
| `EVAL_ALL_CHECKPOINT` | `checkpoints/best.pt` | Checkpoint for unified eval (fallback to `EVAL_CHECKPOINT`) |
| `DATA_TEST_ROOT` | `Data_Test` | Root test directory |
| `EVAL_LANG_SAME` | `English` | Same-language subset used for P3/P4 |
| `EVAL_LANG_CROSS` | `Urdu` | Cross-language subset used for P5/P6 |
| `EVAL_OUTPUT_EN_EN` | `submission_v1_test_English_English.csv` | Output CSV for `key,p3,p4` |
| `EVAL_OUTPUT_EN_UR` | `submission_v1_test_English_Urdu.csv` | Output CSV for `key,p5,p6` |
| `EVAL_ALL_TENSORBOARD_DIR` | `runs/eval_all` | TensorBoard directory for unified eval |
| `EVAL_LABELS_SAME_CSV` | *(empty)* | Optional local labels CSV for P3/P4 metrics |
| `EVAL_LABELS_CROSS_CSV` | *(empty)* | Optional local labels CSV for P5/P6 metrics |

### Unified Eval TensorBoard

To view unified evaluation scalars:

```bash
tensorboard --logdir runs/eval_all
```

The script logs sample counts and unique speaker stats for both protocol pairs,
and logs accuracy/loss when optional labels are supplied.

### Legacy Evaluators

These are still available for protocol-specific runs:

- `python scripts/eval_p3.py` for P3/P4 only
- `python scripts/eval_p6.py` for P5/P6 only

### Training Loss

```
L_total = L_CE  +  λ_con · L_SupCon  +  λ_orth · L_Ortho
```

| Loss | Purpose |
|---|---|
| `L_CE` | Softmax cross-entropy — direct speaker ID supervision |
| `L_SupCon` | Supervised contrastive — cross-modal alignment in shared embedding space |
| `L_Ortho` | Orthogonality penalty on different-speaker embeddings (FOP-style) |

### Key Design Decisions

- **Modality dropout** (`MASK_PROB=0.3`): 30% of steps replace the face embedding with a learned `[MASK]` token, training the audio branch to identify speakers independently. Directly targets P4/P6 protocols (audio-only at test time).
- **Backbone freeze warmup** (`FREEZE_ENCODERS_EPOCHS=5`): projection and fusion heads are trained first, then backbones are unfrozen at 10× lower LR.
- **WeightedRandomSampler**: equalises speaker frequency to counter the 6–7× per-speaker imbalance.
- **K=4 face frames** averaged per step: reduces the impact of noisy or occluded frames.

---

## Challenge Details

See [challenge.md](challenge.md) for the full problem description, evaluation protocols, suggested architectures, and submission format.
