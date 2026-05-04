# Unified Evaluation Guide

This document explains how to evaluate all four POLY-SIM challenge protocols
using a single script.

## Protocols Covered

The unified evaluator runs:

- P3: face + audio, English -> English
- P4: audio-only, English -> English
- P5: face + audio, English -> Urdu
- P6: audio-only, English -> Urdu

All protocols are computed from one checkpoint in one run.

## Script

Run:

```bash
python scripts/eval_all.py
```

## How P4 and P6 Are Implemented

P4/P6 are audio-only protocols. The script still loads face images to keep the
input tensor shape compatible with the trained model, then masks the face
embedding in the fusion model:

- Full-modality pass: `mask_faces=False` (P3 or P5)
- Audio-only pass: `mask_faces=True` (P4 or P6)

The learned face mask token is used for masked passes.

## Outputs

The script writes two challenge-format CSV files:

1. EN-EN file with columns: `key,p3,p4`
2. EN-UR file with columns: `key,p5,p6`

Paths are controlled by `.env`:

- `EVAL_OUTPUT_EN_EN`
- `EVAL_OUTPUT_EN_UR`

Path resolution behavior:

- absolute path: used directly
- relative path with directories: resolved from repo root
- filename only: written under `OUTPUT_DIR`

## Required and Optional Environment Variables

Core:

- `EVAL_ALL_CHECKPOINT` (fallback: `EVAL_CHECKPOINT`)
- `DATA_TEST_ROOT`
- `EVAL_LANG_SAME` (default: `English`)
- `EVAL_LANG_CROSS` (default: `Urdu`)
- `EVAL_OUTPUT_EN_EN`
- `EVAL_OUTPUT_EN_UR`
- `DEVICE` (optional override)

Optional local metrics:

- `EVAL_LABELS_SAME_CSV` for P3/P4 local accuracy/loss
- `EVAL_LABELS_CROSS_CSV` for P5/P6 local accuracy/loss

Labels CSV must contain `key` and one of:

- same-language: `label`, `speaker`, `speaker_id`, `target`, or `p3`
- cross-language: `label`, `speaker`, `speaker_id`, `target`, or `p5`

## TensorBoard

Unified eval logs to:

- `EVAL_ALL_TENSORBOARD_DIR` (default: `runs/eval_all`)

View with:

```bash
tensorboard --logdir runs/eval_all
```

If labels are provided, accuracy/loss scalars are logged in addition to sample
and distribution stats.

## Notes

- Speaker ID mapping is rebuilt from `DATA_ROOT/faces` to match training labels.
- If no matched face/audio stems are found for a language subset, evaluation
  fails early with a clear path diagnostic.
- Existing scripts are still available:
  - `scripts/eval_p3.py` for P3/P4 only
  - `scripts/eval_p6.py` for P5/P6 only
