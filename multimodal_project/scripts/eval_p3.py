"""
eval_p3.py — Evaluate the POLY-SIM model under P3 and P4 protocols.

─── What is an Evaluation Protocol? ────────────────────────────────────────
The POLY-SIM challenge defines four evaluation settings (P3–P6) that differ
along two axes:

  Axis 1 — MODALITY:  what inputs does the model see at test time?
    • Face + Audio  (both modalities available)
    • Audio only    (face is missing — camera failure, occlusion, privacy)

  Axis 2 — LANGUAGE:  what language is the spoken audio?
    • Same language as training  (English → English)
    • Cross-lingual shift        (trained on English, tested on Urdu)

This script covers the two English protocols:

  P3 — Face + Audio, English → English
       Both modalities are available.  This is the "easy" upper-bound
       condition; the full model gets to use what it was designed for.

  P4 — Audio only,   English → English
       The face is deliberately withheld to simulate a real-world failure.
       The model must rely purely on the speaker's voice.  Good P4 accuracy
       shows the model has learned robust audio-only speaker representations.

Both protocols are evaluated in a single forward pass because re-using the
audio computation saves time: only the face branch changes (real face vs.
a learned MASK token).

─── Output CSV ──────────────────────────────────────────────────────────────
Columns in submission_v1_test_English_English.csv:

  key      — Sample identifier (file stem, e.g. "00042").  The evaluator
             uses this to match predictions back to ground-truth labels.

  p3       — Numeric speaker ID predicted by the model when BOTH face and
             audio are available (Protocol 3).  E.g. 42 means "id0042".

  conf_p3  — Softmax confidence for the P3 prediction (0–1).  A value close
             to 1 means the model is very certain; near 1/num_speakers means
             it is essentially guessing.

  p4       — Numeric speaker ID predicted using audio ONLY (Protocol 4).
             The face branch is replaced by a learnable MASK token.

  conf_p4  — Softmax confidence for the P4 prediction (0–1).

─── Configuration ───────────────────────────────────────────────────────────
All values are read from the .env file at the project root (same as train.py).
Two extra keys are recognised here:

    EVAL_CHECKPOINT   path to checkpoint  (default: checkpoints/best.pt)
    DATA_TEST_ROOT    path to test root   (default: Data_Test)

Usage:
    python scripts/eval_p3.py

Test-data layout expected:
    <DATA_TEST_ROOT>/
        faces/
            English/
                00000.jpg  00001.jpg  …
        voices/
            English/
                00000.wav  00001.wav  …

    Face file with stem N is paired with voice file with the same stem N.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

# Allow running from repo root: python scripts/eval_p3.py
sys.path.insert(0, str(Path(__file__).parent))

from model import AudioEncoder, FaceEncoder, FusionModel


TARGET_SR: int = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    """
    Parse the project-root .env file into a plain dictionary.

    Why use a .env file instead of CLI arguments?
    ─────────────────────────────────────────────
    Keeping all knobs (paths, model names, hyper-parameters) in one text file
    makes experiments reproducible: you can commit the .env alongside the
    checkpoint and know *exactly* what settings produced a given result,
    without having to remember long command-line invocations.

    Why check for os.environ as a fallback (see _get below)?
    ─────────────────────────────────────────────────────────
    In CI/CD pipelines or Docker containers, secrets and paths are injected as
    environment variables rather than written to disk.  The fallback ensures
    the script works in both interactive and automated environments.
    """
    env: dict[str, str] = {}
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def _get(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, os.environ.get(key, default))


def _load_labels_csv(path: Path) -> dict[str, int]:
    """Load optional ground-truth labels for local eval metrics.

    Expected CSV columns:
      - key
      - one of: label, speaker, speaker_id, target, p3
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {path}")

        label_col = None
        for candidate in ("label", "speaker", "speaker_id", "target", "p3"):
            if candidate in reader.fieldnames:
                label_col = candidate
                break

        if "key" not in reader.fieldnames or label_col is None:
            raise ValueError(
                "Labels CSV must contain 'key' and one of "
                "['label', 'speaker', 'speaker_id', 'target', 'p3']"
            )

        labels: dict[str, int] = {}
        for row in reader:
            k = (row.get("key") or "").strip()
            v = (row.get(label_col) or "").strip()
            if not k or not v:
                continue
            labels[k] = int(v)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_eval_transform() -> transforms.Compose:
    """
    Build a deterministic image preprocessing pipeline for evaluation.

    Why no augmentation (unlike training)?
    ───────────────────────────────────────
    During training, augmentations (random flip, colour jitter, random crop)
    act as regularisation — they artificially enlarge the dataset and force the
    model to learn features that are invariant to those perturbations.

    At evaluation time we want a *single, reproducible* prediction per sample.
    Applying random transforms would make the result non-deterministic, meaning
    you could get different accuracy scores on the same data just by re-running
    the script.  So we use only the transforms that are always required:

      Resize(224×224)   — All ViT/ResNet HuggingFace models expect this size.
      ToTensor()        — Convert PIL image (H×W×C, uint8) to float32 [0,1].
      Normalize(μ,σ)    — ImageNet channel statistics.  The backbone was
                          pre-trained on ImageNet; applying the same
                          normalisation keeps the input distribution the model
                          "expects" from pre-training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_and_pad_audio(wav_path: Path, max_samples: int) -> torch.Tensor:
    """
    Load a WAV file and return a fixed-length mono waveform tensor.

    Steps and their reasons:
    ────────────────────────
    1. torchaudio.load(wav_path)
       Reads the file into a float32 tensor [channels, samples] and reports
       the original sample rate.  We must check the sample rate because
       different recordings may have been captured at different rates.

    2. Resample to TARGET_SR (16 000 Hz) if needed
       WavLM and other speech transformers were pre-trained on 16 kHz audio.
       Feeding audio at the wrong sample rate would make every 1-second
       window appear to represent a different duration of speech, breaking the
       model's temporal assumptions.

    3. waveform.mean(0)  — stereo → mono
       Most speaker-recognition models expect a single channel.  Averaging
       channels instead of discarding one preserves the full signal energy.

    4. Zero-pad short clips
       All samples in a batch must have the same length for the GPU to process
       them in parallel (tensors must be rectangular).  Silent zero-padding at
       the end is standard practice; the model learns to ignore trailing zeros.

    5. Centre-crop long clips (deterministic at eval time)
       Training used random crop for data augmentation.  At evaluation we take
       the centre so every run produces the exact same crop and results are
       reproducible.  The centre of a recording typically captures the most
       informative speech (away from silence at the start/end).
    """
    waveform, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform = waveform.mean(0)  # stereo → mono  [T]
    T = waveform.shape[0]
    if T < max_samples:
        waveform = F.pad(waveform, (0, max_samples - T))
    elif T > max_samples:
        start = (T - max_samples) // 2  # centre crop — deterministic
        waveform = waveform[start: start + max_samples]
    return waveform  # [max_samples]


def load_face_frames(img_path: Path, transform: transforms.Compose, k: int) -> torch.Tensor:
    """
    Load a single face image and return it tiled k times as a frame stack.

    Why does the model expect K frames instead of one?
    ───────────────────────────────────────────────────
    During training each sample was a video clip, so the face branch received
    K independently sampled frames and averaged their embeddings to produce a
    more stable face representation (temporal averaging reduces the effect of
    blinks, head turns, and partial occlusions).

    At test time the dataset provides one representative image per sample
    instead of a full video.  We tile that single image K times so the tensor
    shape [K, 3, H, W] matches what the model's _encode_faces() method expects.
    The K-frame average then collapses to the embedding of that one image,
    which is the correct behaviour — no information is fabricated.

    Why .convert("RGB")?
    ─────────────────────
    Some JPEG files are saved in CMYK or greyscale mode.  Converting to RGB
    unconditionally guarantees the tensor always has exactly 3 channels,
    preventing shape mismatches inside the face encoder.
    """
    img = Image.open(img_path).convert("RGB")
    frame = transform(img)                           # [3, 224, 224]
    return frame.unsqueeze(0).expand(k, -1, -1, -1)  # [k, 3, 224, 224]


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, num_speakers: int) -> FusionModel:
    """
    Reconstruct the FusionModel architecture using the config stored in the checkpoint.

    Why read the architecture from the checkpoint instead of from .env?
    ───────────────────────────────────────────────────────────────────
    The checkpoint's "cfg" key was written by train.py at save time and
    describes exactly the architecture whose weights were saved.  If we read
    from .env instead, a user who changed .env after training (e.g. to try a
    different backbone) would accidentally build the wrong architecture and
    then fail to load the weights — a hard-to-debug silent mismatch.

    Why pretrained=False when building encoders here?
    ──────────────────────────────────────────────────
    We are about to overwrite every weight with model.load_state_dict(ckpt).
    Setting pretrained=False skips the unnecessary download of HuggingFace
    pre-trained weights, saving bandwidth and startup time.  The final weights
    are those from the checkpoint, not from HuggingFace Hub.
    """
    embed_dim = int(cfg.get("embed_dim", 256))
    face_enc = FaceEncoder(
        backbone=cfg.get("face_encoder", "vit_base"),
        embed_dim=embed_dim,
        pretrained=False,
    )
    audio_enc = AudioEncoder(
        backbone=cfg.get("audio_encoder", "wavlm_base"),
        embed_dim=embed_dim,
        pretrained=False,
    )
    return FusionModel(
        face_encoder=face_enc,
        audio_encoder=audio_enc,
        num_speakers=num_speakers,
        embed_dim=embed_dim,
        mask_prob=float(cfg.get("mask_prob", 0.3)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    env = _load_env()

    # ── Resolve paths ────────────────────────────────────────────────────────
    repo_root = Path(__file__).parent.parent

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    ckpt_path       = _abs(_get(env, "EVAL_CHECKPOINT",  "checkpoints/best.pt"))
    data_train_root = _abs(_get(env, "DATA_ROOT",         "Data"))
    data_test_root  = _abs(_get(env, "DATA_TEST_ROOT",    "Data_Test"))
    output_dir      = _abs(_get(env, "OUTPUT_DIR",        "checkpoints"))
    eval_lang       = _get(env, "EVAL_LANG", "English").strip() or "English"
    eval_set       = _get(env, "EVAL_SET", "dev").strip()
    eval_csv_name   = _get(env, "EVAL_OUTPUT_CSV",
        f"submission_v1_{eval_set}_English_{eval_lang}.csv",).strip()
    max_audio_sec   = float(_get(env, "MAX_AUDIO_SEC",    "6.0"))
    eval_tb_dir     = _abs(_get(env, "EVAL_TENSORBOARD_DIR", "runs/eval"))
    eval_labels_csv = _get(env, "EVAL_LABELS_CSV", "").strip()
    eval_ref_csv    = _abs(_get(env, "EVAL_REF_CSV",  "Data_Test/v1_val_English.csv"))
    # Base directory for resolving relative paths found inside EVAL_REF_CSV.
    # The CSV has paths like "val/v1/faces/English/00001.jpg" which are
    # relative to DATA_TEST_ROOT (e.g. Data_Test/).  Override EVAL_CSV_ROOT
    # in .env only if your CSV paths are rooted somewhere else.
    eval_csv_root   = _abs(_get(env, "EVAL_CSV_ROOT", _get(env, "DATA_TEST_ROOT", "Data_Test")))
    auto_device     = "cuda" if torch.cuda.is_available() else "cpu"
    device          = torch.device(_get(env, "DEVICE", auto_device))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Startup banner ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f" POLY-SIM  |  Evaluation  |  P3 + P4 (English → {eval_lang})")
    print("═" * 60)
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Train data     : {data_train_root}")
    print(f"  Test data      : {data_test_root}")
    print(f"  Device         : {device}")
    print(f"  Eval language  : {eval_lang}")
    print(f"  Eval CSV name  : {eval_csv_name}")
    print(f"  TB dir         : {eval_tb_dir}")
    print(f"  Labels CSV     : {eval_labels_csv or 'not set'}")
    print(f"  Ref CSV        : {eval_ref_csv}")
    print(f"  CSV root       : {eval_csv_root}")
    print("═" * 60 + "\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    # map_location=device ensures weights are loaded directly onto the target
    # device (CPU or GPU).  Without this, torch.load would first load onto the
    # device where training happened, then implicitly move — which can cause an
    # out-of-memory error on a machine with a smaller GPU than the training host.
    #
    # weights_only=False is needed because the checkpoint also stores non-tensor
    # objects (the Python dict cfg, optimizer state strings, etc.).  Setting it
    # to True would restrict loading to tensors alone and raise an error here.
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Set EVAL_CHECKPOINT in .env or copy a checkpoint to checkpoints/best.pt"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg: dict = ckpt.get("cfg", {})

    # Resolve architecture config with env fallbacks so model reconstruction
    # matches the checkpoint even when older checkpoints miss some cfg fields.
    resolved_cfg: dict[str, object] = dict(ckpt_cfg)
    resolved_cfg.setdefault("face_encoder", _get(env, "FACE_ENCODER", "vit_base"))
    resolved_cfg.setdefault("audio_encoder", _get(env, "AUDIO_ENCODER", "wavlm_base"))
    resolved_cfg.setdefault("embed_dim", int(_get(env, "EMBED_DIM", "256")))
    resolved_cfg.setdefault("mask_prob", float(_get(env, "MASK_PROB", "0.3")))

    face_encoder_name = str(resolved_cfg["face_encoder"])
    audio_encoder_name = str(resolved_cfg["audio_encoder"])
    print(
        f"  Loaded epoch {ckpt.get('epoch', '?')}  "
        f"(best train acc {ckpt.get('best_acc', 0.0):.2f}%)"
    )
    print(f"  Face/Image encoder    : {face_encoder_name}")
    print(f"  Audio encoder         : {audio_encoder_name}")

    # ── Reconstruct speaker list from training data ───────────────────────────
    # The model's classifier head has one output neuron per speaker.  During
    # training, MAVCelebDataset assigned labels 0, 1, 2, … by sorting the
    # speaker folder names alphabetically.  We must rebuild that exact same
    # mapping here; even a single extra or missing folder would shift every
    # class index and make all predictions wrong.
    #
    # We read from the *training* directory (not test) because the label space
    # was defined by what was seen during training.  The test set does not
    # enumerate speakers — ground-truth labels are on the challenge server.
    #
    # idx_to_num converts the internal 0-based class index back to the human-
    # readable numeric ID used in the submission CSV (e.g. class 5 → speaker 9
    # if id0004/id0007/id0008 were absent from the dataset).
    faces_train = data_train_root / "faces"
    if not faces_train.is_dir():
        raise FileNotFoundError(f"Training faces directory not found: {faces_train}")
    speakers: list[str] = sorted(p.name for p in faces_train.iterdir() if p.is_dir())
    num_speakers = len(speakers)
    # Map class index → numeric speaker number (e.g. "id0042" → 42)
    idx_to_num: dict[int, int] = {
        i: int(s.lstrip("id").lstrip("0") or "0")
        for i, s in enumerate(speakers)
    }
    num_to_idx: dict[int, int] = {num: idx for idx, num in idx_to_num.items()}
    print(f"  Speakers (from train)  : {num_speakers}")

    # ── Build and load model ──────────────────────────────────────────────────
    model = build_model(resolved_cfg, num_speakers)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    # model.eval() switches off dropout and batch-normalisation running stats.
    # During training these add noise intentionally (regularisation).  At
    # evaluation we want fully deterministic, noise-free predictions.
    model.eval()
    print(f"  Image encoder          : {face_encoder_name}")
    print(f"  Audio encoder          : {audio_encoder_name}")
    print(f"  Model architecture     : {face_encoder_name} + {audio_encoder_name}")
    print(f"  Embed dim              : {resolved_cfg.get('embed_dim','?')}")

    # ── Load test pairs from reference CSV ──────────────────────────────────────
    # The reference CSV (v1_val_English.csv) defines which samples to evaluate.
    # Each row contains:
    #   key    — unique sample identifier used in the submission CSV
    #   faces  — relative path to the face image  (resolved from eval_csv_root)
    #   voices — relative path to the audio file  (resolved from eval_csv_root)
    #
    # Path resolution strategy:
    #   1. Resolve the path from eval_csv_root (configurable via EVAL_CSV_ROOT).
    #   2. If still not found, try resolving from DATA_TEST_ROOT as a fallback
    #      (handles the common case where the CSV was generated with a leading
    #      'val/' prefix but the actual data lives directly under Data_Test/).
    #   Rows whose face or voice file cannot be located are skipped with a warning.
    if not eval_ref_csv.exists():
        raise FileNotFoundError(
            f"Reference CSV not found: {eval_ref_csv}\n"
            "Set EVAL_REF_CSV in .env to point to the correct file."
        )

    def _resolve_csv_path(rel: str) -> Path | None:
        """Resolve a CSV-relative path against eval_csv_root.

        The CSV paths are relative to DATA_TEST_ROOT (e.g. val/v1/faces/…).
        eval_csv_root defaults to DATA_TEST_ROOT so DATA_TEST_ROOT/val/v1/…
        is tried first.  Two additional fallbacks are tried in case the data
        layout differs: absolute repo-root resolution and a stripped-prefix
        resolution that drops the first path component (val/).
        """
        candidates = [
            eval_csv_root / rel,
            repo_root / rel,
            # Drop leading component (e.g. 'val/') and retry under csv_root
            eval_csv_root / Path(*Path(rel).parts[2:]) if len(Path(rel).parts) > 1 else None,
        ]
        for c in candidates:
            if c is not None and c.exists():
                return c
        return None

    test_pairs: list[tuple[str, Path, Path]] = []  # (key, face_path, voice_path)
    skipped_missing = 0

    with open(eval_ref_csv, newline="") as _csv_f:
        _reader = csv.DictReader(_csv_f)
        for _row in _reader:
            _key       = (_row.get("key")    or "").strip()
            _face_rel  = (_row.get("faces")  or "").strip()
            _voice_rel = (_row.get("voices") or "").strip()
            if not _key or not _face_rel or not _voice_rel:
                continue
            _face_path  = _resolve_csv_path(_face_rel)
            _voice_path = _resolve_csv_path(_voice_rel)
            if _face_path is None or _voice_path is None:
                skipped_missing += 1
                if _face_path is None:
                    print(f"  ⚠ missing face  [{_key}]: Data_Test/{_face_rel}")
                if _voice_path is None:
                    print(f"  ⚠ missing voice [{_key}]: Data_Test/{_voice_rel}")
                continue
            test_pairs.append((_key, _face_path, _voice_path))

    if skipped_missing:
        print(f"  ⚠ {skipped_missing} row(s) skipped — file(s) not found on disk")
    if not test_pairs:
        raise ValueError(
            f"No valid face/voice pairs found in {eval_ref_csv}\n"
            f"Check EVAL_CSV_ROOT (currently: {eval_csv_root}) and the paths inside the CSV."
        )
    print(f"  Test samples (valid)   : {len(test_pairs)}")
    print()

    # ── Inference ─────────────────────────────────────────────────────────────
    face_tf      = make_eval_transform()
    max_samples  = int(max_audio_sec * TARGET_SR)
    k_faces      = int(resolved_cfg.get("k_faces", _get(env, "K_FACES", "4")))

    rows: list[dict] = []

    # Optional local metrics if ground-truth labels are available.
    gt_labels: dict[str, int] | None = None
    if eval_labels_csv:
        labels_path = Path(eval_labels_csv)
        labels_path = labels_path if labels_path.is_absolute() else repo_root / labels_path
        if not labels_path.exists():
            raise FileNotFoundError(f"EVAL_LABELS_CSV not found: {labels_path}")
        gt_labels = _load_labels_csv(labels_path)
        print(f"  Loaded GT labels       : {len(gt_labels)} from {labels_path}")

    eval_step = int(ckpt.get("epoch", 0))
    p3_ce_sum = p4_ce_sum = 0.0
    p3_correct = p4_correct = 0
    gt_count = 0

    # torch.no_grad() tells PyTorch NOT to build a computation graph during the
    # forward pass.  During training, the graph is needed to compute gradients
    # (backpropagation).  At evaluation there are no gradients to compute, so
    # building the graph would waste memory and time.  This context manager
    # typically reduces GPU memory usage by ~30–50% during inference.
    with torch.no_grad():
        for key, face_path, voice_path in tqdm(test_pairs, desc="Evaluating", unit="sample"):
            face_tensor = load_face_frames(
                face_path, face_tf, k_faces
            )  # [K, 3, 224, 224]
            waveform = load_and_pad_audio(
                voice_path, max_samples
            )  # [T]

            # unsqueeze(0) adds the batch dimension B=1.  PyTorch models always
            # process data in batches [B, ...].  A single sample has no batch
            # dimension on its own, so we insert it manually before forwarding.
            face_batch = face_tensor.unsqueeze(0).to(device)  # [1, K, 3, 224, 224]
            wave_batch = waveform.unsqueeze(0).to(device)     # [1, T]

            # ── P3: face + audio (mask_faces = all-False) ─────────────────
            # Passing a False mask means "do NOT replace the face embedding".
            # The model uses the real face features alongside the audio,
            # replicating the full-modality condition seen during training.
            mask_none = torch.zeros(1, dtype=torch.bool, device=device)
            _, logits_p3 = model(face_batch, wave_batch, mask_faces=mask_none)

            # softmax converts raw logits (unbounded scores) into probabilities
            # that sum to 1.0 over the speaker dimension.  argmax then picks
            # the speaker with the highest probability as the prediction.
            probs_p3    = torch.softmax(logits_p3, dim=1)
            pred_p3_idx = int(probs_p3.argmax(dim=1).item())
            conf_p3     = float(probs_p3[0, pred_p3_idx].item())

            # ── P4: audio only (mask_faces = all-True) ─────────────────────
            # Passing an all-True mask tells FusionModel.forward() to replace
            # the face embedding with the learned face_mask_token parameter.
            # This token was trained alongside the model during the modality-
            # dropout phase, so the fusion head has seen this "missing face"
            # input before and knows how to handle it using audio alone.
            mask_all = torch.ones(1, dtype=torch.bool, device=device)
            _, logits_p4 = model(face_batch, wave_batch, mask_faces=mask_all)
            probs_p4    = torch.softmax(logits_p4, dim=1)
            pred_p4_idx = int(probs_p4.argmax(dim=1).item())
            conf_p4     = float(probs_p4[0, pred_p4_idx].item())

            rows.append({
                "key": key,
                "p3":  idx_to_num[pred_p3_idx],
                "p4":  idx_to_num[pred_p4_idx],
            })

            if gt_labels is not None and key in gt_labels:
                gt_num = int(gt_labels[key])
                if gt_num in num_to_idx:
                    gt_idx = num_to_idx[gt_num]
                    gt_tensor = torch.tensor([gt_idx], dtype=torch.long, device=device)
                    p3_ce_sum += F.cross_entropy(logits_p3, gt_tensor, reduction="sum").item()
                    p4_ce_sum += F.cross_entropy(logits_p4, gt_tensor, reduction="sum").item()
                    p3_correct += int(pred_p3_idx == gt_idx)
                    p4_correct += int(pred_p4_idx == gt_idx)
                    gt_count += 1

    # ── Write submission CSV ──────────────────────────────────────────────────
    # Path behavior:
    #   - absolute EVAL_OUTPUT_CSV: use as-is
    #   - relative with directory (e.g. checkpoints/foo.csv): resolve from repo root
    #   - filename only (e.g. foo.csv): place under OUTPUT_DIR
    csv_cfg_path = Path(eval_csv_name)
    if csv_cfg_path.is_absolute():
        csv_path = csv_cfg_path
    elif csv_cfg_path.parent != Path("."):
        csv_path = repo_root / csv_cfg_path
    else:
        csv_path = output_dir / csv_cfg_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        # writer = csv.DictWriter(f, fieldnames=["key", "p3", "conf_p3", "p4", "conf_p4"])
        writer = csv.DictWriter(f, fieldnames=["key", "p3", "p4"])
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    unique_p3 = len({r["p3"] for r in rows})
    unique_p4 = len({r["p4"] for r in rows})
    print(f"\n{'═'*60}")
    print(f"  Evaluation complete")
    print(f"  Samples evaluated    : {len(rows)}")
    print(f"  P3 unique speakers   : {unique_p3}")
    print(f"  P4 unique speakers   : {unique_p4}")
    print(f"  CSV saved →  {csv_path}")

    # ── TensorBoard logging ───────────────────────────────────────────────
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(eval_tb_dir))
        try:
            tb_writer.add_scalar("Eval/Samples", len(rows), eval_step)
            tb_writer.add_scalar("Eval/P3_UniqueSpeakers", unique_p3, eval_step)
            tb_writer.add_scalar("Eval/P4_UniqueSpeakers", unique_p4, eval_step)

            if gt_count > 0:
                p3_acc = p3_correct / gt_count * 100.0
                p4_acc = p4_correct / gt_count * 100.0
                p3_loss = p3_ce_sum / gt_count
                p4_loss = p4_ce_sum / gt_count

                tb_writer.add_scalar("Eval/P3_Accuracy", p3_acc, eval_step)
                tb_writer.add_scalar("Eval/P4_Accuracy", p4_acc, eval_step)
                tb_writer.add_scalar("Eval/P3_Loss", p3_loss, eval_step)
                tb_writer.add_scalar("Eval/P4_Loss", p4_loss, eval_step)

                print(f"  Labeled samples      : {gt_count}")
                print(f"  P3 acc / loss        : {p3_acc:.2f}% / {p3_loss:.4f}")
                print(f"  P4 acc / loss        : {p4_acc:.2f}% / {p4_loss:.4f}")
            else:
                print("  Labeled metrics      : skipped (no EVAL_LABELS_CSV matches)")

            print(f"  TensorBoard saved →  {eval_tb_dir}")
        finally:
            tb_writer.flush()
            tb_writer.close()
    else:
        print("  TensorBoard          : not available (pip install tensorboard)")

    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
