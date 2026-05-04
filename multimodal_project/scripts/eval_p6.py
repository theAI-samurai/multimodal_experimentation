"""
eval_p6.py — Evaluate the POLY-SIM model under P5 and P6 protocols.

Protocols covered
─────────────────
  P5 — Face + Audio, English → Urdu (cross-lingual)
  P6 — Audio only,  English → Urdu (cross-lingual + missing face)

This script mirrors eval_p3.py but targets the Urdu test subset and writes a
cross-lingual submission CSV with columns: key,p5,p6.

Usage:
    python scripts/eval_p6.py

Environment variables (all optional):
    EVAL_P56_CHECKPOINT      (fallback: EVAL_CHECKPOINT, then checkpoints/best.pt)
    DATA_TEST_ROOT           (default: Data_Test)
    EVAL_P56_LANG            (default: Urdu)
    EVAL_P56_OUTPUT_CSV      (default: submission_v1_test_English_Urdu.csv)
    EVAL_P56_TENSORBOARD_DIR (default: runs/eval_p56)
    EVAL_P56_LABELS_CSV      (optional; fallback: EVAL_LABELS_CSV)
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

# Allow running from repo root: python scripts/eval_p6.py
sys.path.insert(0, str(Path(__file__).parent))

from model import AudioEncoder, FaceEncoder, FusionModel


TARGET_SR: int = 16_000


def _load_env() -> dict[str, str]:
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
    """Load optional local GT labels.

    Expected columns:
      - key
      - one of: label, speaker, speaker_id, target, p5
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {path}")

        label_col = None
        for candidate in ("label", "speaker", "speaker_id", "target", "p5"):
            if candidate in reader.fieldnames:
                label_col = candidate
                break

        if "key" not in reader.fieldnames or label_col is None:
            raise ValueError(
                "Labels CSV must contain 'key' and one of "
                "['label', 'speaker', 'speaker_id', 'target', 'p5']"
            )

        labels: dict[str, int] = {}
        for row in reader:
            k = (row.get("key") or "").strip()
            v = (row.get(label_col) or "").strip()
            if not k or not v:
                continue
            labels[k] = int(v)
    return labels


def make_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_and_pad_audio(wav_path: Path, max_samples: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform = waveform.mean(0)  # stereo -> mono [T]
    t = waveform.shape[0]
    if t < max_samples:
        waveform = F.pad(waveform, (0, max_samples - t))
    elif t > max_samples:
        start = (t - max_samples) // 2
        waveform = waveform[start: start + max_samples]
    return waveform


def load_face_frames(img_path: Path, transform: transforms.Compose, k: int) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    frame = transform(img)
    return frame.unsqueeze(0).expand(k, -1, -1, -1)


def build_model(cfg: dict, num_speakers: int) -> FusionModel:
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


def main() -> None:
    env = _load_env()
    repo_root = Path(__file__).parent.parent

    def _abs(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else repo_root / p

    ckpt_raw = _get(env, "EVAL_P56_CHECKPOINT", _get(env, "EVAL_CHECKPOINT", "checkpoints/best.pt"))
    ckpt_path = _abs(ckpt_raw)
    data_train_root = _abs(_get(env, "DATA_ROOT", "Data"))
    data_test_root = _abs(_get(env, "DATA_TEST_ROOT", "Data_Test"))
    output_dir = _abs(_get(env, "OUTPUT_DIR", "checkpoints"))

    eval_lang = _get(env, "EVAL_P56_LANG", "Urdu").strip() or "Urdu"
    eval_csv_name = _get(
        env,
        "EVAL_P56_OUTPUT_CSV",
        "submission_v1_dev_English_Urdu.csv",
    ).strip()

    max_audio_sec = float(_get(env, "MAX_AUDIO_SEC", "6.0"))
    eval_tb_dir = _abs(_get(env, "EVAL_P56_TENSORBOARD_DIR", "runs/eval_p56"))
    eval_labels_csv = _get(env, "EVAL_P56_LABELS_CSV", _get(env, "EVAL_LABELS_CSV", "")).strip()
    eval_ref_csv  = _abs(_get(env, "EVAL_P56_REF_CSV", "Data_Test/v1_val_Urdu.csv"))
    # Base directory for resolving the CSV paths (e.g. val/v1/faces/Urdu/…).
    # Defaults to DATA_TEST_ROOT so Data_Test/val/v1/… is resolved directly.
    eval_csv_root = _abs(_get(env, "EVAL_P56_CSV_ROOT", _get(env, "DATA_TEST_ROOT", "Data_Test")))

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_get(env, "DEVICE", auto_device))

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print(f" POLY-SIM  |  Evaluation  |  P5 + P6 (English -> {eval_lang})")
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

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Set EVAL_P56_CHECKPOINT/EVAL_CHECKPOINT in .env"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg: dict = ckpt.get("cfg", {})

    # Resolve architecture with env fallback for older checkpoints.
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

    faces_train = data_train_root / "faces"
    if not faces_train.is_dir():
        raise FileNotFoundError(f"Training faces directory not found: {faces_train}")

    speakers: list[str] = sorted(p.name for p in faces_train.iterdir() if p.is_dir())
    num_speakers = len(speakers)
    idx_to_num: dict[int, int] = {
        i: int(s.lstrip("id").lstrip("0") or "0")
        for i, s in enumerate(speakers)
    }
    num_to_idx: dict[int, int] = {num: idx for idx, num in idx_to_num.items()}
    print(f"  Speakers (from train) : {num_speakers}")

    model = build_model(resolved_cfg, num_speakers)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"  Model architecture    : {face_encoder_name} + {audio_encoder_name}")
    print(f"  Embed dim             : {resolved_cfg.get('embed_dim', '?')}")

    # ── Load test pairs from reference CSV ──────────────────────────────────
    if not eval_ref_csv.exists():
        raise FileNotFoundError(
            f"Reference CSV not found: {eval_ref_csv}\n"
            "Set EVAL_P56_REF_CSV in .env to point to the correct file."
        )

    def _resolve_csv_path(rel: str) -> Path | None:
        candidates = [
            eval_csv_root / rel,
            repo_root / rel,
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
                    print(f"  ⚠ missing face  [{_key}]: {_face_rel}")
                if _voice_path is None:
                    print(f"  ⚠ missing voice [{_key}]: {_voice_rel}")
                continue
            test_pairs.append((_key, _face_path, _voice_path))

    if skipped_missing:
        print(f"  ⚠ {skipped_missing} row(s) skipped — file(s) not found on disk")
    if not test_pairs:
        raise ValueError(
            f"No valid face/voice pairs found in {eval_ref_csv}\n"
            f"Check EVAL_P56_CSV_ROOT (currently: {eval_csv_root}) and the paths inside the CSV."
        )
    print(f"  Test samples (valid)  : {len(test_pairs)}")
    print()

    face_tf = make_eval_transform()
    max_samples = int(max_audio_sec * TARGET_SR)
    k_faces = int(resolved_cfg.get("k_faces", _get(env, "K_FACES", "4")))

    rows: list[dict[str, object]] = []

    gt_labels: dict[str, int] | None = None
    if eval_labels_csv:
        labels_path = Path(eval_labels_csv)
        labels_path = labels_path if labels_path.is_absolute() else repo_root / labels_path
        if not labels_path.exists():
            raise FileNotFoundError(f"EVAL_P56_LABELS_CSV not found: {labels_path}")
        gt_labels = _load_labels_csv(labels_path)
        print(f"  Loaded GT labels      : {len(gt_labels)} from {labels_path}")

    eval_step = int(ckpt.get("epoch", 0))
    p5_ce_sum = p6_ce_sum = 0.0
    p5_correct = p6_correct = 0
    gt_count = 0

    with torch.no_grad():
        for key, face_path, voice_path in tqdm(test_pairs, desc="Evaluating", unit="sample"):
            face_tensor = load_face_frames(face_path, face_tf, k_faces)
            waveform = load_and_pad_audio(voice_path, max_samples)

            face_batch = face_tensor.unsqueeze(0).to(device)
            wave_batch = waveform.unsqueeze(0).to(device)

            # P5: face + audio (mask false)
            mask_none = torch.zeros(1, dtype=torch.bool, device=device)
            _, logits_p5 = model(face_batch, wave_batch, mask_faces=mask_none)
            probs_p5 = torch.softmax(logits_p5, dim=1)
            pred_p5_idx = int(probs_p5.argmax(dim=1).item())

            # P6: audio only (mask true)
            mask_all = torch.ones(1, dtype=torch.bool, device=device)
            _, logits_p6 = model(face_batch, wave_batch, mask_faces=mask_all)
            probs_p6 = torch.softmax(logits_p6, dim=1)
            pred_p6_idx = int(probs_p6.argmax(dim=1).item())

            rows.append({
                "key": key,
                "p5": idx_to_num[pred_p5_idx],
                "p6": idx_to_num[pred_p6_idx],
            })

            if gt_labels is not None and key in gt_labels:
                gt_num = int(gt_labels[key])
                if gt_num in num_to_idx:
                    gt_idx = num_to_idx[gt_num]
                    gt_tensor = torch.tensor([gt_idx], dtype=torch.long, device=device)
                    p5_ce_sum += F.cross_entropy(logits_p5, gt_tensor, reduction="sum").item()
                    p6_ce_sum += F.cross_entropy(logits_p6, gt_tensor, reduction="sum").item()
                    p5_correct += int(pred_p5_idx == gt_idx)
                    p6_correct += int(pred_p6_idx == gt_idx)
                    gt_count += 1

    # Output path behavior:
    #  - absolute: use as-is
    #  - relative with dirs: resolve from repo root
    #  - filename only: place under OUTPUT_DIR
    csv_cfg_path = Path(eval_csv_name)
    if csv_cfg_path.is_absolute():
        csv_path = csv_cfg_path
    elif csv_cfg_path.parent != Path("."):
        csv_path = repo_root / csv_cfg_path
    else:
        csv_path = output_dir / csv_cfg_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "p5", "p6"])
        writer.writeheader()
        writer.writerows(rows)

    unique_p5 = len({int(r["p5"]) for r in rows})
    unique_p6 = len({int(r["p6"]) for r in rows})

    print("\n" + "═" * 60)
    print("  Evaluation complete")
    print(f"  Samples evaluated    : {len(rows)}")
    print(f"  P5 unique speakers   : {unique_p5}")
    print(f"  P6 unique speakers   : {unique_p6}")
    print(f"  CSV saved -> {csv_path}")

    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(eval_tb_dir))
        try:
            tb_writer.add_scalar("EvalP56/Samples", len(rows), eval_step)
            tb_writer.add_scalar("EvalP56/P5_UniqueSpeakers", unique_p5, eval_step)
            tb_writer.add_scalar("EvalP56/P6_UniqueSpeakers", unique_p6, eval_step)

            if gt_count > 0:
                p5_acc = p5_correct / gt_count * 100.0
                p6_acc = p6_correct / gt_count * 100.0
                p5_loss = p5_ce_sum / gt_count
                p6_loss = p6_ce_sum / gt_count

                tb_writer.add_scalar("EvalP56/P5_Accuracy", p5_acc, eval_step)
                tb_writer.add_scalar("EvalP56/P6_Accuracy", p6_acc, eval_step)
                tb_writer.add_scalar("EvalP56/P5_Loss", p5_loss, eval_step)
                tb_writer.add_scalar("EvalP56/P6_Loss", p6_loss, eval_step)

                print(f"  Labeled samples      : {gt_count}")
                print(f"  P5 acc / loss        : {p5_acc:.2f}% / {p5_loss:.4f}")
                print(f"  P6 acc / loss        : {p6_acc:.2f}% / {p6_loss:.4f}")
            else:
                print("  Labeled metrics      : skipped (no labels / no matches)")

            print(f"  TensorBoard saved -> {eval_tb_dir}")
        finally:
            tb_writer.flush()
            tb_writer.close()
    else:
        print("  TensorBoard          : not available (pip install tensorboard)")

    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
