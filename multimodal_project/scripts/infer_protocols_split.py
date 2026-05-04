"""
infer_protocols_split.py — Evaluate P3/P4/P5/P6 from split CSV manifests.

This script is designed for locally created CSV manifests such as:
  splits/test_split.csv

It computes protocol accuracies against ground truth speaker labels and reports
challenge-style overall score:

  Score = (Acc(P3) + Acc(P4) + Acc(P5) + Acc(P6)) / 4

By default:
- P3/P4 are evaluated on --test-split-same
- P5/P6 are evaluated on --test-split-cross
- By default, cross split is required for challenge-correct P5/P6 evaluation

CSV requirements (header columns):
  speaker_id,image_path,voice_path
Optional columns used for nicer key output:
  video_id,key

Usage:
  python scripts/infer_protocols_split.py
    python scripts/infer_protocols_split.py \
        --test-split-same splits/test_split_english.csv \
        --test-split-cross splits/test_split_urdu.csv
"""

from __future__ import annotations

import argparse
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

# Allow running from repo root: python scripts/infer_protocols_split.py
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


def _get_bool(env: dict[str, str], key: str, default: bool = False) -> bool:
    v = _get(env, key, "true" if default else "false").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    env = _load_env()
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Infer P3/P4/P5/P6 from split CSVs")
    parser.add_argument(
        "--checkpoint",
        default=_get(
            env,
            "INFER_CHECKPOINT",
            _get(env, "EVAL_ALL_CHECKPOINT", _get(env, "EVAL_CHECKPOINT", "checkpoints/best.pt")),
        ),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--data-root",
        default=_get(env, "INFER_DATA_ROOT", _get(env, "DATA_ROOT", "Data")),
        help="Data root used to resolve relative image/voice paths",
    )
    parser.add_argument(
        "--test-split-same",
        default=_get(env, "INFER_TEST_SPLIT_SAME", "splits/test_split_english.csv"),
        help="CSV used for P3/P4 (same-language protocols)",
    )
    parser.add_argument(
        "--test-split-cross",
        default=_get(env, "INFER_TEST_SPLIT_CROSS", ""),
        help="CSV used for P5/P6 (cross-language protocols)",
    )
    parser.add_argument(
        "--same-language",
        default=_get(env, "INFER_SAME_LANGUAGE", "English"),
        help="Expected language for P3/P4 split rows",
    )
    parser.add_argument(
        "--cross-language",
        default=_get(env, "INFER_CROSS_LANGUAGE", "Urdu"),
        help="Expected language for P5/P6 split rows",
    )
    parser.add_argument(
        "--allow-same-for-cross",
        action="store_true",
        default=_get_bool(env, "INFER_ALLOW_SAME_FOR_CROSS", False),
        help="Allow reusing --test-split-same for P5/P6 when --test-split-cross is missing (debug only)",
    )
    parser.add_argument(
        "--output-csv",
        default=_get(env, "INFER_OUTPUT_CSV", "checkpoints/inference_p3_p4_p5_p6.csv"),
        help="Where per-sample predictions are written",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=_get(env, "INFER_TENSORBOARD_DIR", "runs/infer_protocols_split"),
        help="Directory for TensorBoard inference metrics",
    )
    parser.add_argument(
        "--device",
        default=_get(env, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="cuda or cpu",
    )
    parser.add_argument(
        "--max-audio-sec",
        type=float,
        default=float(_get(env, "MAX_AUDIO_SEC", "6.0")),
        help="Audio crop/pad duration in seconds",
    )
    parser.add_argument(
        "--k-faces",
        type=int,
        default=int(_get(env, "K_FACES", "4")),
        help="Number of tiled frames for face branch",
    )

    args = parser.parse_args()

    def _abs(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else repo_root / p

    args.checkpoint = _abs(args.checkpoint)
    args.data_root = _abs(args.data_root)
    args.test_split_same = _abs(args.test_split_same)
    args.test_split_cross = _abs(args.test_split_cross) if args.test_split_cross else None
    args.output_csv = _abs(args.output_csv)
    args.tensorboard_dir = _abs(args.tensorboard_dir)
    return args


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
    waveform = waveform.mean(0)
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


def _speaker_to_num(speaker_id: str) -> int:
    # id0042 -> 42
    return int(speaker_id.strip().lstrip("id").lstrip("0") or "0")


def _resolve_data_path(data_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else data_root / p


def _infer_lang_from_voice_path(voice_path: str) -> str:
    lower = voice_path.lower()
    if "/english/" in lower:
        return "english"
    if "/urdu/" in lower:
        return "urdu"
    return ""


def load_split_rows(
    split_csv: Path,
    data_root: Path,
    expected_language: str | None = None,
) -> list[dict[str, str | Path | int]]:
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    rows: list[dict[str, str | Path | int]] = []
    expected_norm = (expected_language or "").strip().lower()
    skipped_wrong_lang = 0
    with open(split_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"speaker_id", "image_path", "voice_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns {sorted(missing)}: {split_csv}")

        for i, row in enumerate(reader):
            speaker_id = (row.get("speaker_id") or "").strip()
            image_rel = (row.get("image_path") or "").strip()
            voice_rel = (row.get("voice_path") or "").strip()
            video_id = (row.get("video_id") or "").strip()
            lang_val = (row.get("language") or "").strip().lower()
            if not speaker_id or not image_rel or not voice_rel:
                continue

            if expected_norm:
                resolved_lang = lang_val or _infer_lang_from_voice_path(voice_rel)
                if resolved_lang and resolved_lang != expected_norm:
                    skipped_wrong_lang += 1
                    continue

            image_path = _resolve_data_path(data_root, image_rel)
            voice_path = _resolve_data_path(data_root, voice_rel)
            if not image_path.exists() or not voice_path.exists():
                continue

            key = (row.get("key") or "").strip()
            if not key:
                stem = voice_path.stem
                key = f"{speaker_id}_{video_id}_{stem}" if video_id else f"{speaker_id}_{stem}_{i}"

            rows.append(
                {
                    "key": key,
                    "speaker_id": speaker_id,
                    "speaker_num": _speaker_to_num(speaker_id),
                    "image_path": image_path,
                    "voice_path": voice_path,
                }
            )

    if not rows:
        raise ValueError(f"No valid rows found in split CSV: {split_csv}")
    if skipped_wrong_lang > 0:
        print(
            f"  Note: skipped {skipped_wrong_lang} row(s) not matching expected language "
            f"'{expected_language}' in {split_csv.name}"
        )
    return rows


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


def _run_protocols(
    *,
    model: FusionModel,
    device: torch.device,
    rows: list[dict[str, str | Path | int]],
    face_tf: transforms.Compose,
    k_faces: int,
    max_samples: int,
    idx_to_num: dict[int, int],
    num_to_idx: dict[int, int],
    proto_full: str,
    proto_audio: str,
) -> tuple[list[dict[str, int | str]], dict[str, float]]:
    out_rows: list[dict[str, int | str]] = []
    n = len(rows)
    correct_full = 0
    correct_audio = 0
    ce_full_sum = 0.0
    ce_audio_sum = 0.0

    with torch.no_grad():
        for row in tqdm(rows, desc=f"Running {proto_full}/{proto_audio}", unit="sample"):
            image_path = row["image_path"]
            voice_path = row["voice_path"]
            gt_num = int(row["speaker_num"])
            gt_idx = num_to_idx.get(gt_num, 0)
            gt_tensor = torch.tensor([gt_idx], dtype=torch.long, device=device)

            face_tensor = load_face_frames(image_path, face_tf, k_faces)
            waveform = load_and_pad_audio(voice_path, max_samples)

            face_batch = face_tensor.unsqueeze(0).to(device)
            wave_batch = waveform.unsqueeze(0).to(device)

            # Full-modality prediction (P3 or P5)
            mask_none = torch.zeros(1, dtype=torch.bool, device=device)
            _, logits_full = model(face_batch, wave_batch, mask_faces=mask_none)
            pred_full_idx = int(torch.softmax(logits_full, dim=1).argmax(dim=1).item())
            pred_full_num = idx_to_num[pred_full_idx]
            ce_full_sum += F.cross_entropy(logits_full, gt_tensor).item()

            # Audio-only prediction (P4 or P6)
            mask_all = torch.ones(1, dtype=torch.bool, device=device)
            _, logits_audio = model(face_batch, wave_batch, mask_faces=mask_all)
            pred_audio_idx = int(torch.softmax(logits_audio, dim=1).argmax(dim=1).item())
            pred_audio_num = idx_to_num[pred_audio_idx]
            ce_audio_sum += F.cross_entropy(logits_audio, gt_tensor).item()

            correct_full += int(pred_full_num == gt_num)
            correct_audio += int(pred_audio_num == gt_num)

            out_rows.append(
                {
                    "key": str(row["key"]),
                    "speaker_id": str(row["speaker_id"]),
                    "gt_num": gt_num,
                    proto_full: pred_full_num,
                    proto_audio: pred_audio_num,
                }
            )

    metrics = {
        f"{proto_full}_acc": (correct_full / n) * 100.0,
        f"{proto_audio}_acc": (correct_audio / n) * 100.0,
        f"{proto_full}_ce": ce_full_sum / n,
        f"{proto_audio}_ce": ce_audio_sum / n,
        f"{proto_full}_count": float(n),
        f"{proto_audio}_count": float(n),
    }
    return out_rows, metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not (args.data_root / "faces").is_dir():
        raise FileNotFoundError(f"Training faces directory not found: {args.data_root / 'faces'}")

    faces_train = args.data_root / "faces"
    speakers: list[str] = sorted(p.name for p in faces_train.iterdir() if p.is_dir())
    num_speakers = len(speakers)
    idx_to_num: dict[int, int] = {
        i: int(s.lstrip("id").lstrip("0") or "0") for i, s in enumerate(speakers)
    }
    num_to_idx: dict[int, int] = {v: k for k, v in idx_to_num.items()}

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg: dict = ckpt.get("cfg", {})

    model = build_model(ckpt_cfg, num_speakers)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    same_rows = load_split_rows(
        args.test_split_same,
        args.data_root,
        expected_language=args.same_language,
    )

    if args.test_split_cross is None:
        if not args.allow_same_for_cross:
            raise ValueError(
                "--test-split-cross is required for challenge-correct P5/P6 evaluation. "
                "Pass --allow-same-for-cross only for debug runs."
            )
        cross_csv = args.test_split_same
    else:
        cross_csv = args.test_split_cross

    cross_rows = load_split_rows(
        cross_csv,
        args.data_root,
        expected_language=args.cross_language,
    )

    face_tf = make_eval_transform()
    max_samples = int(args.max_audio_sec * TARGET_SR)

    rows_34, metrics_34 = _run_protocols(
        model=model,
        device=device,
        rows=same_rows,
        face_tf=face_tf,
        k_faces=args.k_faces,
        max_samples=max_samples,
        idx_to_num=idx_to_num,
        num_to_idx=num_to_idx,
        proto_full="p3",
        proto_audio="p4",
    )

    rows_56, metrics_56 = _run_protocols(
        model=model,
        device=device,
        rows=cross_rows,
        face_tf=face_tf,
        k_faces=args.k_faces,
        max_samples=max_samples,
        idx_to_num=idx_to_num,
        num_to_idx=num_to_idx,
        proto_full="p5",
        proto_audio="p6",
    )

    by_key: dict[str, dict[str, int | str]] = {}
    for row in rows_34:
        by_key[str(row["key"])] = dict(row)
    for row in rows_56:
        key = str(row["key"])
        if key not in by_key:
            by_key[key] = {
                "key": row["key"],
                "speaker_id": row["speaker_id"],
                "gt_num": row["gt_num"],
            }
        by_key[key]["p5"] = row["p5"]
        by_key[key]["p6"] = row["p6"]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "speaker_id", "gt_num", "p3", "p4", "p5", "p6"])
        writer.writeheader()
        writer.writerows(sorted(by_key.values(), key=lambda x: str(x["key"])))

    p3 = metrics_34["p3_acc"]
    p4 = metrics_34["p4_acc"]
    p5 = metrics_56["p5_acc"]
    p6 = metrics_56["p6_acc"]
    score = (p3 + p4 + p5 + p6) / 4.0

    if SummaryWriter is not None:
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
        try:
            tb_writer.add_scalar("EvalSplit/P3_Accuracy", p3, 1)
            tb_writer.add_scalar("EvalSplit/P4_Accuracy", p4, 1)
            tb_writer.add_scalar("EvalSplit/P5_Accuracy", p5, 1)
            tb_writer.add_scalar("EvalSplit/P6_Accuracy", p6, 1)
            tb_writer.add_scalar("EvalSplit/ChallengeScore", score, 1)
            tb_writer.add_scalar("EvalSplit/P3_Samples", metrics_34["p3_count"], 1)
            tb_writer.add_scalar("EvalSplit/P4_Samples", metrics_34["p4_count"], 1)
            tb_writer.add_scalar("EvalSplit/P5_Samples", metrics_56["p5_count"], 1)
            tb_writer.add_scalar("EvalSplit/P6_Samples", metrics_56["p6_count"], 1)
            tb_writer.add_scalar("InferCE/P3_CE", metrics_34["p3_ce"], 1)
            tb_writer.add_scalar("InferCE/P4_CE", metrics_34["p4_ce"], 1)
            tb_writer.add_scalar("InferCE/P5_CE", metrics_56["p5_ce"], 1)
            tb_writer.add_scalar("InferCE/P6_CE", metrics_56["p6_ce"], 1)
            tb_writer.flush()
        finally:
            tb_writer.close()

    print("\n" + "=" * 72)
    print(" POLY-SIM Split Inference Complete")
    print("=" * 72)
    print(f"  Checkpoint           : {args.checkpoint}")
    print(f"  Same split CSV       : {args.test_split_same}")
    print(f"  Cross split CSV      : {cross_csv}")
    print(f"  P3/P4 language       : {args.same_language}")
    print(f"  P5/P6 language       : {args.cross_language}")
    if args.test_split_cross is None:
        print("  Note                 : Reused same split for P5/P6 (debug mode)")
    print(f"  P3 accuracy          : {p3:.2f}%   CE: {metrics_34['p3_ce']:.4f}")
    print(f"  P4 accuracy          : {p4:.2f}%   CE: {metrics_34['p4_ce']:.4f}")
    print(f"  P5 accuracy          : {p5:.2f}%   CE: {metrics_56['p5_ce']:.4f}")
    print(f"  P6 accuracy          : {p6:.2f}%   CE: {metrics_56['p6_ce']:.4f}")
    print(f"  Challenge score      : {score:.2f}%")
    print(f"  Predictions CSV      : {args.output_csv}")
    if SummaryWriter is not None:
        print(f"  TensorBoard dir      : {args.tensorboard_dir}")
    else:
        print("  TensorBoard dir      : unavailable (pip install tensorboard)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
