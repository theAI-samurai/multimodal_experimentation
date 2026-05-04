"""
Create 70/20/10 train/test/val CSV manifests from Data/ for POLY-SIM.

Each manifest row corresponds to one audio clip paired with its face video folder.
The training dataset samples K frames from face_dir at runtime, while image_path
stores a representative frame for traceability.

Expected updated layout:
        Data/
            faces/<speaker_id>/<video_id>/*.jpg
            voices/<speaker_id>/<language>/<video_id>/*.wav

Where <language> is typically English or Urdu.

Usage:
    python scripts/create_data_splits.py

Optional:
    python scripts/create_data_splits.py \
      --data-root Data \
      --out-dir splits \
      --train-ratio 0.7 \
      --test-ratio 0.2 \
      --val-ratio 0.1 \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/test/val split CSVs")
    parser.add_argument("--data-root", default="Data", help="Root containing faces/ and voices/")
    parser.add_argument("--out-dir", default="splits", help="Where CSVs are written")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ratio_counts(n: int, train_ratio: float, test_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_n = int(round(n * train_ratio))
    test_n = int(round(n * test_ratio))
    if train_n + test_n > n:
        test_n = max(0, n - train_n)
    val_n = n - train_n - test_n
    if val_n < 0:
        val_n = 0
        test_n = max(0, n - train_n)
    return train_n, test_n, val_n


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def discover_records(data_root: Path) -> list[dict[str, str]]:
    faces_root = data_root / "faces"
    voices_root = data_root / "voices"

    if not faces_root.is_dir() or not voices_root.is_dir():
        raise FileNotFoundError(f"Expected faces/ and voices/ in {data_root}")

    records: list[dict[str, str]] = []
    speakers = sorted(p.name for p in faces_root.iterdir() if p.is_dir())

    for speaker_id in speakers:
        face_spk = faces_root / speaker_id
        voice_spk = voices_root / speaker_id
        if not voice_spk.is_dir():
            continue

        for lang_dir in sorted(v for v in voice_spk.iterdir() if v.is_dir()):
            language = lang_dir.name
            for video_dir in sorted(v for v in lang_dir.iterdir() if v.is_dir()):
                face_video_dir = face_spk / video_dir.name
                if not face_video_dir.is_dir():
                    continue

                face_images = sorted(face_video_dir.glob("*.jpg"))
                if not face_images:
                    continue
                rep_face = face_images[0]

                for wav_path in sorted(video_dir.glob("*.wav")):
                    records.append(
                        {
                            "speaker_id": speaker_id,
                            "language": language,
                            "video_id": video_dir.name,
                            "voice_path": relative_to_root(wav_path, data_root),
                            "image_path": relative_to_root(rep_face, data_root),
                            "face_dir": relative_to_root(face_video_dir, data_root),
                        }
                    )

    return records


def split_records(
    records: list[dict[str, str]], train_ratio: float, test_ratio: float, val_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    by_speaker_lang: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_speaker_lang[(row["speaker_id"], row["language"])].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []

    for (speaker_id, language), rows in sorted(by_speaker_lang.items()):
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        train_n, test_n, _ = ratio_counts(n, train_ratio, test_ratio, val_ratio)

        train_part = rows[:train_n]
        test_part = rows[train_n: train_n + test_n]
        val_part = rows[train_n + test_n:]

        train_rows.extend(train_part)
        test_rows.extend(test_part)
        val_rows.extend(val_part)

        if not train_part:
            print(f"Warning: speaker {speaker_id} ({language}) has 0 train samples")
        if not test_part:
            print(f"Warning: speaker {speaker_id} ({language}) has 0 test samples")
        if not val_part:
            print(f"Warning: speaker {speaker_id} ({language}) has 0 val samples")

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    rng.shuffle(val_rows)
    return train_rows, test_rows, val_rows


def write_csv(path: Path, split_name: str, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "speaker_id", "language", "video_id", "image_path", "voice_path", "face_dir"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["split"] = split_name
            writer.writerow(out)


def write_language_csvs(out_dir: Path, base_name: str, split_name: str, rows: list[dict[str, str]]) -> None:
    by_lang: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_lang[row["language"].lower()].append(row)

    for lang, lang_rows in sorted(by_lang.items()):
        path = out_dir / f"{base_name}_{lang}.csv"
        write_csv(path, split_name, lang_rows)


def main() -> None:
    args = parse_args()

    total_ratio = args.train_ratio + args.test_ratio + args.val_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train/test/val ratios must sum to 1.0")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    print(f"Scanning data root: {data_root}")
    records = discover_records(data_root)
    if not records:
        raise ValueError("No valid face/voice pairs found")

    train_rows, test_rows, val_rows = split_records(
        records,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_csv = out_dir / "train_split.csv"
    test_csv = out_dir / "test_split.csv"
    val_csv = out_dir / "val_split.csv"
    all_csv = out_dir / "all_splits.csv"

    write_csv(train_csv, "train", train_rows)
    write_csv(test_csv, "test", test_rows)
    write_csv(val_csv, "val", val_rows)

    write_language_csvs(out_dir, "train_split", "train", train_rows)
    write_language_csvs(out_dir, "test_split", "test", test_rows)
    write_language_csvs(out_dir, "val_split", "val", val_rows)

    write_csv(all_csv, "train", train_rows)
    with open(all_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "speaker_id", "language", "video_id", "image_path", "voice_path", "face_dir"],
        )
        for row in test_rows:
            out = dict(row)
            out["split"] = "test"
            writer.writerow(out)
        for row in val_rows:
            out = dict(row)
            out["split"] = "val"
            writer.writerow(out)

    total = len(records)
    print("\nSplit complete")
    print(f"  Total samples : {total}")
    print(f"  Train         : {len(train_rows)} ({len(train_rows) / total * 100:.2f}%)")
    print(f"  Test          : {len(test_rows)} ({len(test_rows) / total * 100:.2f}%)")
    print(f"  Validation    : {len(val_rows)} ({len(val_rows) / total * 100:.2f}%)")
    print(f"  Wrote         : {train_csv}")
    print(f"  Wrote         : {test_csv}")
    print(f"  Wrote         : {val_csv}")
    print(f"  Wrote         : {all_csv}")
    print("  Wrote         : per-language CSVs (e.g., test_split_english.csv, test_split_urdu.csv)")


if __name__ == "__main__":
    main()
