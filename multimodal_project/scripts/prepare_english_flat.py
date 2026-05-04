#!/usr/bin/env python3
"""
Prepare an English-only training split from MAV-Celeb.

Output layout
-------------
<output_root>/
    faces/
        <speaker_id>/
            <video_folder>/
                *.jpg
    voices/
        <speaker_id>/
            <video_folder>/
                *.wav

Run
---
    python scripts/prepare_english_flat.py
    python scripts/prepare_english_flat.py --source-root Dataset/mavceleb_v1_train \
                                            --output-root Data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an English-only dataset split.")
    parser.add_argument(
        "--source-root",
        default="Dataset/mavceleb_v1_train",
        help="Path to the original MAV-Celeb v1 train split.",
    )
    parser.add_argument(
        "--output-root",
        default="Data",
        help="Root directory for the prepared split.",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copy2(src, dst)


def process_modality(
    src_modality: Path,
    dst_modality: Path,
) -> tuple[int, int]:
    """Copy English-only files for one modality (faces or voices).

    Returns (num_speakers_processed, num_files_copied).
    """
    speakers = 0
    files = 0

    for speaker_dir in sorted(p for p in src_modality.iterdir() if p.is_dir()):
        speaker_id = speaker_dir.name
        eng_dir = speaker_dir / "English"
        if not eng_dir.is_dir():
            continue

        speakers += 1

        for video_dir in sorted(p for p in eng_dir.iterdir() if p.is_dir()):
            dst_video_dir = dst_modality / speaker_id / video_dir.name
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            for media_file in sorted(video_dir.iterdir()):
                if not media_file.is_file():
                    continue
                link_or_copy(media_file, dst_video_dir / media_file.name)
                files += 1

    return speakers, files


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    for modality in ("faces", "voices"):
        src = source_root / modality
        dst = output_root / modality

        if not src.is_dir():
            raise FileNotFoundError(f"Source directory not found: {src}")

        print(f"Copying {modality} …")
        speakers, files = process_modality(src, dst)
        print(f"  {speakers} speakers, {files} files → {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
