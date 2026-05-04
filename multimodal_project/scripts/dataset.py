"""
dataset.py — MAV-Celeb English dataset for multimodal speaker identification.

Data layout expected:
    <data_root>/
        faces/
            <speaker_id>/
                <video_folder>/
                    *.jpg          (frames already at 224×224)
        voices/
            <speaker_id>/
                <video_folder>/
                    *.wav          (mono, 16 kHz)

Each __getitem__ returns:
    face_frames  : Tensor [K, 3, H, W]   — K frames sampled from the same video
    waveform     : Tensor [T]            — cropped/padded mono waveform at 16 kHz
    label        : int                   — speaker index (0 … num_speakers-1)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from torch.utils.data import Dataset


class MAVCelebDataset(Dataset):
    """English-only MAV-Celeb dataset.

    Args:
        data_root:       Path to the prepared Data/ directory.
        k_faces:         Number of face frames to sample per audio clip.
        max_audio_sec:   Maximum audio duration in seconds (clips are cropped/padded).
        sample_rate:     Target sample rate. Source files are already 16 kHz.
        face_transform:  torchvision transform applied to each PIL face image.
    """

    TARGET_SR: int = 16_000

    def __init__(
        self,
        data_root: str | Path,
        k_faces: int = 4,
        max_audio_sec: float = 6.0,
        sample_rate: int = TARGET_SR,
        face_transform=None,
        manifest_csv: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.k_faces = k_faces
        self.max_audio_samples = int(max_audio_sec * sample_rate)
        self.sample_rate = sample_rate
        self.face_transform = face_transform
        self.manifest_csv = Path(manifest_csv) if manifest_csv else None

        faces_root = self.data_root / "faces"
        voices_root = self.data_root / "voices"

        if not faces_root.is_dir():
            raise FileNotFoundError(f"Faces directory not found: {faces_root}")
        if not voices_root.is_dir():
            raise FileNotFoundError(f"Voices directory not found: {voices_root}")

        # Sorted speaker list — deterministic label assignment
        self.speakers: list[str] = sorted(
            p.name for p in faces_root.iterdir() if p.is_dir()
        )
        self.speaker_to_idx: dict[str, int] = {
            s: i for i, s in enumerate(self.speakers)
        }

        # samples[i] = (speaker_id, wav_path, face_pool)
        # face_pool is a list of jpg Paths for the same video
        self.samples: list[tuple[str, Path, list[Path]]] = []

        # per-speaker sample counts (for WeightedRandomSampler)
        self._speaker_counts: list[int] = []

        if self.manifest_csv is not None:
            self._load_from_manifest(self.manifest_csv)
        else:
            for speaker in self.speakers:
                count_before = len(self.samples)
                voice_spk = voices_root / speaker
                face_spk = faces_root / speaker

                if not voice_spk.is_dir():
                    self._speaker_counts.append(0)
                    continue

                for video_dir in sorted(v for v in voice_spk.iterdir() if v.is_dir()):
                    face_video_dir = face_spk / video_dir.name
                    if not face_video_dir.is_dir():
                        continue

                    face_pool = sorted(face_video_dir.glob("*.jpg"))
                    if not face_pool:
                        continue

                    for wav in sorted(video_dir.glob("*.wav")):
                        self.samples.append((speaker, wav, face_pool))

                self._speaker_counts.append(len(self.samples) - count_before)

    def _load_from_manifest(self, manifest_csv: Path) -> None:
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

        speaker_counts_by_name: dict[str, int] = {s: 0 for s in self.speakers}
        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            required = {"speaker_id", "voice_path", "face_dir"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manifest CSV missing required columns {sorted(missing)}: {manifest_csv}"
                )

            for row in reader:
                speaker = (row.get("speaker_id") or "").strip()
                voice_rel = (row.get("voice_path") or "").strip()
                face_dir_rel = (row.get("face_dir") or "").strip()
                if not speaker or not voice_rel or not face_dir_rel:
                    continue
                if speaker not in self.speaker_to_idx:
                    continue

                wav_path = self.data_root / voice_rel
                face_dir = self.data_root / face_dir_rel
                if not wav_path.exists() or not face_dir.is_dir():
                    continue

                face_pool = sorted(face_dir.glob("*.jpg"))
                if not face_pool:
                    continue

                self.samples.append((speaker, wav_path, face_pool))
                speaker_counts_by_name[speaker] += 1

        self._speaker_counts = [speaker_counts_by_name[s] for s in self.speakers]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def get_sample_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler (equalises speakers)."""
        weights: list[float] = []
        for speaker, _, _ in self.samples:
            idx = self.speaker_to_idx[speaker]
            n = self._speaker_counts[idx]
            weights.append(1.0 / max(n, 1))
        return weights

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        speaker, wav_path, face_pool = self.samples[idx]
        label = self.speaker_to_idx[speaker]

        # ---- Audio -------------------------------------------------------
        waveform, sr = torchaudio.load(wav_path)           # [C, T]
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(0)                        # mono [T]

        T = waveform.shape[0]
        if T < self.max_audio_samples:
            # Zero-pad short clips
            waveform = F.pad(waveform, (0, self.max_audio_samples - T))
        else:
            # Random crop
            start = random.randint(0, T - self.max_audio_samples)
            waveform = waveform[start : start + self.max_audio_samples]

        # ---- Faces -------------------------------------------------------
        chosen_paths = random.choices(face_pool, k=self.k_faces)
        frames: list[torch.Tensor] = []
        for fp in chosen_paths:
            img = Image.open(fp).convert("RGB")
            if self.face_transform is not None:
                img = self.face_transform(img)
            else:
                img = torch.tensor(
                    list(img.getdata()),
                    dtype=torch.float32,
                ).reshape(224, 224, 3).permute(2, 0, 1) / 255.0
            frames.append(img)

        face_tensor = torch.stack(frames)  # [K, 3, H, W]

        return face_tensor, waveform, label
