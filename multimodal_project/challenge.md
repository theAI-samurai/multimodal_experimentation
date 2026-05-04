# POLY-SIM: Polyglot Speaker Identification with Missing Modality

> **Grand Challenge 2026** — Multimodal speaker identification under missing-modality and cross-lingual conditions.

---

## Overview

POLY-SIM 2026 is a grand challenge that advances research in multimodal speaker identification under two simultaneous real-world difficulties:

1. **Missing modality** — the visual (face) input is absent at inference time due to occlusions, camera failures, or privacy constraints.
2. **Cross-lingual shift** — the model is trained on English audio but evaluated on Urdu audio, introducing acoustic and phonetic variability.

The challenge is hosted on [CodaBench](https://www.codabench.org/competitions/11283) and is co-located with **Interspeech 2026**. It builds on the FAME 2024 and FAME 2026 grand challenges focused on face–voice association across languages.

---

## The Problem

A multimodal model is trained on **paired face images and English audio** from bilingual celebrity speakers (MAV-Celeb dataset). At test time:

- The **face modality is missing** — only audio is available.
- The **audio is in Urdu** — a different language from training.

```
Train:   Face (image) + Audio (English)  →  Speaker ID
Test:    Audio (Urdu) only               →  Speaker ID
```

The baseline FOP model drops from ~98% accuracy (full multimodal, same language) to ~32–44% in the hardest setting — demonstrating the severity of the combined challenge.

---

## Dataset: MAV-Celeb

The dataset consists of audio-visual samples from YouTube interviews, talk shows, and television debates. Each speaker is bilingual (English and Urdu).

| Language Pair | Lang  | Total Videos (Tr/Val/Test) | Samples (Tr/Val/Test)     |
|---------------|-------|----------------------------|---------------------------|
| English–Urdu  | Eng   | 262 / 70 / 70              | 4039 / 1290 / 1521        |
| English–Urdu  | Urdu  | 415 / 70 / 70              | 9304 / 1779 / 1623        |

**Train split structure:**
```
train/
├── faces/
│   └── id0001/  (multiple .jpg per video)
└── voices/
    ├── english/
    │   └── id0001/  (.wav files)
    └── urdu/
        └── id0001/  (.wav files)
```

Pre-extracted features are released alongside raw data, encoded with state-of-the-art pretrained architectures (FaceNet for faces, ECAPA-TDNN for audio).

---

## Evaluation Protocol

Four configurations are evaluated:

| Protocol | Training | Test modality | Language     |
|----------|----------|---------------|--------------|
| **P3**   | Face + Audio | Face + Audio | Same (English) |
| **P4**   | Face + Audio | Audio only   | Same (English) |
| **P5**   | Face + Audio | Face + Audio | Cross (→ Urdu) |
| **P6**   | Face + Audio | Audio only   | Cross (→ Urdu) |

**Overall Score:**
```
Score = (Acc(P3) + Acc(P4) + Acc(P5) + Acc(P6)) / 4
```

**Metric:** P-accuracy — proportion of test pairs where the correct speaker is identified among P candidates.

### Baseline Results

| Phase    | P3    | P4    | P5    | P6    | Avg   |
|----------|-------|-------|-------|-------|-------|
| Progress | 97.44 | 37.75 | 98.48 | 31.70 | 66.34 |
| Eval     | 98.82 | 52.53 | 98.27 | 43.87 | 73.37 |

---

## Baseline Model

The released baseline is **FOP (Fusion and Orthogonal Projection)** — a two-branch network combining face and audio embeddings, optimized with an orthogonality constraint on embeddings of different speakers.

- **Face encoder:** FaceNet (pretrained on large-scale facial recognition data)
- **Audio encoder:** ECAPA-TDNN (trained on English speech for speaker recognition)
- **Fusion:** Joint projection with orthogonality loss

Starter kit and pretrained weights: [github.com/msaadsaeed/polysim](https://github.com/msaadsaeed/polysim)

---

## Suggested Architectures

### 1. Multilingual Audio Encoder + Alignment

Replace ECAPA-TDNN with a multilingual speech model (e.g., WavLM-Large or MMS) and align audio embeddings to the face embedding space using triplet or orthogonality loss.

**Strengths:** Cross-lingual robustness via multilingual pre-training; drop-in replacement for the audio branch.

---

### 2. Cross-Modal Knowledge Distillation

Train a teacher (face+audio, English) and distill into a student (audio-only) that matches the teacher's fused embeddings via KL-divergence or cosine distillation. Augment student training with Urdu audio.

**Strengths:** Student learns to compensate for the missing face modality at inference; strong missing-modality robustness.

---

### 3. Single-Branch Modality-Agnostic Network (SBAN)

A single transformer-based encoder accepting any combination of modalities. Missing modalities are replaced by a learnable `[MASK]` token. Trained with random modality dropout.

**Strengths:** Architecturally unified; never breaks on missing input — the missing-modality case is part of the training distribution.

---

### 4. Foundation Model + Prompt Tuning

Freeze a large pre-trained audio-visual backbone (e.g., AV-HuBERT, ImageBind). Add lightweight language-conditional prompt adapters to shift audio embeddings toward a language-neutral space. Use speaker prototypes for retrieval-augmented inference.

**Strengths:** Leverages massive pre-training; minimal fine-tuning required.

---

### 5. Adversarial Language-Invariant Representations

Add a gradient-reversal language discriminator to the audio encoder. The adversarial signal forces the encoder to remove language-specific features while retaining speaker identity.

**Strengths:** Directly targets cross-lingual degradation at the representation level.

---

### Architecture Comparison

| Architecture                  | Missing modality | Cross-lingual | Complexity |
|-------------------------------|:---:|:---:|:---:|
| Multilingual encoder + FOP    | ★★★ | ★★★★ | Low    |
| Cross-modal distillation      | ★★★★★ | ★★★ | Medium |
| Single-branch (SBAN)          | ★★★★★ | ★★★ | Medium |
| Foundation model + prompts    | ★★★ | ★★★★ | Low    |
| Adversarial lang-invariant    | ★★★ | ★★★★ | Medium |

> **Recommended combination:** SBAN-style architecture using WavLM-Large with an adversarial language-stripping head and knowledge distillation from a full multimodal teacher.

---

## Submission Format

Submit a ZIP archive of CSV files, one per language pair:

```
submission.zip
├── submission_v1_val_English_English.csv
├── submission_v1_val_English_Urdu.csv
├── submission_v1_test_English_English.csv
└── submission_v1_test_English_Urdu.csv
```

**Monolingual CSV** (`lang1 == lang2`): columns `key, p3, p4`  
**Cross-lingual CSV** (`lang1 != lang2`): columns `key, p5, p6`

Example:
```csv
key,p3,p4
t5M7dziYVY,1,0
RmUYdg2luC,50,0
BvKCMACzXt,20,0
```

**Submission limits:**
- Progress phase: max 150 total, 15 per day
- Evaluation phase: max 15 total

---

## Timeline

| Milestone              | Date              |
|------------------------|-------------------|
| Registration opens     | 27 March 2026     |
| Progress phase opens   | 27 March 2026     |
| Progress phase closes  | 15 May 2026       |
| Evaluation phase       | 16–23 May 2026    |
| Challenge results      | 25 May 2026       |
| Final paper deadline   | 8 June 2026       |

---

## Registration

Register your team via the [Google Form](https://forms.gle/EwmVBiph2QsZ2QRB9).

---

## Rules

- All participants must submit a **system description paper**. Teams without one will be disqualified.
- A **working code repository** (e.g., GitHub) must be submitted alongside results.
- Systems violating challenge rules will be disqualified.

---

## Organizers

Marta Moscati, Muhammad Saad Saeed, Marina Zanoni, Mubashir Noman, Rohan Kumar Das, Monorama Swain, Yufang Hou, Elisabeth André, Khalid Mahmood Malik, Markus Schedl, Shah Nawaz

*Johannes Kepler University Linz · University of Michigan-Flint · Sapienza University of Rome · MBZUAI · Fortemedia Singapore · IT:U Austria · University of Augsburg*

Contact: mavceleb@gmail.com

---

## Citation

```bibtex
@article{moscati2026polysim,
  title   = {POLY-SIM: Polyglot Speaker Identification with Missing Modality Grand Challenge 2026},
  author  = {Moscati, Marta and Saeed, Muhammad Saad and Zanoni, Marina and Noman, Mubashir
             and Das, Rohan Kumar and Swain, Monorama and Hou, Yufang and André, Elisabeth
             and Malik, Khalid Mahmood and Schedl, Markus and Nawaz, Shah},
  journal = {arXiv preprint arXiv:2603.24569},
  year    = {2026}
}
```

---

## Acknowledgements

This research was funded by the Austrian Science Fund (FWF): Cluster of Excellence Bilateral Artificial Intelligence (COE12), the doc.funds.connect project Human-Centered AI (DFH23), and the PI project Intent-aware Music Recommender Systems (P36413).
