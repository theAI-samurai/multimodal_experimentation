# POLY-SIM Project Report

## 1. Objective

This project targets the **POLY-SIM 2026** challenge: multimodal speaker identification under two difficult real-world conditions:

1. **Missing modality**: face information may be unavailable at inference.
2. **Cross-lingual shift**: model is trained on English and evaluated on Urdu.

The goal is to maintain strong speaker identification performance across four protocols:

- **P3**: Face + Audio, English -> English
- **P4**: Audio only, English -> English
- **P5**: Face + Audio, English -> Urdu
- **P6**: Audio only, English -> Urdu

Challenge score is computed as:

\[
\text{Score} = \frac{\text{Acc}(P3) + \text{Acc}(P4) + \text{Acc}(P5) + \text{Acc}(P6)}{4}
\]

## 2. Model Architecture

The system is a two-branch multimodal model with a fusion head:

### 2.1 Face branch

- Backbone: **ViT-Large** (`google/vit-large-patch16-224`)
- Output feature: [CLS] token embedding
- Projection: Linear + LayerNorm -> shared embedding space

### 2.2 Audio branch

- Backbone: **WavLM-Large** (`microsoft/wavlm-large`)
- Temporal aggregation: mean pooling over time
- Projection: Linear + LayerNorm -> shared embedding space

### 2.3 Fusion and classification

- Face and audio embeddings are concatenated
- Fusion MLP: `Linear(2D -> D) + GELU + LayerNorm`
- Classifier: `Linear(D -> num_speakers)`

### 2.4 Missing-modality robustness

A learnable **face mask token** is used with modality dropout (`MASK_PROB=0.3`) during training.
This enables robust inference for audio-only protocols (P4, P6) by replacing face embeddings when needed.

## 3. Experimental Setup (Current Run)

Configuration (from `.env`):

- `FACE_ENCODER=vit_large`
- `AUDIO_ENCODER=wavlm_large`
- `EMBED_DIM=384`
- `MASK_PROB=0.3`
- `INFER_CHECKPOINT=checkpoints/exp1/best.pt`
- `INFER_TEST_SPLIT_SAME=splits/test_split_english.csv`
- `INFER_TEST_SPLIT_CROSS=splits/test_split_urdu.csv`
- `INFER_OUTPUT_CSV=checkpoints/inference_p3_p4_p5_p6.csv`
- `INFER_TENSORBOARD_DIR=runs/infer_protocols_split`

Split sizes:

- Same-language split (English): **1159** samples
- Cross-language split (Urdu): **101** samples

TensorBoard logs used in this report:

- Training logs: `runs/exp1`
- Split inference logs: `runs/infer_protocols_split`

## 4. Training Metrics from TensorBoard

Training metrics were extracted from the TensorBoard event files in `runs/exp1`.

| Metric | First | Last | Best |
|---|---:|---:|---:|
| Train accuracy | 41.99% | 99.39% | **99.65%** |
| Validation accuracy | 81.80% | 100.00% | **100.00%** |
| Train loss | 4.0741 | 0.0926 | **0.0784** |
| Validation loss | 2.0790 | 0.3148 | **0.2042** |

## 5. Inference Results

Metrics were computed from:

- `checkpoints/inference_p3_p4_p5_p6.csv`
- TensorBoard evaluation scalars in `runs/infer_protocols_split`

Per-protocol accuracies:

| Protocol | Setting | Samples | Accuracy |
|---|---|---:|---:|
| P3 | Face + Audio, English -> English | 1159 | **99.74%** |
| P4 | Audio only, English -> English | 1159 | **96.29%** |
| P5 | Face + Audio, English -> Urdu | 101 | **100.00%** |
| P6 | Audio only, English -> Urdu | 101 | **98.02%** |

Overall challenge score:

\[
\frac{99.7412 + 96.2899 + 100.0000 + 98.0198}{4} = 98.51\%
\]

## 6. Observations

1. The model achieves very high full-modality performance (P3/P5), showing strong speaker separability.
2. Audio-only performance (P4/P6) remains high, indicating effective missing-face robustness.
3. Cross-lingual performance is strong in this evaluation setup, suggesting good transfer from English-trained representations.
4. TensorBoard records now capture both training progression and final split-inference evaluation metrics, making the experiment easier to audit and present.

## 7. Conclusion

The current ViT-Large + WavLM-Large fusion model with modality dropout is highly effective for all four POLY-SIM protocols in this experiment, achieving a strong overall score of **98.51%**.

Future work can focus on:

- more rigorous held-out splits for robustness checks,
- calibration and confidence analysis,
- protocol-specific ablations (mask probability, embedding size, and backbone variants).
