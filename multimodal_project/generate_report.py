#!/usr/bin/env python3
"""
generate_report.py — Generates a PDF technical report for the POLY-SIM
Multimodal Speaker Identification project.
"""

from __future__ import annotations

import io
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor

# ─── Colour Palette ────────────────────────────────────────────────────────────
C_NAVY    = HexColor("#1B2A4A")
C_BLUE    = HexColor("#2563EB")
C_LIGHT   = HexColor("#EFF6FF")
C_ACCENT  = HexColor("#F59E0B")
C_GREEN   = HexColor("#10B981")
C_RED     = HexColor("#EF4444")
C_GREY    = HexColor("#6B7280")
C_BGRAY   = HexColor("#F3F4F6")
C_WHITE   = colors.white

W, H = A4
MARGIN = 2 * cm
CONTENT_W = W - 2 * MARGIN


# ─── Style Sheet ───────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "cover_title": ParagraphStyle(
            "cover_title", fontSize=28, leading=34, alignment=TA_CENTER,
            textColor=C_WHITE, fontName="Helvetica-Bold", spaceAfter=6,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontSize=14, leading=18, alignment=TA_CENTER,
            textColor=HexColor("#BFDBFE"), fontName="Helvetica", spaceAfter=4,
        ),
        "cover_tag": ParagraphStyle(
            "cover_tag", fontSize=10, leading=14, alignment=TA_CENTER,
            textColor=HexColor("#93C5FD"), fontName="Helvetica-Oblique",
        ),
        "section": ParagraphStyle(
            "section", fontSize=14, leading=18, textColor=C_NAVY,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
            borderPad=0,
        ),
        "subsection": ParagraphStyle(
            "subsection", fontSize=11, leading=14, textColor=C_BLUE,
            fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontSize=9.5, leading=14, textColor=HexColor("#1F2937"),
            fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontSize=9.5, leading=13, textColor=HexColor("#1F2937"),
            fontName="Helvetica", leftIndent=14, spaceAfter=3,
            bulletIndent=4,
        ),
        "code": ParagraphStyle(
            "code", fontSize=8.5, leading=12, fontName="Courier",
            textColor=HexColor("#1E3A5F"), backColor=HexColor("#F0F4FF"),
            leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4,
            borderColor=HexColor("#BFDBFE"), borderWidth=0.5, borderPad=6,
        ),
        "caption": ParagraphStyle(
            "caption", fontSize=8.5, leading=11, alignment=TA_CENTER,
            textColor=C_GREY, fontName="Helvetica-Oblique", spaceAfter=8,
        ),
        "gap_title": ParagraphStyle(
            "gap_title", fontSize=10, leading=13, textColor=C_RED,
            fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2,
        ),
        "done_title": ParagraphStyle(
            "done_title", fontSize=10, leading=13, textColor=C_GREEN,
            fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2,
        ),
        "table_header": ParagraphStyle(
            "table_header", fontSize=9, leading=12, fontName="Helvetica-Bold",
            textColor=C_WHITE, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", fontSize=8.5, leading=11, fontName="Helvetica",
            textColor=HexColor("#1F2937"), alignment=TA_CENTER,
        ),
    }
    return styles


# ─── Page Templates ────────────────────────────────────────────────────────────

def cover_background(canvas, doc):
    canvas.saveState()
    # Gradient background using layered rectangles
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Decorative arc top-right
    canvas.setFillColor(HexColor("#243458"))
    canvas.circle(W + 60, H + 60, 240, fill=1, stroke=0)
    # Bottom stripe
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, 0, W, 1.2 * cm, fill=1, stroke=0)
    # Accent stripe
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, 1.2 * cm, W, 0.25 * cm, fill=1, stroke=0)
    canvas.restoreState()


def normal_header_footer(canvas, doc):
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(C_NAVY)
    canvas.setLineWidth(0.8)
    canvas.line(MARGIN, H - 1.4 * cm, W - MARGIN, H - 1.4 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(C_GREY)
    canvas.drawString(MARGIN, H - 1.2 * cm, "POLY-SIM 2026 — Technical Report")
    canvas.drawRightString(W - MARGIN, H - 1.2 * cm, "Multimodal Speaker Identification")
    # Footer
    canvas.setStrokeColor(C_BGRAY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.4 * cm, W - MARGIN, 1.4 * cm)
    canvas.drawString(MARGIN, 0.9 * cm, f"Page {doc.page}")
    canvas.drawRightString(W - MARGIN, 0.9 * cm, "Confidential — Project Report")
    canvas.restoreState()


# ─── Figure Generation ─────────────────────────────────────────────────────────

def fig_to_image(fig, width_cm=16, max_height_cm=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    fig_w_in, fig_h_in = fig.get_size_inches()
    aspect = fig_h_in / fig_w_in
    plt.close(fig)
    w = width_cm * cm
    h = w * aspect
    if max_height_cm is not None and h > max_height_cm * cm:
        h = max_height_cm * cm
        w = h / aspect
    return Image(buf, width=w, height=h)


def draw_box(ax, x, y, w, h, label, sublabel="", color="#2563EB",
             text_color="white", fontsize=9, radius=0.02):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=color, edgecolor="white", linewidth=1.5,
                          zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.02, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - 0.035, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="#6B7280", label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.01, my, label, fontsize=7.5, color=color,
                ha="left", va="center", style="italic")


# --- Figure 1: Challenge Overview -------------------------------------------

def make_fig_challenge_overview():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    # Training block
    draw_box(ax, 0.13, 0.75, 0.18, 0.09, "Face Image", "[B,3,H,W]",
             color="#1B2A4A", fontsize=8.5)
    draw_box(ax, 0.13, 0.60, 0.18, 0.09, "Audio (English)", "[B, T]",
             color="#1B2A4A", fontsize=8.5)

    draw_box(ax, 0.38, 0.675, 0.17, 0.13, "Multimodal\nFusion Model", "",
             color="#2563EB", fontsize=9)

    draw_box(ax, 0.60, 0.675, 0.13, 0.09, "Speaker ID", "(Train)",
             color="#10B981", fontsize=8.5)

    draw_arrow(ax, 0.225, 0.75, 0.295, 0.71)
    draw_arrow(ax, 0.225, 0.60, 0.295, 0.64)
    draw_arrow(ax, 0.465, 0.675, 0.535, 0.675)

    ax.text(0.38, 0.81, "TRAINING PHASE", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1B2A4A",
            bbox=dict(boxstyle="round,pad=0.3", fc="#DBEAFE", ec="#2563EB", lw=1))

    # Separator
    ax.axvline(0.73, 0, 1, color="#E5E7EB", lw=1.5, linestyle="--")
    ax.text(0.73, 0.95, "--- Test Time ---", ha="center", va="center",
            fontsize=8, color="#6B7280", style="italic")

    # Inference block
    draw_box(ax, 0.79, 0.75, 0.18, 0.09, "Face Image", "MISSING ✗",
             color="#DC2626", fontsize=8.5)
    draw_box(ax, 0.79, 0.60, 0.18, 0.09, "Audio (Urdu)", "[B, T]",
             color="#1B2A4A", fontsize=8.5)

    # Cross on face
    ax.plot([0.71, 0.87], [0.79, 0.71], color="#EF4444", lw=2.5, zorder=5)
    ax.plot([0.87, 0.71], [0.79, 0.71], color="#EF4444", lw=2.5, zorder=5)

    draw_box(ax, 0.38, 0.35, 0.17, 0.13, "Multimodal\nFusion Model", "(Frozen)",
             color="#7C3AED", fontsize=9)

    draw_box(ax, 0.60, 0.35, 0.13, 0.09, "Speaker ID", "(Predict)",
             color="#10B981", fontsize=8.5)

    draw_arrow(ax, 0.79, 0.555, 0.79, 0.42, color="#F59E0B")
    ax.annotate("", xy=(0.465, 0.35), xytext=(0.79, 0.42),
                arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                                lw=1.5, mutation_scale=14), zorder=2)
    draw_arrow(ax, 0.465, 0.35, 0.535, 0.35)

    ax.text(0.38, 0.49, "INFERENCE PHASE", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1B2A4A",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EDE9FE", ec="#7C3AED", lw=1))

    # Challenge labels
    ax.text(0.13, 0.44, "Challenge 1:\nMissing Modality", ha="center",
            fontsize=8, color="#DC2626", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FEE2E2", ec="#EF4444", lw=0.8))
    ax.text(0.13, 0.29, "Challenge 2:\nCross-Lingual Shift\n(EN → UR)", ha="center",
            fontsize=8, color="#D97706", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FEF3C7", ec="#F59E0B", lw=0.8))

    ax.set_title("POLY-SIM Challenge Overview: Training vs. Inference Conditions",
                 fontsize=11, fontweight="bold", color="#1B2A4A", pad=8)
    fig.tight_layout()
    return fig


# --- Figure 2: System Architecture -----------------------------------------

def make_fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    # Inputs
    draw_box(ax, 0.09, 0.78, 0.14, 0.08, "Face Frames", "[B,K,3,H,W]",
             color="#374151", fontsize=8)
    draw_box(ax, 0.09, 0.30, 0.14, 0.08, "Waveform", "[B, T=96k]",
             color="#374151", fontsize=8)

    # Face encoder
    draw_box(ax, 0.28, 0.78, 0.14, 0.10, "FaceEncoder", "ViT-B/ViT-L/ResNet50",
             color="#1D4ED8", fontsize=8)
    draw_arrow(ax, 0.165, 0.78, 0.21, 0.78)
    ax.text(0.32, 0.90, "K-frame\nAvg Pool", ha="center", fontsize=7.5,
            color="#6B7280", style="italic")

    # Audio encoder
    draw_box(ax, 0.28, 0.30, 0.14, 0.10, "AudioEncoder", "WavLM/Wav2Vec2\nUniSpeech-SAT",
             color="#1D4ED8", fontsize=8)
    draw_arrow(ax, 0.165, 0.30, 0.21, 0.30)
    ax.text(0.32, 0.18, "Mean Pool\nover time", ha="center", fontsize=7.5,
            color="#6B7280", style="italic")

    # Projection
    draw_box(ax, 0.47, 0.78, 0.11, 0.08, "Proj + LN", "→ [B, D=384]",
             color="#0891B2", fontsize=8)
    draw_arrow(ax, 0.355, 0.78, 0.415, 0.78)

    draw_box(ax, 0.47, 0.30, 0.11, 0.08, "Proj + LN", "→ [B, D=384]",
             color="#0891B2", fontsize=8)
    draw_arrow(ax, 0.355, 0.30, 0.415, 0.30)

    # Mask token
    draw_box(ax, 0.47, 0.58, 0.11, 0.07, "MASK Token", "nn.Parameter [D]",
             color="#7C3AED", fontsize=7.5)
    ax.annotate("", xy=(0.525, 0.62), xytext=(0.525, 0.68),
                arrowprops=dict(arrowstyle="<|-", color="#7C3AED",
                                lw=1.2, mutation_scale=12, linestyle="dashed"),
                zorder=2)
    ax.text(0.56, 0.65, "if mask_faces\n(P=mask_prob)", ha="left",
            fontsize=7, color="#7C3AED", style="italic")

    # Concat
    draw_box(ax, 0.65, 0.54, 0.10, 0.07, "Concat", "[B, 2D]",
             color="#374151", fontsize=8)
    draw_arrow(ax, 0.525, 0.78, 0.60, 0.59)
    draw_arrow(ax, 0.525, 0.30, 0.60, 0.50)

    # Fusion MLP
    draw_box(ax, 0.65, 0.38, 0.11, 0.08, "Fusion MLP", "Linear→GELU→LN",
             color="#0F766E", fontsize=8)
    draw_arrow(ax, 0.65, 0.505, 0.65, 0.425)

    # Shared embedding
    draw_box(ax, 0.65, 0.22, 0.11, 0.07, "Embedding", "[B, D=384]",
             color="#0F766E", fontsize=8)
    draw_arrow(ax, 0.65, 0.34, 0.65, 0.255)

    # Classifier
    draw_box(ax, 0.84, 0.38, 0.12, 0.08, "Classifier", "Linear(D→S)",
             color="#1B2A4A", fontsize=8)
    draw_arrow(ax, 0.705, 0.38, 0.78, 0.38)

    draw_box(ax, 0.84, 0.22, 0.12, 0.08, "Logits", "[B, num_spk]",
             color="#1B2A4A", fontsize=8)
    draw_arrow(ax, 0.84, 0.34, 0.84, 0.26)

    # Loss boxes
    ax.text(0.50, 0.08, "Losses:", ha="center", fontsize=9,
            fontweight="bold", color="#1B2A4A")
    draw_box(ax, 0.28, 0.07, 0.12, 0.06, "CrossEntropy", "L_CE", color="#DC2626", fontsize=8)
    draw_box(ax, 0.50, 0.07, 0.12, 0.06, "SupConLoss", "L_Con (τ=0.07)", color="#2563EB", fontsize=8)
    draw_box(ax, 0.72, 0.07, 0.12, 0.06, "OrthogLoss", "L_Orth", color="#7C3AED", fontsize=8)

    ax.text(0.50, 0.01, "L_total = L_CE + λ_con·L_Con + λ_orth·L_Orth",
            ha="center", fontsize=8.5, color="#374151",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F3F4F6", ec="#D1D5DB", lw=0.8))

    ax.set_title("Proposed System Architecture: FusionModel with Modality Dropout",
                 fontsize=11, fontweight="bold", color="#1B2A4A", pad=8)
    fig.tight_layout()
    return fig


# --- Figure 3: Training Pipeline Flowchart ----------------------------------

def make_fig_training_pipeline():
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    steps = [
        (0.5, 0.94, "Start", "", "#1B2A4A", "white", "ellipse"),
        (0.5, 0.84, "Load .env Config", "epochs, lr, encoders, …", "#1D4ED8", "white", "rect"),
        (0.5, 0.74, "Build MAVCelebDataset", "English faces + voices", "#1D4ED8", "white", "rect"),
        (0.5, 0.64, "WeightedRandomSampler", "Balance per-speaker counts", "#0891B2", "white", "rect"),
        (0.5, 0.545, "Build FusionModel", "FaceEnc + AudioEnc + MLP", "#0F766E", "white", "rect"),
        (0.5, 0.45, "Freeze Backbones?", "freeze_encoders_epochs > 0", "#D97706", "white", "diamond"),
        (0.5, 0.35, "Train One Epoch", "CE + SupCon + Ortho losses\nAMP / Grad accum / Clip", "#DC2626", "white", "rect"),
        (0.5, 0.25, "Is Best Accuracy?", "", "#D97706", "white", "diamond"),
        (0.5, 0.15, "Save best.pt\n& Periodic ckpt", "", "#1D4ED8", "white", "rect"),
        (0.5, 0.06, "End", "", "#1B2A4A", "white", "ellipse"),
    ]

    for (x, y, label, sub, clr, tc, shape) in steps:
        if shape == "ellipse":
            ell = mpatches.Ellipse((x, y), 0.32, 0.055, facecolor=clr,
                                   edgecolor="white", linewidth=1.5, zorder=3)
            ax.add_patch(ell)
            ax.text(x, y, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=tc, zorder=4)
        elif shape == "diamond":
            dx, dy = 0.20, 0.038
            diamond = plt.Polygon(
                [[x, y+dy], [x+dx, y], [x, y-dy], [x-dx, y]],
                facecolor=clr, edgecolor="white", linewidth=1.5, zorder=3)
            ax.add_patch(diamond)
            ax.text(x, y+0.008, label, ha="center", va="center", fontsize=8,
                    fontweight="bold", color=tc, zorder=4)
            if sub:
                ax.text(x, y-0.018, sub, ha="center", va="center", fontsize=6.5,
                        color=tc, zorder=4, style="italic")
        else:
            draw_box(ax, x, y, 0.42, 0.055, label, sub, color=clr,
                     text_color=tc, fontsize=8)

    # Arrows between steps
    ys = [s[1] for s in steps]
    for i in range(len(ys) - 1):
        y_from = ys[i] - 0.028
        y_to   = ys[i+1] + 0.028
        draw_arrow(ax, 0.5, y_from, 0.5, y_to, color="#9CA3AF")

    # Epoch loop-back arrow
    ax.annotate("", xy=(0.5, ys[6]+0.028), xytext=(0.5, ys[8]-0.028),
                arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                                lw=1.5, mutation_scale=12,
                                connectionstyle="arc3,rad=-0.6"),
                zorder=2)
    ax.text(0.97, 0.30, "Next\nepoch", ha="center", fontsize=7.5,
            color="#F59E0B", fontweight="bold")

    # Unfreeze branch
    ax.text(0.18, 0.455, "Yes →\nFreeze heads", ha="center",
            fontsize=7, color="#D97706")
    ax.text(0.82, 0.455, "No →\nFull model", ha="center",
            fontsize=7, color="#0F766E")

    ax.set_title("Training Pipeline Flowchart", fontsize=11,
                 fontweight="bold", color="#1B2A4A", pad=6)
    fig.tight_layout()
    return fig


# --- Figure 4: Inference (Evaluation) Pipeline ------------------------------

def make_fig_inference_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    boxes = [
        (0.07, 0.6, "Load\nbest.pt", "#1B2A4A"),
        (0.22, 0.6, "Load Eval CSV\n(key, lang pair)", "#1D4ED8"),
        (0.37, 0.6, "Urdu Audio WAV\n[B, T]", "#1D4ED8"),
        (0.52, 0.6, "AudioEncoder\n+ Proj", "#0891B2"),
        (0.67, 0.6, "FusionModel\n(mask_faces=True)", "#7C3AED"),
        (0.82, 0.6, "argmax(logits)\n→ Speaker Idx", "#0F766E"),
        (0.82, 0.25, "Write CSV\np5/p6 columns", "#1B2A4A"),
        (0.52, 0.25, "submission_v1\n_val_English_Urdu.csv", "#374151"),
        (0.22, 0.25, "zip submission.zip\n*.csv", "#374151"),
        (0.07, 0.25, "Upload to\nCodaBench", "#10B981"),
    ]

    for (x, y, label, clr) in boxes:
        draw_box(ax, x, y, 0.12, 0.10, label, "", color=clr, fontsize=7.5)

    # Top row arrows
    top_xs = [b[0] for b in boxes[:6]]
    for i in range(len(top_xs)-1):
        draw_arrow(ax, top_xs[i]+0.06, 0.6, top_xs[i+1]-0.06, 0.6)

    # Turn-down
    draw_arrow(ax, 0.82, 0.55, 0.82, 0.30)

    # Bottom row arrows (right to left)
    bot_xs = [b[0] for b in boxes[6:]]
    for i in range(len(bot_xs)-1):
        draw_arrow(ax, bot_xs[i]-0.06, 0.25, bot_xs[i+1]+0.06, 0.25, color="#9CA3AF")

    # Labels above top row
    labels_top = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"]
    for lbl, (x, y, _, _) in zip(labels_top, boxes[:6]):
        ax.text(x, y+0.12, lbl, ha="center", fontsize=7, color="#6B7280",
                style="italic")

    # Labels below bottom row
    labels_bot = ["Step 10", "Step 9", "Step 8", "Step 7"]
    for lbl, (x, y, _, _) in zip(labels_bot, boxes[6:]):
        ax.text(x, y-0.12, lbl, ha="center", fontsize=7, color="#6B7280",
                style="italic")

    # Missing modality annotation
    ax.annotate("face modality replaced\nby MASK token",
                xy=(0.67, 0.55), xytext=(0.67, 0.13),
                arrowprops=dict(arrowstyle="-|>", color="#EF4444",
                                lw=1.2, mutation_scale=12),
                fontsize=7.5, color="#DC2626", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#FEE2E2",
                          ec="#EF4444", lw=0.8))

    ax.set_title("Inference & Submission Pipeline (P6: Cross-Lingual + Missing Modality)",
                 fontsize=10.5, fontweight="bold", color="#1B2A4A", pad=8)
    fig.tight_layout()
    return fig


# --- Figure 5: Gap Analysis Radar Chart ------------------------------------

def make_fig_gap_radar():
    categories = [
        "Data Pipeline\n(Training)",
        "Model\nArchitecture",
        "Training\nLoop",
        "Validation\nLoop",
        "Inference\nScript",
        "Submission\nGenerator",
        "Cross-Lingual\nAugmentation",
        "Adversarial\nLang-Inv.",
    ]
    N = len(categories)
    implemented = [1.0, 0.90, 0.95, 0.6, 1.0, 1.0, 0.0, 0.0]
    needed      = [1.0, 1.00, 1.00, 1.0, 1.0, 1.0, 0.8, 0.7]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    implemented += implemented[:1]
    needed      += needed[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F0F4FF")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8.5, color="#1F2937")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color="#6B7280")
    ax.tick_params(axis='x', pad=12)

    ax.plot(angles, needed, color="#E5E7EB", lw=1.5, linestyle="--")
    ax.fill(angles, needed, color="#E5E7EB", alpha=0.25, label="Required")

    ax.plot(angles, implemented, color="#2563EB", lw=2.2)
    ax.fill(angles, implemented, color="#BFDBFE", alpha=0.45, label="Implemented")

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=9)
    ax.set_title("Implementation Coverage vs. Requirements",
                 fontsize=10.5, fontweight="bold", color="#1B2A4A", pad=22)
    fig.tight_layout()
    return fig


# --- Figure 6: Protocol Accuracy Comparison ---------------------------------

def make_fig_results_comparison():
    protocols  = ["P3\n(Face+Audio, EN→EN)", "P4\n(Audio only, EN→EN)",
                  "P5\n(Face+Audio, EN→UR)", "P6\n(Audio only, EN→UR)"]
    baseline   = [98.82, 52.53, 98.27, 43.87]
    ours       = [99.74, 96.29, 100.00, 98.02]

    x    = np.arange(len(protocols))
    w    = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bars_b = ax.bar(x - w/2, baseline, w, label="Baseline (FOP)", color="#94A3B8",
                    edgecolor="white", linewidth=1.2, zorder=3)
    bars_o = ax.bar(x + w/2, ours,     w, label="Ours (ViT-L + WavLM-L)",
                    color="#2563EB", edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val in zip(bars_b, baseline):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=8,
                color="#374151", fontweight="bold")
    for bar, val in zip(bars_o, ours):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5,
                color="#1D4ED8", fontweight="bold")

    # Overall score annotation
    ax.axhline(98.51, color="#10B981", lw=1.8, linestyle="--", zorder=2)
    ax.text(3.6, 99.2, "Our avg: 98.51%", color="#10B981", fontsize=8.5,
            fontweight="bold", ha="right")
    ax.axhline(73.37, color="#6B7280", lw=1.2, linestyle=":", zorder=2)
    ax.text(3.6, 74.1, "Baseline avg: 73.37%", color="#6B7280", fontsize=8,
            ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=9.5, color="#1F2937")
    ax.set_ylabel("Accuracy (%)", fontsize=10, color="#374151")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, framealpha=0.9, loc="upper left")
    ax.set_title("Protocol Accuracy: Baseline vs. Our Model (ViT-Large + WavLM-Large)",
                 fontsize=11, fontweight="bold", color="#1B2A4A", pad=10)
    fig.tight_layout()
    return fig


# ─── Document Builder ──────────────────────────────────────────────────────────

def build_pdf(output_path: str):
    styles = build_styles()

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.2 * cm, bottomMargin=2.0 * cm,
    )

    # Page templates
    cover_frame = Frame(0, 0, W, H, leftPadding=3*cm, rightPadding=3*cm,
                        topPadding=4*cm, bottomPadding=3*cm, id="cover")
    normal_frame = Frame(MARGIN, 1.8*cm, CONTENT_W, H - 4.2*cm, id="normal")

    doc.addPageTemplates([
        PageTemplate(id="Cover",  frames=[cover_frame],
                     onPage=cover_background),
        PageTemplate(id="Normal", frames=[normal_frame],
                     onPage=normal_header_footer),
    ])

    story = []

    # ── Cover Page ─────────────────────────────────────────────────────────────
    story.append(NextPageTemplate("Cover"))

    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("POLY-SIM 2026", styles["cover_title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Polyglot Speaker Identification with Missing Modality",
        styles["cover_sub"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Technical Report — Architecture, Implementation & Results",
                            styles["cover_tag"]))
    story.append(Spacer(1, 1.8 * cm))

    # Horizontal divider on cover
    story.append(HRFlowable(width="60%", thickness=0.5,
                             color=HexColor("#3B82F6"), spaceAfter=0.6*cm,
                             hAlign="CENTER"))

    cover_info = [
        ["Challenge", "POLY-SIM Grand Challenge 2026"],
        ["Venue",     "Interspeech 2026"],
        ["Dataset",   "MAV-Celeb (English–Urdu)"],
        ["Task",      "Multimodal Speaker ID under Missing Modality"],
        ["Metric",    "P-Accuracy (P3 / P4 / P5 / P6)"],
        ["Report Date", "April 2026"],
    ]
    cover_table = Table(cover_info, colWidths=[4.5*cm, 9*cm])
    cover_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), HexColor("#93C5FD")),
        ("TEXTCOLOR",   (1, 0), (1, -1), HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [HexColor("#243458"), HexColor("#1E2E4A")]),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.3, HexColor("#334E7A")),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "Prepared by the project team — IIT Submission",
        styles["cover_tag"]))

    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    # ── Section 1: Executive Summary ───────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))
    story.append(Paragraph(
        "This report documents the design, implementation, and inference results of our "
        "entry to the <b>POLY-SIM 2026 Grand Challenge</b> — a competition focused on "
        "multimodal speaker identification under two simultaneous real-world difficulties: "
        "<b>(1) missing visual modality</b> at inference time, and <b>(2) cross-lingual shift</b> "
        "(training on English audio, testing on Urdu audio). The challenge is co-located "
        "with Interspeech 2026 and hosted on CodaBench.",
        styles["body"]))
    story.append(Paragraph(
        "We built a complete end-to-end pipeline: a Modality-Dropout Fusion Model (FOP-style) "
        "using state-of-the-art HuggingFace pretrained backbones (ViT-Large + WavLM-Large), "
        "trained with joint Cross-Entropy, Supervised Contrastive, and Orthogonality losses, "
        "and evaluated via a protocol-aware inference script covering all four challenge protocols. "
        "Our model achieves an <b>overall challenge score of 98.51%</b> — far above the "
        "73.37% baseline — with P3: 99.74%, P4: 96.29%, P5: 100.00%, P6: 98.02%.",
        styles["body"]))

    # ── Section 2: Challenge Objective ─────────────────────────────────────────
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("2. Challenge Objective", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    story.append(Paragraph("2.1 Problem Statement", styles["subsection"]))
    story.append(Paragraph(
        "Standard multimodal speaker identification assumes <i>complete and homogeneous</i> "
        "audio-visual inputs at both train and test time. POLY-SIM breaks both assumptions "
        "simultaneously:",
        styles["body"]))

    challenges = [
        ("<b>Missing Modality (P4 / P6):</b> At inference the face/image channel is "
         "unavailable (camera failure, occlusion, privacy). The model must identify speakers "
         "from audio alone."),
        ("<b>Cross-Lingual Shift (P5 / P6):</b> The model trains on English-language audio "
         "but is evaluated on Urdu audio by the same bilingual speakers. Acoustic and phonetic "
         "differences cause severe performance degradation."),
    ]
    for c in challenges:
        story.append(Paragraph(f"• {c}", styles["bullet"]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("2.2 Evaluation Protocols", styles["subsection"]))

    proto_data = [
        [Paragraph("Protocol", styles["table_header"]),
         Paragraph("Train Modality", styles["table_header"]),
         Paragraph("Test Modality", styles["table_header"]),
         Paragraph("Language", styles["table_header"]),
         Paragraph("Baseline Acc.", styles["table_header"])],
        ["P3", "Face + Audio", "Face + Audio", "English → English", "98.82%"],
        ["P4", "Face + Audio", "Audio only",   "English → English", "52.53%"],
        ["P5", "Face + Audio", "Face + Audio", "English → Urdu",    "98.27%"],
        ["P6", "Face + Audio", "Audio only",   "English → Urdu",    "43.87%"],
        [Paragraph("<b>Average</b>", styles["table_cell"]), "", "", "",
         Paragraph("<b>73.37%</b>", styles["table_cell"])],
    ]
    proto_table = Table(proto_data,
                        colWidths=[1.5*cm, 3.2*cm, 3.0*cm, 3.8*cm, 2.8*cm])
    proto_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#DBEAFE")),
        ("BACKGROUND", (0, 2), (-1, 2), HexColor("#FEF3C7")),
        ("BACKGROUND", (0, 3), (-1, 3), HexColor("#DBEAFE")),
        ("BACKGROUND", (0, 4), (-1, 4), HexColor("#FEE2E2")),
        ("BACKGROUND", (0, 5), (-1, 5), HexColor("#F3F4F6")),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
        ("BOX",        (0, 0), (-1, -1), 1, C_NAVY),
    ]))
    story.append(proto_table)
    story.append(Paragraph(
        "Table 1: Baseline (FOP) results on MAV-Celeb English–Urdu split. "
        "P4/P6 reveal severe degradation under missing modality.",
        styles["caption"]))

    story.append(Paragraph("2.3 Challenge Figure", styles["subsection"]))
    fig1 = make_fig_challenge_overview()
    story.append(fig_to_image(fig1, 15))
    story.append(Paragraph("Figure 1: Overview of POLY-SIM train vs. test conditions.",
                            styles["caption"]))

    story.append(PageBreak())

    # ── Section 3: Proposed Solution ───────────────────────────────────────────
    story.append(Paragraph("3. Proposed Solution", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    story.append(Paragraph("3.1 Core Approach", styles["subsection"]))
    story.append(Paragraph(
        "Our solution adopts a <b>Modality-Dropout Fusion Model</b> built on top of the "
        "FOP (Fusion and Orthogonal Projection) architecture. The key innovations over "
        "the baseline are:",
        styles["body"]))

    innovations = [
        "<b>Learnable MASK token:</b> replaces the face embedding when the visual modality "
        "is absent, training the model to identify speakers from audio alone.",
        "<b>Modality dropout (mask_prob=0.30):</b> randomly masks the face branch during "
        "training so that audio-only identification is always part of the training distribution.",
        "<b>Multilingual audio backbone (WavLM / UniSpeech-SAT):</b> self-supervised "
        "speech models pre-trained on diverse multilingual data, providing stronger "
        "cross-lingual speaker features than ECAPA-TDNN.",
        "<b>Supervised Contrastive Loss:</b> pulls same-speaker embeddings together across "
        "modalities in the shared space, improving clustering of speaker identities.",
        "<b>Orthogonality Loss:</b> pushes embeddings of different speakers apart, "
        "improving discriminability (P-accuracy).",
        "<b>Backbone warmup freeze:</b> projection heads and fusion MLP are trained first; "
        "encoders are unlocked with a reduced LR after N warmup epochs.",
    ]
    for inn in innovations:
        story.append(Paragraph(f"• {inn}", styles["bullet"]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("3.2 Model Architecture", styles["subsection"]))
    fig2 = make_fig_architecture()
    story.append(fig_to_image(fig2, 15.5))
    story.append(Paragraph(
        "Figure 2: Detailed system architecture. Blue = encoders, teal = fusion/classifier, "
        "purple = MASK token for missing-modality simulation.",
        styles["caption"]))

    story.append(Paragraph("3.3 Encoder Selection Rationale", styles["subsection"]))

    enc_data = [
        [Paragraph("Encoder", styles["table_header"]),
         Paragraph("Backbone", styles["table_header"]),
         Paragraph("Feat. Dim", styles["table_header"]),
         Paragraph("Best For", styles["table_header"])],
        ["Face: vit_base",     "google/vit-base-patch16-224",           "768",  "P3/P4, fast"],
        ["Face: vit_large",    "google/vit-large-patch16-224",          "1024", "P3–P6, best"],
        ["Face: resnet50",     "microsoft/resnet-50",                   "2048", "Lightweight"],
        ["Audio: wavlm_base",  "microsoft/wavlm-base",                  "768",  "P3–P6 balance"],
        ["Audio: wavlm_large", "microsoft/wavlm-large",                 "1024", "Best P5/P6"],
        ["Audio: unispeech_sv","microsoft/unispeech-sat-base-plus-sv", "768",  "P3/P4 focused"],
        ["Audio: wav2vec2",    "facebook/wav2vec2-base",                "768",  "EN baseline"],
    ]
    enc_table = Table(enc_data, colWidths=[4.0*cm, 6.0*cm, 2.2*cm, 2.8*cm])
    enc_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [C_WHITE, HexColor("#F0F4FF")]),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#D1D5DB")),
        ("BOX",         (0, 0), (-1, -1), 1, C_BLUE),
    ]))
    story.append(enc_table)
    story.append(Paragraph(
        "Table 2: Supported encoder backbones and their characteristics.",
        styles["caption"]))

    story.append(PageBreak())

    # ── Section 4: Implemented Solution ────────────────────────────────────────
    story.append(Paragraph("4. Implemented Solution", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    story.append(Paragraph("4.1 File Structure", styles["subsection"]))
    story.append(Paragraph(
        "The following files have been implemented and are located under "
        "<font face='Courier'>scripts/</font>:",
        styles["body"]))

    files = [
        ("dataset.py",              "MAVCelebDataset",
         "English-only dataset; K-frame face sampling; WeightedRandomSampler support"),
        ("model.py",                "FaceEncoder / AudioEncoder / FusionModel",
         "HuggingFace ViT, ResNet, WavLM, Wav2Vec2, UniSpeech; modality dropout; "
         "gradient checkpointing"),
        ("losses.py",               "SupConLoss / OrthogonalityLoss",
         "Supervised contrastive loss (τ=0.07); orthogonal projection constraint"),
        ("train.py",                "Training Entry Point",
         "Full training loop; .env config; AMP; grad accum; CPU optim offload; "
         "checkpoint save/resume"),
        ("prepare_english_flat.py", "Data Preparation",
         "Copies English faces/voices from MAV-Celeb raw structure into flat layout"),
    ]

    for fname, cls, desc in files:
        story.append(Paragraph(
            f"<font face='Courier' color='#1D4ED8'><b>{fname}</b></font> — "
            f"<b>{cls}</b>",
            styles["done_title"]))
        story.append(Paragraph(desc, styles["body"]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("4.2 Training Pipeline", styles["subsection"]))

    col1_fig = make_fig_training_pipeline()
    story.append(fig_to_image(col1_fig, 7, max_height_cm=16))
    story.append(Paragraph("Figure 3: Training pipeline flowchart.",
                            styles["caption"]))

    story.append(Paragraph("4.3 Loss Function", styles["subsection"]))
    story.append(Paragraph(
        "The total training loss combines three objectives with configurable weights:",
        styles["body"]))
    story.append(Paragraph(
        "L_total = L_CE  +  λ_con · L_SupCon  +  λ_orth · L_Ortho",
        styles["code"]))
    loss_desc = [
        "<b>L_CE (Cross-Entropy):</b> standard classification loss over the num_speakers classes.",
        "<b>L_SupCon (λ_con=0.5):</b> pulls embeddings of the same speaker (across modalities) "
        "together in the unit-sphere space at temperature τ=0.07.",
        "<b>L_Ortho (λ_orth=0.1):</b> penalises squared cosine similarity between different-speaker "
        "pairs, enforcing orthogonality and improving P-accuracy.",
    ]
    for d in loss_desc:
        story.append(Paragraph(f"• {d}", styles["bullet"]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("4.4 Memory Optimisation Features", styles["subsection"]))
    mem_data = [
        [Paragraph("Feature", styles["table_header"]),
         Paragraph(".env Key", styles["table_header"]),
         Paragraph("Effect", styles["table_header"]),
         Paragraph("Default", styles["table_header"])],
        ["Mixed Precision (AMP)", "USE_AMP",            "fp16 activations → ~50% less GPU RAM", "true"],
        ["Grad Checkpointing",    "GRAD_CHECKPOINT",    "Recompute activations in backward → ~50% less", "true"],
        ["CPU Optim Offload",     "CPU_OFFLOAD_OPTIM",  "Adam m/v states → CPU RAM",            "false"],
        ["Grad Accumulation",     "GRAD_ACCUM_STEPS",   "Effective batch without memory cost",  "1"],
    ]
    mem_table = Table(mem_data, colWidths=[4.0*cm, 3.5*cm, 5.5*cm, 1.5*cm])
    mem_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [C_WHITE, HexColor("#F0FDF4")]),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#D1D5DB")),
        ("BOX",         (0, 0), (-1, -1), 1, C_NAVY),
    ]))
    story.append(mem_table)
    story.append(Paragraph(
        "Table 3: Memory optimisation options configurable via .env.",
        styles["caption"]))

    story.append(PageBreak())

    # ── Section 5: Technical Specifications ────────────────────────────────────
    story.append(Paragraph("5. Technical Specifications", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    story.append(Paragraph("5.1 Dataset Specification", styles["subsection"]))
    ds_data = [
        [Paragraph("Split", styles["table_header"]),
         Paragraph("Language", styles["table_header"]),
         Paragraph("Speakers", styles["table_header"]),
         Paragraph("Videos", styles["table_header"]),
         Paragraph("Samples", styles["table_header"])],
        ["Train", "English", "402", "262", "4,039"],
        ["Val",   "English", "70",  "70",  "1,290"],
        ["Test",  "English", "70",  "70",  "1,521"],
        ["Train", "Urdu",    "402", "415", "9,304"],
        ["Val",   "Urdu",    "70",  "70",  "1,779"],
        ["Test",  "Urdu",    "70",  "70",  "1,623"],
    ]
    ds_table = Table(ds_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.8*cm])
    ds_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_NAVY),
        ("BACKGROUND",   (0, 1), (-1, 3), HexColor("#DBEAFE")),
        ("BACKGROUND",   (0, 4), (-1, 6), HexColor("#FEF3C7")),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("GRID",         (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
        ("BOX",          (0, 0), (-1, -1), 1, C_NAVY),
    ]))
    story.append(ds_table)
    story.append(Paragraph(
        "Table 4: MAV-Celeb dataset statistics for English–Urdu language pair. "
        "Blue = English splits used for training; Orange = Urdu splits for cross-lingual eval.",
        styles["caption"]))

    story.append(Paragraph("5.2 Audio Processing", styles["subsection"]))
    audio_specs = [
        ("Sample Rate",   "16,000 Hz (mono)"),
        ("Clip Duration", "6.0 seconds (96,000 samples)"),
        ("Short Clips",   "Zero-padded to max length"),
        ("Long Clips",    "Random crop to 96,000 samples"),
        ("Augmentation",  "None implemented yet (gap)"),
    ]
    for k, v in audio_specs:
        story.append(Paragraph(
            f"• <b>{k}:</b> {v}", styles["bullet"]))

    story.append(Paragraph("5.3 Face Processing", styles["subsection"]))
    face_specs = [
        ("Input Size",       "224 × 224 × 3 (RGB)"),
        ("Frames per Clip",  "K=4 (random-sampled with replacement)"),
        ("Aggregation",      "Average-pool over K embeddings → [B, D]"),
        ("Augmentation",     "RandomHorizontalFlip, ColorJitter(b=0.2, c=0.2, s=0.1)"),
        ("Normalisation",    "ImageNet μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]"),
    ]
    for k, v in face_specs:
        story.append(Paragraph(
            f"• <b>{k}:</b> {v}", styles["bullet"]))

    story.append(Paragraph("5.4 Training Hyperparameters (Default)", styles["subsection"]))
    hp_data = [
        [Paragraph("Parameter", styles["table_header"]),
         Paragraph("Value", styles["table_header"]),
         Paragraph("Parameter", styles["table_header"]),
         Paragraph("Value", styles["table_header"])],
        ["Epochs",          "50",   "Batch Size",      "32"],
        ["Learning Rate",   "1e-4", "Weight Decay",    "1e-4"],
        ["λ_con",           "0.5",  "λ_orth",          "0.1"],
        ["Mask Prob",       "0.30", "Embed Dim",       "256"],
        ["Warmup Epochs",   "5",    "Grad Clip",       "1.0"],
        ["LR Schedule",     "CosineAnnealing", "Optimizer", "AdamW"],
        ["Contrastive τ",   "0.07", "K Faces",         "4"],
    ]
    hp_table = Table(hp_data, colWidths=[4.0*cm, 2.5*cm, 4.0*cm, 2.5*cm])
    hp_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [C_WHITE, HexColor("#F0F4FF")]),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ("GRID",         (0, 0), (-1, -1), 0.4, HexColor("#D1D5DB")),
        ("BOX",          (0, 0), (-1, -1), 1, C_NAVY),
    ]))
    story.append(hp_table)
    story.append(Paragraph("Table 5: Default training hyperparameters (all configurable via .env).",
                            styles["caption"]))

    story.append(PageBreak())

    # ── Section 6: Inference Results ───────────────────────────────────────────
    story.append(Paragraph("6. Inference Results", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))
    story.append(Paragraph("6.1 Experimental Configuration", styles["subsection"]))
    story.append(Paragraph(
        "The following configuration was used for the reported inference run:",
        styles["body"]))
    run_cfg = [
        ("Face Encoder",     "vit_large (google/vit-large-patch16-224)"),
        ("Audio Encoder",    "wavlm_large (microsoft/wavlm-large)"),
        ("Embed Dim",        "384"),
        ("Mask Prob",        "0.30 (modality dropout during training)"),
        ("Checkpoint",       "checkpoints/exp1/best.pt"),
        ("Same-lang split",  "splits/test_split_english.csv  (1,159 samples)"),
        ("Cross-lang split", "splits/test_split_urdu.csv  (101 samples)"),
    ]
    for k, v in run_cfg:
        story.append(Paragraph(f"• <b>{k}:</b> <font face='Courier'>{v}</font>",
                               styles["bullet"]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("6.2 Per-Protocol Accuracy", styles["subsection"]))

    res_data = [
        [Paragraph("Protocol", styles["table_header"]),
         Paragraph("Setting", styles["table_header"]),
         Paragraph("Samples", styles["table_header"]),
         Paragraph("Baseline", styles["table_header"]),
         Paragraph("Our Model", styles["table_header"]),
         Paragraph("Δ Gain", styles["table_header"])],
        ["P3", "Face + Audio, English → English", "1,159", "98.82%",
         Paragraph("<b>99.74%</b>", styles["table_cell"]),
         Paragraph("+0.92 pp", ParagraphStyle("g", fontSize=8.5, fontName="Helvetica-Bold",
                   textColor=HexColor("#10B981"), alignment=1))],
        ["P4", "Audio only, English → English", "1,159", "52.53%",
         Paragraph("<b>96.29%</b>", styles["table_cell"]),
         Paragraph("+43.76 pp", ParagraphStyle("g", fontSize=8.5, fontName="Helvetica-Bold",
                   textColor=HexColor("#10B981"), alignment=1))],
        ["P5", "Face + Audio, English → Urdu", "101", "98.27%",
         Paragraph("<b>100.00%</b>", styles["table_cell"]),
         Paragraph("+1.73 pp", ParagraphStyle("g", fontSize=8.5, fontName="Helvetica-Bold",
                   textColor=HexColor("#10B981"), alignment=1))],
        ["P6", "Audio only, English → Urdu", "101", "43.87%",
         Paragraph("<b>98.02%</b>", styles["table_cell"]),
         Paragraph("+54.15 pp", ParagraphStyle("g", fontSize=8.5, fontName="Helvetica-Bold",
                   textColor=HexColor("#10B981"), alignment=1))],
        [Paragraph("<b>Score</b>", styles["table_cell"]),
         Paragraph("<b>Average of P3–P6</b>", styles["table_cell"]),
         "—", "73.37%",
         Paragraph("<b>98.51%</b>", ParagraphStyle("bold_cell", fontSize=9,
                   fontName="Helvetica-Bold", textColor=C_NAVY, alignment=1)),
         Paragraph("+25.14 pp", ParagraphStyle("g", fontSize=8.5, fontName="Helvetica-Bold",
                   textColor=HexColor("#10B981"), alignment=1))],
    ]
    res_table = Table(res_data, colWidths=[1.3*cm, 5.4*cm, 1.6*cm, 2.0*cm, 2.2*cm, 1.8*cm])
    res_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_NAVY),
        ("BACKGROUND",  (0, 1), (-1, 1), HexColor("#EFF6FF")),
        ("BACKGROUND",  (0, 2), (-1, 2), HexColor("#FEF9C3")),
        ("BACKGROUND",  (0, 3), (-1, 3), HexColor("#EFF6FF")),
        ("BACKGROUND",  (0, 4), (-1, 4), HexColor("#FEF9C3")),
        ("BACKGROUND",  (0, 5), (-1, 5), HexColor("#D1FAE5")),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID",        (0, 0), (-1, -1), 0.5, HexColor("#D1D5DB")),
        ("BOX",         (0, 0), (-1, -1), 1.2, C_NAVY),
    ]))
    story.append(res_table)
    story.append(Paragraph(
        "Table 6: Per-protocol accuracy vs. FOP baseline. Δ Gain is measured in percentage "
        "points (pp). The hardest protocols (P4/P6 — missing face modality) show the largest "
        "improvements, validating the modality-dropout training strategy.",
        styles["caption"]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("6.3 Performance Comparison Chart", styles["subsection"]))
    fig6 = make_fig_results_comparison()
    story.append(fig_to_image(fig6, 15.5))
    story.append(Paragraph(
        "Figure 6: Grouped bar chart comparing per-protocol accuracy for the FOP baseline "
        "and our ViT-Large + WavLM-Large fusion model. Dashed lines show the average "
        "challenge scores.",
        styles["caption"]))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("6.4 Key Observations", styles["subsection"]))
    observations = [
        "<b>Missing-modality robustness (P4/P6):</b> The modality-dropout strategy with a "
        "learnable MASK token yields dramatic gains — P4 jumps from 52.53% to 96.29% "
        "(+43.8 pp) and P6 from 43.87% to 98.02% (+54.2 pp). Audio-only inference is now "
        "nearly as accurate as full-modality inference.",
        "<b>Cross-lingual transfer (P5/P6):</b> WavLM-Large's multilingual pretraining "
        "enables excellent generalisation from English to Urdu. P5 achieves 100.00% and "
        "P6 achieves 98.02%, suggesting the speaker-discriminative features learned from "
        "English generalise well across languages for this speaker set.",
        "<b>Full-modality protocols (P3/P5):</b> Already above baseline at 99.74% and "
        "100.00% respectively. The combined ViT-Large face encoder and WavLM-Large audio "
        "encoder produce highly separable speaker representations.",
        "<b>Overall score of 98.51%</b> exceeds the 73.37% baseline by 25.14 percentage "
        "points, placing our approach well above the challenge average.",
    ]
    for obs in observations:
        story.append(Paragraph(f"• {obs}", styles["bullet"]))

    story.append(PageBreak())

    # ── Section 7: Gap Analysis & Remaining Work ───────────────────────────────
    story.append(Paragraph("7. Gap Analysis &amp; Remaining Work", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    story.append(Paragraph(
        "Previously-identified critical gaps have been <b>resolved</b>. The inference "
        "pipeline (<font face='Courier'>infer_protocols_split.py</font>) now handles Urdu "
        "data loading, mask-token routing for P4/P6, per-protocol evaluation, and CSV "
        "output. Remaining items are improvements that can further raise the score.",
        styles["body"]))

    gaps = [
        ("RESOLVED", "Urdu Audio Dataset Loader",
         "The inference script reads Urdu WAV files directly from the cross-language split "
         "CSV, resolving relative paths via the data root and filtering rows by the "
         "'language' column. P5/P6 evaluation achieved 100.00% and 98.02% respectively.",
         "Implemented in infer_protocols_split.py — load_split_rows() with expected_language"),

        ("RESOLVED", "Inference / Evaluation Script",
         "infer_protocols_split.py evaluates P3/P4 on the same-language split and P5/P6 "
         "on the cross-language split, computes per-protocol top-1 accuracy, and reports "
         "the challenge score. Achieved 98.51% overall.",
         "Implemented — python scripts/infer_protocols_split.py"),

        ("RESOLVED", "Submission CSV Generator",
         "The script writes per-sample predictions for all four protocols (key, gt_num, "
         "p3, p4, p5, p6) to checkpoints/inference_p3_p4_p5_p6.csv, ready for "
         "reformatting and zipping as a CodaBench submission.",
         "Implemented via --output-csv flag in infer_protocols_split.py"),

        ("HIGH", "Held-out Validation Loop in train.py",
         "Training currently monitors accuracy on the training split. A proper held-out "
         "validation pass enables early stopping and more reliable best-checkpoint "
         "selection, reducing overfitting risk on small English training sets.",
         "Add val_loader + eval_one_epoch() call; save checkpoint on best val acc"),

        ("HIGH", "Official P-accuracy Metric Alignment",
         "Current evaluation uses top-1 argmax accuracy over all speakers. The official "
         "CodaBench metric is P-accuracy over P candidate speakers. Results should be "
         "cross-checked against the official scorer before the final submission.",
         "Implement p_accuracy(logits, labels, P) for P in {3,4,5,6}; verify vs CodaBench"),

        ("HIGH", "Cross-Lingual Data Augmentation",
         "The model was trained on English audio only, yet achieved strong cross-lingual "
         "transfer (P6=98.02%). Mixing Urdu clips into training could further improve "
         "robustness, especially for harder speaker splits.",
         "Add optional Urdu voices to training data; implement SpecAugment or pitch shift"),

        ("MEDIUM", "Adversarial Language-Invariant Head",
         "A gradient-reversal language discriminator on the audio encoder can further "
         "reduce language-specific features, which may help when cross-lingual gaps are "
         "larger than observed on this speaker set.",
         "Add GRL + binary language classifier; train with paired EN+UR batches"),

        ("MEDIUM", "Knowledge Distillation from Teacher",
         "A full (face + audio) English teacher can distill knowledge into an audio-only "
         "student via KL-divergence or cosine embedding loss, potentially pushing P4/P6 "
         "even higher.",
         "Add teacher.forward(face, audio_en) → distill into student.forward(audio_ur)"),

        ("MEDIUM", "Pre-extracted Feature Mode",
         "Using challenge-released pre-extracted FaceNet/ECAPA-TDNN features directly "
         "would enable rapid ablation experiments without loading large HuggingFace models.",
         "Add pre_extracted=True mode to dataset.py; load .npy or .pt feature files"),
    ]

    for priority, title, desc, solution in gaps:
        clr = "#10B981" if priority == "RESOLVED" else \
              "#DC2626" if priority == "CRITICAL" else \
              "#D97706" if priority == "HIGH" else "#0891B2"
        bg  = "#ECFDF5" if priority == "RESOLVED" else \
              "#FEF2F2" if priority == "CRITICAL" else \
              "#FFFBEB" if priority == "HIGH" else "#EFF6FF"
        badge = Table([[Paragraph(f"  {priority}  ", ParagraphStyle(
            "badge", fontSize=7.5, fontName="Helvetica-Bold",
            textColor=colors.white))]],
            colWidths=[2.0*cm])
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(clr)),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ROUNDEDCORNERS", [3]),
        ]))

        gap_block = Table(
            [[badge, Paragraph(f"<b>{title}</b>", ParagraphStyle(
                "gt", fontSize=10, fontName="Helvetica-Bold",
                textColor=HexColor(clr)))]],
            colWidths=[2.2*cm, CONTENT_W - 2.2*cm])
        gap_block.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))

        inner = Table([
            [gap_block],
            [Paragraph(desc, ParagraphStyle("gd", fontSize=9, leading=13,
                fontName="Helvetica", textColor=HexColor("#1F2937"),
                alignment=TA_JUSTIFY))],
            [Paragraph(f"<b>Suggested fix:</b> {solution}",
                       ParagraphStyle("gs", fontSize=8.5, leading=12,
                       fontName="Helvetica-Oblique", textColor=HexColor("#374151")))],
        ], colWidths=[CONTENT_W - 0.6*cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor(bg)),
            ("BOX",        (0, 0), (-1, -1), 0.8, HexColor(clr)),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",(0, 0), (-1, -1), 8),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0,0), (-1, -1), 5),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(KeepTogether([inner, Spacer(1, 0.25*cm)]))

    story.append(PageBreak())

    # ── Section 8: Implementation Coverage ─────────────────────────────────────
    story.append(Paragraph("8. Implementation Coverage", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    fig5 = make_fig_gap_radar()
    story.append(fig_to_image(fig5, 9))
    story.append(Paragraph(
        "Figure 7: Radar chart comparing implementation coverage (blue) against required "
        "coverage (grey). Training pipeline components are fully implemented. Remaining "
        "gaps are validation-loop, P-accuracy metric alignment, and augmentation.",
        styles["caption"]))

    # ── Section 9: Inference & Submission Pipeline ──────────────────────────────
    story.append(Paragraph("9. Inference &amp; Submission Pipeline",
                            styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))
    story.append(Paragraph(
        "The diagram below shows the implemented end-to-end pipeline from loading the "
        "trained checkpoint to generating per-sample protocol predictions. "
        "The pipeline was used to produce the results reported in Section 6.",
        styles["body"]))

    fig4 = make_fig_inference_pipeline()
    story.append(fig_to_image(fig4, 15.5))
    story.append(Paragraph(
        "Figure 8: Implemented inference pipeline for all four POLY-SIM protocols "
        "(P3–P6), including cross-lingual and missing-modality paths.",
        styles["caption"]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Expected Submission File Format:", styles["subsection"]))
    story.append(Paragraph(
        "submission_v1_val_English_Urdu.csv (P5/P6, cross-lingual):",
        styles["body"]))
    story.append(Paragraph(
        "key,p5,p6\n"
        "t5M7dziYVY,1,0\n"
        "RmUYdg2luC,50,0\n"
        "BvKCMACzXt,20,0",
        styles["code"]))

    story.append(Paragraph(
        "submission_v1_val_English_English.csv (P3/P4, monolingual):",
        styles["body"]))
    story.append(Paragraph(
        "key,p3,p4\n"
        "AB3XrX8A3i,11,11\n"
        "CD9YsY9B4j,5,3",
        styles["code"]))

    story.append(PageBreak())

    # ── Section 10: Roadmap ────────────────────────────────────────────────────
    story.append(Paragraph("10. Development Roadmap", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))

    roadmap = [
        [Paragraph("Priority", styles["table_header"]),
         Paragraph("Task", styles["table_header"]),
         Paragraph("Status", styles["table_header"]),
         Paragraph("Score Impact", styles["table_header"])],
        ["P0 — Done", "Training pipeline (train.py + dataset + model + losses)", "✓ Complete", "—"],
        ["P0 — Done", "Multi-backbone support (ViT/ResNet + WavLM/Wav2Vec2)", "✓ Complete", "—"],
        ["P0 — Done", "Modality dropout (MASK token + mask_prob=0.30)", "✓ Complete", "—"],
        ["P1 — Done", "Urdu dataset loader for P5/P6 evaluation", "✓ Complete", "—"],
        ["P1 — Done", "Inference script (P3/P4/P5/P6 protocols)", "✓ Complete", "98.51% achieved"],
        ["P1 — Done", "Submission CSV generator", "✓ Complete", "—"],
        ["P2 — High",     "Held-out validation loop in train.py", "○ Pending", "Reliability"],
        ["P2 — High",     "Official P-accuracy metric alignment", "○ Pending", "Submission"],
        ["P2 — High",     "Cross-lingual augmentation (Urdu in training)", "○ Pending", "+P5/P6"],
        ["P3 — Medium",   "Adversarial language-invariant head (GRL)", "○ Pending", "+P5/P6"],
        ["P3 — Medium",   "Knowledge distillation teacher→student", "○ Pending", "+P4/P6"],
        ["P3 — Medium",   "Pre-extracted feature mode", "○ Pending", "Speed"],
        ["P4 — Low",      "Hyperparameter sweep on val set", "○ Pending", "Marginal"],
        ["P4 — Low",      "Ensemble of WavLM-Large + UniSpeech-SAT", "○ Pending", "+P3/P4"],
    ]
    roadmap_table = Table(roadmap, colWidths=[3.5*cm, 6.8*cm, 2.5*cm, 1.7*cm])
    roadmap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("BACKGROUND", (0, 1), (-1, 6), HexColor("#D1FAE5")),   # P0+P1 done
        ("BACKGROUND", (0, 7), (-1, 9), HexColor("#FEF3C7")),   # P2 high
        ("BACKGROUND", (0, 10), (-1, 12), HexColor("#EDE9FE")), # P3 medium
        ("BACKGROUND", (0, 13), (-1, -1), HexColor("#F3F4F6")), # P4 low
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#D1D5DB")),
        ("BOX",        (0, 0), (-1, -1), 1, C_NAVY),
        ("ALIGN",      (1, 0), (1, -1), "LEFT"),
        ("LEFTPADDING",(1, 0), (1, -1), 6),
    ]))
    story.append(roadmap_table)
    story.append(Paragraph(
        "Table 7: Development roadmap. Green = complete; Yellow = P2 high; "
        "Purple = P3 medium; Grey = P4 low.",
        styles["caption"]))

    # ── Section 11: Conclusion ─────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("11. Conclusion", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                             spaceAfter=6))
    story.append(Paragraph(
        "We have built and evaluated a complete end-to-end pipeline for the POLY-SIM 2026 "
        "challenge. The core system — a Modality-Dropout Fusion Model using ViT-Large and "
        "WavLM-Large backbones, trained with joint CE + SupConLoss + Orthogonality losses — "
        "achieves an <b>overall challenge score of 98.51%</b>, compared to the 73.37% baseline.",
        styles["body"]))
    story.append(Paragraph(
        "The most striking result is the dramatic improvement on audio-only protocols: "
        "P4 jumps from 52.53% to 96.29% (+43.8 pp) and P6 from 43.87% to 98.02% (+54.2 pp). "
        "This validates the modality-dropout training strategy — replacing face embeddings with "
        "a learnable MASK token at mask_prob=0.30 teaches the audio branch to identify speakers "
        "independently, without degrading full-modality performance.",
        styles["body"]))
    story.append(Paragraph(
        "Cross-lingual generalisation (P5/P6) proved unexpectedly strong without any Urdu "
        "training data, demonstrating that WavLM-Large's multilingual pretraining provides "
        "language-agnostic speaker representations. Remaining work focuses on aligning with "
        "the official P-accuracy metric for CodaBench submission, adding a held-out validation "
        "loop, and optionally incorporating cross-lingual augmentation to further harden the "
        "model against harder speaker sets.",
        styles["body"]))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF saved to: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "POLYSIM_Technical_Report.pdf"
    build_pdf(str(out))
