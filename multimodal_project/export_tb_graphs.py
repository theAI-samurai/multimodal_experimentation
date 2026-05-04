"""Export TensorBoard scalars as PNG graphs for documentation."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTDIR = "tb_graphs"
os.makedirs(OUTDIR, exist_ok=True)


def load_ea(logdir: str) -> tuple[EventAccumulator, set[str]]:
    ea = EventAccumulator(logdir)
    ea.Reload()
    return ea, set(ea.Tags()["scalars"])


# ── Training / validation run ────────────────────────────────────────────────
ea_train, train_tags = load_ea("runs/exp1")

train_groups = {
    "Loss_train_val": {
        "title": "Loss — Train vs Val",
        "series": {"Train Loss": "Loss/train", "Val Loss": "Loss/val"},
        "ylabel": "Loss",
    },
    "Accuracy_train_val": {
        "title": "Accuracy — Train vs Val",
        "series": {"Train Accuracy": "Accuracy/train", "Val Accuracy": "Accuracy/val"},
        "ylabel": "Accuracy",
    },
    "CE_train_val": {
        "title": "Cross-Entropy Loss — Train vs Val",
        "series": {"Train CE": "CE/train", "Val CE": "CE/val"},
        "ylabel": "CE Loss",
    },
    "ConLoss_train_val": {
        "title": "Contrastive Loss — Train vs Val",
        "series": {"Train ConLoss": "ConLoss/train", "Val ConLoss": "ConLoss/val"},
        "ylabel": "Contrastive Loss",
    },
    "OrthLoss_train_val": {
        "title": "Orthogonality Loss — Train vs Val",
        "series": {"Train OrthLoss": "OrthLoss/train", "Val OrthLoss": "OrthLoss/val"},
        "ylabel": "Orth Loss",
    },
    "LearningRate": {
        "title": "Learning Rate Schedule",
        "series": {"Learning Rate": "LearningRate"},
        "ylabel": "LR",
    },
    "EvalSplit_Accuracy": {
        "title": "Eval-Split Accuracy during Training (P3–P6)",
        "series": {
            "P3 Accuracy": "EvalSplit/P3_Accuracy",
            "P4 Accuracy": "EvalSplit/P4_Accuracy",
            "P5 Accuracy": "EvalSplit/P5_Accuracy",
            "P6 Accuracy": "EvalSplit/P6_Accuracy",
        },
        "ylabel": "Accuracy (%)",
    },
    "EvalSplit_ChallengeScore": {
        "title": "Challenge Score during Training",
        "series": {"Challenge Score": "EvalSplit/ChallengeScore"},
        "ylabel": "Score (%)",
    },
    "BatchLoss_train": {
        "title": "Batch-Level Loss Components (Train)",
        "series": {
            "Batch Loss":     "BatchLoss/train",
            "Batch CE":       "BatchCE/train",
            "Batch ConLoss":  "BatchConLoss/train",
            "Batch OrthLoss": "BatchOrthLoss/train",
        },
        "ylabel": "Loss",
    },
    "BatchAccuracy_train": {
        "title": "Batch-Level Accuracy (Train)",
        "series": {"Batch Accuracy": "BatchAccuracy/train"},
        "ylabel": "Accuracy",
    },
}

# ── Test-set inference run ───────────────────────────────────────────────────
ea_infer, infer_tags = load_ea("runs/infer_test")

infer_groups = {
    "Test_Protocol_Accuracy": {
        "title": "Test Accuracy — P3 / P4 / P5 / P6",
        "series": {
            "P3 (face+audio, English)": "EvalSplit/P3_Accuracy",
            "P4 (audio-only, English)": "EvalSplit/P4_Accuracy",
            "P5 (face+audio, Urdu)":    "EvalSplit/P5_Accuracy",
            "P6 (audio-only, Urdu)":    "EvalSplit/P6_Accuracy",
        },
        "ylabel": "Accuracy (%)",
        "bar": True,
    },
    "Test_ChallengeScore": {
        "title": "Test Challenge Score",
        "series": {"Challenge Score": "EvalSplit/ChallengeScore"},
        "ylabel": "Score (%)",
        "bar": True,
    },
    "Test_CE_Loss": {
        "title": "Test CE Loss — P3 / P4 / P5 / P6",
        "series": {
            "P3 CE (face+audio, English)": "InferCE/P3_CE",
            "P4 CE (audio-only, English)": "InferCE/P4_CE",
            "P5 CE (face+audio, Urdu)":    "InferCE/P5_CE",
            "P6 CE (audio-only, Urdu)":    "InferCE/P6_CE",
        },
        "ylabel": "Mean CE Loss",
        "bar": True,
    },
}


def _plot_group(cfg: dict, ea: EventAccumulator, available_tags: set[str], outpath: str) -> bool:
    use_bar = cfg.get("bar", False)
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    if use_bar:
        labels, values = [], []
        for label, tag in cfg["series"].items():
            if tag not in available_tags:
                print(f"  [skip] {tag}")
                continue
            evs = ea.Scalars(tag)
            labels.append(label)
            values.append(evs[-1].value)
        if labels:
            bars = ax.bar(labels, values, color=plt.cm.tab10.colors[:len(labels)])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002 * max(values, default=1),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            plotted = True
    else:
        for label, tag in cfg["series"].items():
            if tag not in available_tags:
                print(f"  [skip] {tag}")
                continue
            evs = ea.Scalars(tag)
            ax.plot([e.step for e in evs], [e.value for e in evs], label=label, linewidth=1.8)
            plotted = True
        ax.legend()

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(cfg["title"], fontsize=13)
    ax.set_xlabel("Protocol" if use_bar else "Step / Epoch")
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.3, axis="y" if use_bar else "both")
    if use_bar:
        plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return True


saved = []

for fname, cfg in train_groups.items():
    path = os.path.join(OUTDIR, f"{fname}.png")
    if _plot_group(cfg, ea_train, train_tags, path):
        saved.append(path)
        print(f"Saved {path}")

for fname, cfg in infer_groups.items():
    path = os.path.join(OUTDIR, f"{fname}.png")
    if _plot_group(cfg, ea_infer, infer_tags, path):
        saved.append(path)
        print(f"Saved {path}")

print(f"\nDone — {len(saved)} graphs written to ./{OUTDIR}/")
