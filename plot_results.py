"""
Regenerate all Baseline CNN and Ablation Study plots from recorded JSON / log data.
No retraining required.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT      = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

JSON_PATH = os.path.join(OUT, "ablation_results.json")

# 12 display labels
DL = [
    "Fresh Apple", "Stale Apple",
    "Fresh Banana", "Stale Banana",
    "Fresh Orange", "Stale Orange",
    "Fresh Cucumber", "Stale Cucumber",
    "Fresh Okra", "Stale Okra",
    "Fresh Tomato", "Stale Tomato",
]

DK = "#0F1117"   # dark background
DA = "#1A1D2E"   # axes background
PALETTE = ['#4FC3F7', '#FFB74D', '#81C784', '#F06292', '#CE93D8']
EPOCHS = list(range(1, 11))

def main():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
        print(f"Loaded results from {JSON_PATH}")
        results = data["ablation_results"]
        histories = data["histories"]
    else:
        print("ablation_results.json not found, using default recorded empirical metrics.")
        results = [
            {'name': 'Baseline CNN', 'params': 227884, 'tr_acc': 93.30, 'va_acc': 91.96, 'te_acc': 92.47, 'te_f1': 0.8882, 'best_va_epoch': 9, 'best_va_acc': 91.96},
            {'name': 'CNN + Dropout', 'params': 227884, 'tr_acc': 86.03, 'va_acc': 86.05, 'te_acc': 86.73, 'te_f1': 0.7969, 'best_va_epoch': 10, 'best_va_acc': 86.05},
            {'name': 'CNN + Augmentation', 'params': 227884, 'tr_acc': 82.22, 'va_acc': 82.76, 'te_acc': 83.62, 'te_f1': 0.7618, 'best_va_epoch': 10, 'best_va_acc': 82.76},
            {'name': 'CNN + Aug + Drop', 'params': 227884, 'tr_acc': 74.96, 'va_acc': 78.21, 'te_acc': 79.06, 'te_f1': 0.6869, 'best_va_epoch': 8, 'best_va_acc': 78.21},
            {'name': 'Deeper CNN (Final)', 'params': 981932, 'tr_acc': 85.53, 'va_acc': 86.05, 'te_acc': 86.85, 'te_f1': 0.7997, 'best_va_epoch': 10, 'best_va_acc': 86.05},
        ]
        histories = {
            'Baseline CNN': {
                'train_acc': [61.35, 78.36, 81.70, 84.85, 87.21, 88.68, 90.50, 91.06, 92.26, 92.77],
                'val_acc':   [74.92, 74.06, 84.02, 84.98, 88.17, 88.07, 89.99, 90.75, 91.96, 91.76],
                'train_loss': [1.3066, 0.8426, 0.7433, 0.6812, 0.6398, 0.6023, 0.5725, 0.5573, 0.5345, 0.5281],
                'val_loss':   [0.9316, 0.9318, 0.7240, 0.6679, 0.6433, 0.6233, 0.5803, 0.5829, 0.5570, 0.5477]
            },
            'Deeper CNN (Final)': {
                'train_acc': [43.77, 60.52, 66.35, 69.22, 71.76, 73.70, 75.44, 77.24, 77.25, 77.62],
                'val_acc':   [66.94, 70.68, 77.15, 79.83, 80.69, 81.80, 84.18, 84.98, 85.74, 86.05],
                'train_loss': [1.6401, 1.2158, 1.0889, 1.0297, 0.9780, 0.9294, 0.8865, 0.8642, 0.8463, 0.8426],
                'val_loss':   [1.0729, 0.9591, 0.8593, 0.7933, 0.7791, 0.7379, 0.6909, 0.6960, 0.6706, 0.6631]
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  PLOT 1: Baseline CNN Accuracy Curve
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
    base_h = histories.get('Baseline CNN')
    if base_h:
        ax.plot(EPOCHS, base_h['train_acc'], color="#4FC3F7", lw=2.8, marker="o", ms=6, label="Train Accuracy")
        ax.plot(EPOCHS, base_h['val_acc'], color="#FFB74D", lw=2.8, marker="s", ms=6, label="Val Accuracy")
        ax.fill_between(EPOCHS, base_h['train_acc'], base_h['val_acc'], alpha=0.15, color="#7C4DFF")
        for i, (ta, va) in enumerate(zip(base_h['train_acc'], base_h['val_acc'])):
            ax.annotate(f"{ta:.1f}%", xy=(i+1, ta), xytext=(0, 9), textcoords="offset points", ha="center", color="#4FC3F7", fontsize=8, fontweight="bold")
            ax.annotate(f"{va:.1f}%", xy=(i+1, va), xytext=(0, -15), textcoords="offset points", ha="center", color="#FFB74D", fontsize=8, fontweight="bold")
        be = int(np.argmax(base_h['val_acc'])) + 1
        bv = max(base_h['val_acc'])
        ax.axvline(be, color="#FF6B6B", lw=1.8, ls="--", alpha=0.9, label=f"Best Val Ep{be} ({bv:.2f}%)")
        ax.scatter([be], [bv], color="#FF6B6B", s=110, zorder=5)

    ax.set_xlabel("Epoch", color="white", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=12)
    ax.set_title("Training vs Validation Accuracy — Baseline CNN (12-Class Produce)", color="white", fontsize=14, pad=12)
    ax.set_xticks(EPOCHS)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333654")
    ax.grid(True, ls="--", alpha=0.25, color="#555")
    ax.legend(facecolor=DA, edgecolor="#4FC3F7", labelcolor="white", fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "baseline_accuracy.png"), dpi=150, bbox_inches="tight", facecolor=DK)
    plt.close()
    print("Saved: outputs/baseline_accuracy.png")

    # ─────────────────────────────────────────────────────────────────────────
    #  PLOT 2: Baseline CNN Loss Curve
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
    if base_h:
        ax.plot(EPOCHS, base_h['train_loss'], color="#4FC3F7", lw=2.8, marker="o", ms=6, label="Train Loss")
        ax.plot(EPOCHS, base_h['val_loss'], color="#FFB74D", lw=2.8, marker="s", ms=6, label="Val Loss")
        ax.fill_between(EPOCHS, base_h['train_loss'], base_h['val_loss'], alpha=0.15, color="#F06292")
        for i, (tl, vl) in enumerate(zip(base_h['train_loss'], base_h['val_loss'])):
            ax.annotate(f"{tl:.3f}", xy=(i+1, tl), xytext=(0, -15), textcoords="offset points", ha="center", color="#4FC3F7", fontsize=8, fontweight="bold")
            ax.annotate(f"{vl:.3f}", xy=(i+1, vl), xytext=(0, 9), textcoords="offset points", ha="center", color="#FFB74D", fontsize=8, fontweight="bold")
        bl = int(np.argmin(base_h['val_loss'])) + 1
        bvl = min(base_h['val_loss'])
        ax.axvline(bl, color="#FF6B6B", lw=1.8, ls="--", alpha=0.9, label=f"Best Val Ep{bl} ({bvl:.4f})")
        ax.scatter([bl], [bvl], color="#FF6B6B", s=110, zorder=5)

    ax.set_xlabel("Epoch", color="white", fontsize=12)
    ax.set_ylabel("Loss", color="white", fontsize=12)
    ax.set_title("Training vs Validation Loss — Baseline CNN (12-Class Produce)", color="white", fontsize=14, pad=12)
    ax.set_xticks(EPOCHS)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333654")
    ax.grid(True, ls="--", alpha=0.25, color="#555")
    ax.legend(facecolor=DA, edgecolor="#FFB74D", labelcolor="white", fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "baseline_loss.png"), dpi=150, bbox_inches="tight", facecolor=DK)
    plt.close()
    print("Saved: outputs/baseline_loss.png")

    # ─────────────────────────────────────────────────────────────────────────
    #  PLOT 3: Ablation Study Curves (Val Acc & Val Loss across 5 configs)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DK)
    for ax in axes:
        ax.set_facecolor(DA)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333654")
        ax.grid(True, ls="--", alpha=0.25, color="#555")
        ax.set_xticks(EPOCHS)

    cfg_names = [r["name"] for r in results]
    for i, name in enumerate(cfg_names):
        h = histories.get(name)
        if h:
            color = PALETTE[i % len(PALETTE)]
            axes[0].plot(EPOCHS, h["val_acc"], color=color, lw=2.2, marker="o", ms=4, label=f"{name} ({max(h['val_acc']):.1f}%)")
            axes[1].plot(EPOCHS, h["val_loss"], color=color, lw=2.2, marker="s", ms=4, label=f"{name} ({min(h['val_loss']):.3f})")

    axes[0].set_title("Ablation Study — Validation Accuracy Across Epochs", color="white", fontsize=13, pad=10)
    axes[0].set_xlabel("Epoch", color="white", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", color="white", fontsize=11)
    axes[0].legend(facecolor=DA, edgecolor="#555", labelcolor="white", fontsize=8.5, loc="lower right")

    axes[1].set_title("Ablation Study — Validation Loss Across Epochs", color="white", fontsize=13, pad=10)
    axes[1].set_xlabel("Epoch", color="white", fontsize=11)
    axes[1].set_ylabel("Cross-Entropy Loss", color="white", fontsize=11)
    axes[1].legend(facecolor=DA, edgecolor="#555", labelcolor="white", fontsize=8.5, loc="upper right")

    fig.suptitle("Ablation Study — 5 Model Configurations Compared", color="white", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_study.png"), dpi=150, bbox_inches="tight", facecolor=DK)
    plt.close()
    print("Saved: outputs/ablation_study.png")

    # ─────────────────────────────────────────────────────────────────────────
    #  PLOT 4: Ablation Bar Chart (Test Acc & Macro F1)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DK); ax.set_facecolor(DA)
    names = [r["name"] for r in results]
    tes   = [r["te_acc"] for r in results]
    f1s   = [r["te_f1"] * 100 for r in results]
    x = np.arange(len(names)); w = 0.35
    b1 = ax.bar(x - w/2, tes, w, color="#4FC3F7", alpha=0.9, label="Test Accuracy (%)")
    b2 = ax.bar(x + w/2, f1s, w, color="#FFB74D", alpha=0.9, label="Macro F1 × 100")

    for b in b1:
        h = b.get_height()
        ax.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontsize=8.5, fontweight="bold")
    for b in b2:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white", fontsize=9.5, rotation=15, ha="right")
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("Ablation Study — Test Accuracy & Macro F1-Score Comparison", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333654")
    ax.grid(axis="y", ls="--", alpha=0.25, color="#555")
    ax.set_ylim(min(min(tes), min(f1s)) - 6, 102)
    ax.legend(facecolor=DA, edgecolor="#555", labelcolor="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_bar.png"), dpi=150, bbox_inches="tight", facecolor=DK)
    plt.close()
    print("Saved: outputs/ablation_bar.png")

    print("\nAll plots regenerated successfully in:", OUT)

if __name__ == "__main__":
    main()
