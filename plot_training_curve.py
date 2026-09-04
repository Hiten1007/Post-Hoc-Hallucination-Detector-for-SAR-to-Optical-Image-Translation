"""
Parses any model's training log and plots the convergence curve.
Saves output to plots/<model>/ to keep all model plots organized.

Usage:
    python3 plot_training_curve.py --model pix2pix
    python3 plot_training_curve.py --model cyclegan
    python3 plot_training_curve.py --model palette
"""

import re
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Config per model ---
MODEL_CONFIG = {
    "pix2pix": {
        "log_file": "pix2pix_training.log",
        "loss_label": "Validation L1 Pixel Loss",
        "log_pattern": r'Epoch (\d+) Complete\. Validation L1 Pixel Loss: ([\d.]+)',
        "color": "#2196F3",
        "title": "Pix2Pix SAR→Optical Translation: Training Convergence Curve",
    },
    "cyclegan": {
        "log_file": "cyclegan_training.log",
        "loss_label": "Validation Cycle Loss",
        "log_pattern": r'Epoch (\d+) Complete\. Validation Cycle Loss: ([\d.]+)',
        "color": "#FF9800",
        "title": "CycleGAN SAR→Optical Translation: Training Convergence Curve",
    },
    "palette": {
        "log_file": "palette_training.log",
        "loss_label": "Validation Noise MSE Loss",
        "log_pattern": r'Epoch (\d+) Complete\. Validation Noise Loss: ([\d.]+)',
        "color": "#4CAF50",
        "title": "Palette (DDPM) SAR→Optical Translation: Training Convergence Curve",
    },
}


def parse_log(log_file, pattern):
    epochs, losses = [], []
    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found: {log_file}")
        return epochs, losses

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return epochs, losses


def main():
    parser = argparse.ArgumentParser(description="Plot training convergence curve for any model.")
    parser.add_argument("--model", type=str, required=True, choices=["pix2pix", "cyclegan", "palette"],
                        help="Which model's training log to plot.")
    args = parser.parse_args()

    cfg = MODEL_CONFIG[args.model]

    # Output directory: plots/<model>/
    out_dir = os.path.join("plots", args.model)
    os.makedirs(out_dir, exist_ok=True)
    output_png = os.path.join(out_dir, f"{args.model}_training_curve.png")

    epochs, losses = parse_log(cfg["log_file"], cfg["log_pattern"])

    if not epochs:
        print(f"No epoch data found in {cfg['log_file']}. Has training run yet?")
        return

    best_idx = int(np.argmin(losses))
    best_epoch = epochs[best_idx]
    best_loss = losses[best_idx]

    print(f"\n{'='*55}")
    print(f"  {args.model.upper()} Training Summary")
    print(f"{'='*55}")
    print(f"  Total epochs logged : {len(epochs)}")
    print(f"  Starting loss       : {losses[0]:.4f}")
    print(f"  Best epoch          : {best_epoch}  (loss = {best_loss:.4f})")
    print(f"  Final epoch loss    : {losses[-1]:.4f}")
    print(f"  Total improvement   : {(1 - best_loss/losses[0])*100:.1f}% reduction")
    print(f"{'='*55}")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Main loss curve
    ax.plot(epochs, losses, color=cfg["color"], linewidth=2.5,
            marker='o', markersize=5, label=cfg["loss_label"], zorder=3)
    ax.fill_between(epochs, losses, alpha=0.1, color=cfg["color"])

    # Best epoch marker
    ax.axvline(x=best_epoch, color='#4CAF50', linestyle='--', linewidth=2,
               label=f'Best Epoch: {best_epoch}  (loss={best_loss:.4f})')
    ax.scatter([best_epoch], [best_loss], color='#4CAF50', s=150, zorder=5)
    ax.annotate(f'  Best: Epoch {best_epoch}\n  loss={best_loss:.4f}',
                xy=(best_epoch, best_loss), fontsize=10, color='#228B22', fontweight='bold')

    # Theoretical perfection line
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.6,
               label='Theoretical Perfection (loss=0)')

    # Gap to perfection annotation
    ax.annotate('', xy=(epochs[-1] + 0.3, 0),
                xytext=(epochs[-1] + 0.3, best_loss),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(epochs[-1] + 0.8, best_loss / 2,
            f'Gap to\nperfection:\n{best_loss:.4f}',
            color='red', fontsize=9, va='center')

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel(cfg["loss_label"], fontsize=13, fontweight='bold')
    ax.set_title(f'{cfg["title"]}\n(Lower = Better | loss=0 = Theoretical Perfect Reconstruction)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 3)
    ax.set_ylim(-best_loss * 0.05, max(losses) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(epochs[::2] if len(epochs) > 10 else epochs)

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches='tight')
    print(f"\n  Graph saved to: {output_png}")
    print(f"  Download with:")
    print(f"  scp ubuntu@<your-ip>:/data/home/ubuntu/Post-Hoc-Hallucination-Detector-for-SAR-to-Optical-Image-Translation/{output_png} .")


if __name__ == "__main__":
    main()
