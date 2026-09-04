"""
Parses the Pix2Pix training log and plots:
1. Validation L1 loss per epoch
2. Marks the best epoch (lowest L1)
3. Shows the "perfection" line (L1 = 0)

Run on server:
    python3 plot_pix2pix_training.py
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = "pix2pix_training.log"
OUTPUT_PNG = "pix2pix_training_curve.png"

def parse_log(log_file):
    epochs = []
    val_l1_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # Match lines like: "Epoch 21 Complete. Validation L1 Pixel Loss: 0.0304"
            match = re.search(r'Epoch (\d+) Complete\. Validation L1 Pixel Loss: ([\d.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                val_l1_losses.append(float(match.group(2)))

    return epochs, val_l1_losses


def main():
    epochs, val_l1 = parse_log(LOG_FILE)

    if not epochs:
        print("No epoch data found in log file!")
        return

    best_epoch = epochs[np.argmin(val_l1)]
    best_loss = min(val_l1)

    print(f"Total epochs found: {len(epochs)}")
    print(f"Best epoch: {best_epoch}  (Val L1 = {best_loss:.4f})")
    print(f"Final epoch: {epochs[-1]}  (Val L1 = {val_l1[-1]:.4f})")
    print(f"Improvement from Epoch 1 to best: {val_l1[0]:.4f} → {best_loss:.4f}  ({(1 - best_loss/val_l1[0])*100:.1f}% reduction)")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Main loss curve
    ax.plot(epochs, val_l1, color='#2196F3', linewidth=2.5, marker='o',
            markersize=5, label='Validation L1 Pixel Loss', zorder=3)

    # Shade the area under the curve
    ax.fill_between(epochs, val_l1, alpha=0.1, color='#2196F3')

    # Mark best epoch
    ax.axvline(x=best_epoch, color='#4CAF50', linestyle='--', linewidth=2,
               label=f'Best Epoch: {best_epoch} (L1={best_loss:.4f})')
    ax.scatter([best_epoch], [best_loss], color='#4CAF50', s=150, zorder=5)
    ax.annotate(f'  Best: Epoch {best_epoch}\n  L1={best_loss:.4f}',
                xy=(best_epoch, best_loss), fontsize=10, color='#4CAF50', fontweight='bold')

    # "Perfection" line at 0
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.6,
               label='Theoretical Perfection (L1=0)')

    # Gap annotation showing distance from perfection
    ax.annotate('', xy=(epochs[-1], 0), xytext=(epochs[-1], best_loss),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(epochs[-1] + 0.3, best_loss / 2, f'Gap to\nperfection:\n{best_loss:.4f}',
            color='red', fontsize=9, va='center')

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation L1 Pixel Loss', fontsize=13, fontweight='bold')
    ax.set_title('Pix2Pix SAR→Optical Translation: Training Convergence Curve\n'
                 '(Lower = Better | L1=0 = Pixel-Perfect Reconstruction)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 2)
    ax.set_ylim(-0.002, max(val_l1) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(epochs[::2])

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches='tight')
    print(f"\nGraph saved to: {OUTPUT_PNG}")
    print(f"Download it with: scp ubuntu@<your-ip>:/data/home/ubuntu/Post-Hoc-Hallucination-Detector-for-SAR-to-Optical-Image-Translation/{OUTPUT_PNG} .")


if __name__ == "__main__":
    main()
