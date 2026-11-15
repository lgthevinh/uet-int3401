"""
Draws one MNIST training image per digit (0-9) with its label, for the Q3 section of the report.

Run from the repo root:  python scripts/figures/q3_samples.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "report" / "figures" / "q3_samples.pdf"


def main():
    with np.load(ROOT / "data" / "mnist.npz") as data:
        images, labels = data["train_images"], data["train_labels"]

    # IEEE column width is ~3.5in; one row of ten 28x28 images
    fig, axes = plt.subplots(1, 10, figsize=(3.5, 0.6))
    for digit, ax in enumerate(axes):
        # first training image of each digit, so every class appears once
        i = int(np.argmax(labels == digit))
        ax.imshow(images[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(str(digit), fontsize=8, pad=2)
        ax.axis("off")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, wspace=0.1)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print("saved", OUT)


if __name__ == "__main__":
    main()
