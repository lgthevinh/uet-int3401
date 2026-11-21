"""
Trains the Q5 convolutional model, then draws one input image, the learned 3x3
kernel and the resulting feature map, for the Q5 section of the report.

Run from the repo root:  python scripts/figures/q5_kernel.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import backend
import train
from models import Convolve, DigitConvolutionalModel

OUT = ROOT / "docs" / "report" / "figures" / "q5_kernel.pdf"
# patch position (row, col) highlighted on the input and its output pixel
PATCH = (5, 12)


def main():
    torch.manual_seed(0)
    model = DigitConvolutionalModel()
    dataset = backend.DigitClassificationDataset2(model)
    train.Train_DigitConvolution(model, dataset)

    image = torch.tensor(dataset.x[0]).reshape(28, 28).float()
    kernel = model.convolution_weights.detach()
    with torch.no_grad():
        feature = Convolve(image, kernel)

    fig, (ax_in, ax_k, ax_out) = plt.subplots(
        1, 3, figsize=(3.5, 1.4), gridspec_kw={"width_ratios": [28, 14, 26]})

    ax_in.imshow(image, cmap="gray_r")
    ax_in.set_title(r"Ảnh $X$ ($28 \times 28$)", fontsize=7)
    r, c = PATCH
    ax_in.add_patch(Rectangle((c - 0.5, r - 0.5), 3, 3, fill=False, color="tab:red", lw=1))

    # kernel drawn with its values so the reader can see the learned weights
    ax_k.imshow(kernel, cmap="Blues", vmin=0, vmax=3)
    for (i, j), v in np.ndenumerate(kernel.numpy()):
        ax_k.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5)
    ax_k.set_title(r"Kernel $K$ ($3 \times 3$)", fontsize=7)

    ax_out.imshow(feature, cmap="gray_r")
    ax_out.set_title(r"Feature map $Y$ ($26 \times 26$)", fontsize=7)
    ax_out.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, color="tab:red", lw=1))

    for ax in (ax_in, ax_k, ax_out):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print("kernel:\n", kernel.numpy())
    print("saved", OUT)


if __name__ == "__main__":
    main()
