"""
Q3 demo: cross-entropy vs MSE on MNIST, trained live side by side.

Three copies of DigitClassificationModel start from the same weights and see the
same batches; only the loss and learning rate differ. The window shows the race
to 97% validation accuracy and the gradient size behind it, to show that MSE is
slow here mostly because its gradient is smaller, not because the loss fails.

Run from the repo root:  python scripts/demo/q3_loss_race.py
"""
import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn.functional import one_hot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models import DigitClassificationModel
from losses import digitclassifier_loss, regression_loss

BATCH_SIZE = 5  # same as train_digitclassifier
EVAL_EVERY = 500  # steps between plot updates
TARGET = 0.97
MAX_EPOCHS = 15

RUNS = [
    {"name": "CE, η=0.01", "loss": digitclassifier_loss, "lr": 0.01, "color": "tab:blue"},
    {"name": "MSE, η=0.01", "loss": regression_loss, "lr": 0.01, "color": "tab:red"},
    {"name": "MSE, η=0.05", "loss": regression_loss, "lr": 0.05, "color": "tab:orange"},
]


def load_mnist():
    """
    Loads the MNIST train set and the dev split that the Q3 training loop validates on.

    Reads the npz directly so batches are plain tensor slices, which is faster than
    backend's per-item Dataset when every step trains three models.
    """
    with np.load(ROOT / "data" / "mnist.npz") as data:
        x = torch.tensor(data["train_images"])
        y = one_hot(torch.tensor(data["train_labels"]).long(), 10).float()
        # backend.py uses the even-indexed test images as the dev set
        dev_x = torch.tensor(data["test_images"][0::2])
        dev_y = torch.tensor(data["test_labels"][0::2]).long()
    return x, y, dev_x, dev_y


def grad_norm(model):
    """Total L2 norm of all parameter gradients, i.e. the size of the SGD direction."""
    return torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()


def setup_plot():
    """Creates the three stacked panels with one line per run in each."""
    plt.rcParams["font.size"] = 13
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    ax_acc, ax_grad, ax_step = axes

    ax_acc.set_title("Validation accuracy")
    ax_acc.axhline(TARGET, color="gray", linestyle="--")
    ax_acc.set_ylim(0.8, 1.0)
    ax_grad.set_title(r"Gradient size $\|\nabla W\|$")
    ax_step.set_title(r"Step size $\eta \cdot \|\nabla W\|$")
    ax_step.set_xlabel("epoch")

    for run in RUNS:
        run["lines"] = [ax.plot([], [], color=run["color"], label=run["name"])[0] for ax in axes]
    ax_acc.legend(loc="lower right")
    fig.tight_layout()
    plt.show(block=False)
    return axes


def update_plot(epoch, dev_x, dev_y, axes):
    """
    Evaluates every run on the dev set and redraws the panels.

    Gradient norms are averaged over the steps since the last update, because at
    batch size 5 single-step norms are too noisy to read.
    """
    ax_acc = axes[0]
    for run in RUNS:
        with torch.no_grad():
            acc = (run["model"](dev_x).argmax(dim=1) == dev_y).float().mean().item()
        run["xs"].append(epoch)
        run["acc"].append(acc)
        run["grad"].append(np.mean(run["step_grads"]))
        run["step_grads"].clear()

        step_size = [run["lr"] * g for g in run["grad"]]
        for line, ys in zip(run["lines"], (run["acc"], run["grad"], step_size)):
            line.set_data(run["xs"], ys)

        # mark where the run first reaches the target
        if acc >= TARGET and run["crossed"] is None:
            run["crossed"] = epoch
            ax_acc.plot(epoch, acc, "o", color=run["color"], markersize=10)
            run["lines"][0].set_label(f"{run['name']}: {TARGET:.0%} at epoch {epoch:.1f}")
            ax_acc.legend(loc="lower right")

    # accuracy keeps its fixed y-range (set_ylim turned its y-autoscale off)
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    plt.pause(0.001)


def main():
    # batches of 5 are too small to split across threads; one thread is ~2.4x faster here
    torch.set_num_threads(1)
    x, y, dev_x, dev_y = load_mnist()
    steps_per_epoch = len(x) // BATCH_SIZE

    # identical starting weights, so only the loss and learning rate differ
    torch.manual_seed(0)
    base = DigitClassificationModel()
    for run in RUNS:
        run["model"] = copy.deepcopy(base)
        run["opt"] = optim.SGD(run["model"].parameters(), run["lr"])
        run["step_grads"] = []
        run["xs"], run["acc"], run["grad"] = [], [], []
        run["crossed"] = None

    axes = setup_plot()
    step = 0
    for epoch in range(MAX_EPOCHS):
        start = time.time()
        # same batch order as train_digitclassifier (no shuffle)
        for i in range(0, len(x), BATCH_SIZE):
            xb, yb = x[i:i + BATCH_SIZE], y[i:i + BATCH_SIZE]
            for run in RUNS:
                loss = run["loss"](run["model"](xb), yb)
                run["opt"].zero_grad()
                loss.backward()
                run["step_grads"].append(grad_norm(run["model"]))
                run["opt"].step()

            step += 1
            if step % EVAL_EVERY == 0:
                update_plot(step / steps_per_epoch, dev_x, dev_y, axes)

        accs = ", ".join(f"{run['name']} {run['acc'][-1]:.4f}" for run in RUNS)
        print(f"epoch {epoch + 1} ({time.time() - start:.1f}s): {accs}")

        # MSE at η=0.01 needs ~33 epochs, so stop once the other two have reached the target
        if RUNS[0]["crossed"] and RUNS[2]["crossed"]:
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
