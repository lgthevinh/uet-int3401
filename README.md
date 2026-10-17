# uet-int3401

INT3401_11 group project: [UC Berkeley CS188 Project 5: Machine Learning](https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/) (PyTorch version).

We build and train neural networks with PyTorch: a perceptron, a regression network, an MNIST digit classifier, an RNN for language identification, a hand-written convolution, and causal self-attention.

## Project structure

```
models.py        # model definitions for Q1–Q6 (our code)
train.py         # training loops (our code)
losses.py        # loss functions (our code)
gpt_model.py     # character-level GPT built on the Q6 Attention
chargpt.py       # trains and samples the GPT (needs an input.txt text file)
autograder.py    # local grader, do not edit
backend.py       # datasets and plotting used by the grader, do not edit
data/            # mnist.npz, lang_id.npz
docs/sumary.md   # question summary, required knowledge, submission checklist
docs/report/     # LaTeX report (final_report.tex)
```

## Questions

| Q | Task | Where |
|---|------|-------|
| Q1 | Perceptron | `PerceptronModel` in `models.py`, `train_perceptron` in `train.py` |
| Q2 | Regression on sin(x) | `RegressionModel`, `regression_loss`, `train_regression` |
| Q3 | MNIST digit classification | `DigitClassificationModel`, `digitclassifier_loss`, `train_digitclassifier` |
| Q4 | Language identification (RNN) | `LanguageIDModel`, `languageid_loss`, `train_languageid` |
| Q5 | Convolution | `Convolve`, `DigitConvolutionalModel`, `digitconvolution_Loss`, `Train_DigitConvolution` |
| Q6 | Causal self-attention | `Attention` in `models.py` |

Pass conditions and required knowledge per question: [docs/sumary.md](docs/sumary.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

## Run

```bash
python autograder.py                  # grade all questions
python autograder.py -q q3            # grade one question
python autograder.py --no-graphics    # skip the matplotlib plots (faster, works over SSH)
python chargpt.py                     # train the character GPT (place input.txt in the repo root first)
```
