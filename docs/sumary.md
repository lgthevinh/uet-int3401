### Questions

| Q | Pts | Task | Code to fill in | Pass condition |
|---|-----|------|-----------------|----------------|
| Q1 | 5 | Perceptron binary classifier (+1 / -1) | `models.py`: `PerceptronModel`<br>`train.py`: `train_perceptron` | 100% training accuracy |
| Q2 | 5 | Regression network approximating sin(x) on [-2π, 2π] | `models.py`: `RegressionModel`<br>`losses.py`: `regression_loss`<br>`train.py`: `train_regression` | Final loss ≤ 0.02 |
| Q3 | 5 | MNIST handwritten digit classification (784 → 10) | `models.py`: `DigitClassificationModel`<br>`losses.py`: `digitclassifier_loss`<br>`train.py`: `train_digitclassifier` | Test accuracy ≥ 97% |
| Q4 | 5 | Language identification of a single word with an RNN (5 languages, 47 characters) | `models.py`: `LanguageIDModel`<br>`losses.py`: `languageid_loss`<br>`train.py`: `train_languageid` | Validation accuracy ≥ 81% |
| Q5 | 3 | Hand-written 2D convolution, then a digit classifier on top of it | `models.py`: `Convolve`, `DigitConvolutionalModel`<br>`losses.py`: `digitconvolution_Loss`<br>`train.py`: `Train_DigitConvolution` | Test accuracy ≥ 80% |
| Q6 | 2 | Causal masked self-attention | `models.py`: `Attention` | Output matches the autograder's expected attention output |

Run all with `python autograder.py`, or one question with `python autograder.py -q q1`. Total: 25 points.

### Knowledge required

| Q | Concepts | PyTorch tools |
|---|----------|---------------|
| Q1 | Linear classifier, dot product score, sign decision rule, perceptron update rule (`w += y * x` on a mistake), convergence on linearly separable data | `Parameter`, `ones`, `tensordot`, `DataLoader`, `no_grad` |
| Q2 | Feed-forward network, hidden layers, ReLU non-linearity, mean squared error, gradient descent, batching, learning rate and layer size tuning | `Linear`, `relu`, `mse_loss`, `optim` (e.g. Adam), `loss.backward()`, `optimizer.step()`, `zero_grad()` |
| Q3 | Multi-class classification, logits, softmax and cross-entropy, one-hot labels, epochs, train vs test accuracy, overfitting | `Linear`, `relu`, `cross_entropy`, `optim`, `DataLoader` |
| Q4 | Recurrent neural network, hidden state carried across characters, shared weights over time steps, variable-length input, one-hot character encoding | `Linear`, `relu`, `cross_entropy`, `movedim` (batch × length → length × batch) |
| Q5 | 2D convolution as a sliding window, kernel/filter, output size = (H − kH + 1) × (W − kW + 1), feature maps feeding a dense classifier | `tensordot`, `zeros`, tensor slicing `t[y:y+h, x:x+w]`, `Linear`, `cross_entropy` |
| Q6 | Self-attention: query, key and value matrices, scaled dot product `QKᵀ / √d`, causal (lower-triangular) mask, softmax over the last dimension | `Linear`, `matmul`, `movedim`, `masked_fill`, `softmax(dim=-1)` |

### Output

Final submission for INT3401_11. `X` is the group number.

| Deliverable | Requirement |
|-------------|-------------|
| Zip file | One file per group, named `INT3401_11_NhomX.zip`, submitted by the group leader only |
| Zip contents | Report PDF, all source code, any extra documents. No data (exclude `data/`) |
| Report | Written in LaTeX (Overleaf or equivalent) from the provided template, editing `IEEE-conference-template-062824.tex` directly. In Vietnamese, at most 7 pages |
| GitHub repository | All source code plus a README describing the project and how to re-run it. Repository link goes in the report's abstract (Tóm tắt) |
| CoReach | Group leader creates a Reach named `2627I_INT3401_11_Nhóm X` on coreach.ai, adds all members, and adds `duyld@vnu.edu.vn` as Supervisor. Discussion, documents and tasks are tracked there |

Report must contain at least:

1. Work assignment table (bảng phân công công việc)
2. Overview and problem description, making clear what the project solves
3. Method
4. Experiments and results

Groups that propose their own problem must also cite theory and methods in IEEE format, with at least 7 references.

Grading is based on: CoReach activity, the report, reusable source code, and a live Q&A (schedule announced later).
