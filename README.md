# uet-int3401

Bài tập lớn nhóm môn Trí tuệ nhân tạo INT3401_11: [UC Berkeley CS188 Project 5: Machine Learning](https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/) (phiên bản PyTorch).

Nhóm cài đặt và huấn luyện các mạng nơ-ron bằng PyTorch: perceptron, mạng hồi quy, bộ phân loại chữ số MNIST, RNN nhận diện ngôn ngữ, phép tích chập tự cài đặt và self-attention có mặt nạ nhân quả.

## Cấu trúc dự án

```
models.py        # định nghĩa mô hình Q1–Q6 (code của nhóm)
train.py         # vòng huấn luyện (code của nhóm)
losses.py        # hàm mất mát (code của nhóm)
gpt_model.py     # GPT mức ký tự dựng trên Attention của Q6
chargpt.py       # huấn luyện và sinh văn bản với GPT (cần file input.txt)
autograder.py    # bộ chấm cục bộ, không sửa
backend.py       # dữ liệu và vẽ đồ thị cho bộ chấm, không sửa
data/            # mnist.npz, lang_id.npz
docs/sumary.md   # tóm tắt các bài, kiến thức cần có, yêu cầu nộp bài
docs/report/     # báo cáo LaTeX (final_report.tex)
```

## Các bài

| Bài | Nhiệm vụ | Vị trí |
|-----|----------|--------|
| Q1 | Perceptron | `PerceptronModel` trong `models.py`, `train_perceptron` trong `train.py` |
| Q2 | Hồi quy hàm sin(x) | `RegressionModel`, `regression_loss`, `train_regression` |
| Q3 | Phân loại chữ số MNIST | `DigitClassificationModel`, `digitclassifier_loss`, `train_digitclassifier` |
| Q4 | Nhận diện ngôn ngữ bằng RNN | `LanguageIDModel`, `languageid_loss`, `train_languageid` |
| Q5 | Tích chập | `Convolve`, `DigitConvolutionalModel`, `digitconvolution_Loss`, `Train_DigitConvolution` |
| Q6 | Self-attention có mặt nạ nhân quả | `Attention` trong `models.py` |

Điều kiện đạt và kiến thức cần có của từng bài: [docs/sumary.md](docs/sumary.md).

## Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

## Chạy

```bash
python autograder.py                  # chấm tất cả các bài
python autograder.py -q q3            # chấm một bài
python autograder.py --no-graphics    # bỏ đồ thị matplotlib (nhanh hơn, chạy được qua SSH)
python chargpt.py                     # huấn luyện GPT mức ký tự (đặt input.txt ở thư mục gốc trước)
```
