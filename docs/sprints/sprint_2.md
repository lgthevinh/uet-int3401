# Sprint 2

**Start:** 2026-10-25
**End:** 2026-10-31

## Goal

Implement code cho Q1–Q6 dựa trên kiến thức đã tìm hiểu ở sprint 1, chạy autograder để chấm điểm, và hoàn thiện report.

**Cần làm:**
- Mỗi người implement các question mình phụ trách trong `models.py`, `losses.py`, `train.py`, `gpt_model.py`.
- Chạy autograder cho từng question: `make grade Q=qN` (hoặc `python autograder.py -q qN`).
- Ghi kết quả (điểm, accuracy/loss) vào file `.tex` của question nếu cần.

**Điều kiện output:**
- Code có comment giải thích từng bước: mỗi hàm có docstring nói hàm dùng để làm gì; bên trong comment ngắn cho từng khối (forward, tính loss, backward, update weight...).
- Pass autograder của question đó (đạt full điểm).
- Q1, Q2 đã có code sẵn trên nhánh `vinh/master`: Khang tự implement lại theo hiểu biết từ sprint 1, sau đó so sánh với code có sẵn.

## Backlog

| Task | Nội dung | Output | Owner |
| ---- | -------- | ------ | ----- |
| Q1 Perceptron | `PerceptronModel`: `run`, `get_prediction`, `train` (lặp đến khi không còn misclassify) | Code + comment, pass `q1` (5đ) | Khang |
| Q2 Regression | `RegressionModel` + `regression_loss` + `train_regression`, loss < 0.02 | Code + comment, pass `q2` (5đ) | Khang |
| Q3 MNIST digit classification | Đã implement. Review lại comment theo điều kiện output, chạy lại autograder | Code + comment, pass `q3` (5đ) | Vinh |
| Q5 Convolution | `Convolve` + `DigitConvolutionalModel` + `digitconvolution_Loss` + `Train_DigitConvolution` | Code + comment, pass `q5` (3đ) | Vinh |
| Q4 Language identification (RNN) | `LanguageIDModel` + `languageid_loss` + `train_languageid` | Code + comment, pass `q4` (5đ) | Đức |
| Q6 Causal self-attention | `Attention` trong `models.py`, `Transformer_Block` + `Character_GPT` trong `gpt_model.py`, chạy thử `chargpt.py` | Code + comment, pass `q6` (2đ) | Đức |
| Ghi kết quả | Điểm autograder, accuracy/loss cuối, số epoch, hyperparameter đã chọn | `.tex` của question (nếu cần) | Owner từng Q |
| Report: review & chỉnh sửa | Review các file `qN_*.tex` của các bạn, sửa nội dung, thống nhất format, gộp vào `final_report.tex` | `final_report.tex` | Vinh |
| Chấm điểm tổng | Chạy `python autograder.py` cho cả 6 question trên nhánh đã merge | Tổng điểm /25 | Vinh |

## Notes

**Thành viên:**

| MSSV | Họ tên | Vai trò |
| ---- | ------ | ------- |
| 23020715 | Lương Thế Vinh | Nhóm trưởng |
| 24020169 | Nguyễn Trọng Khang | Thành viên |
| 24022628 | Nguyễn Minh Đức | Thành viên |
