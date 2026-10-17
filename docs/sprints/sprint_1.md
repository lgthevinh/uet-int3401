# Sprint 1

**Start:** 2026-10-17
**End:** 2026-10-24

## Goal

Hiểu rõ bài toán của project 5 và bắt đầu report.

**Cần làm:**
- Với mỗi question Q1–Q6, tìm hiểu và trả lời:
    - Dùng kiến thức gì?
    - Kiến thức đó giải quyết vấn đề gì, ứng dụng thực tế là gì?
    - Implement thế nào, công thức gì? (optional)
- Thống nhất output của bài toán để viết phần mô tả bài toán.

**Output cuối sprint:**
- Mỗi question có một file riêng `docs/report/qN_<tên>.tex` theo format trên, khoảng 1/2 trang.
- `final_report.tex` có bản nháp: thông tin sinh viên, giới thiệu tổng quan (các vấn đề và các neural network trong project 5), mô tả bài toán.

## Backlog

| Task | Nội dung | Output | Owner |
| ---- | -------- | ------ | ----- |
| Q1 Perceptron | Linear classifier, dot product score, sign decision rule, update rule `w += y * x`, hội tụ trên dữ liệu linearly separable | `q1_perceptron.tex` | Vinh |
| Q2 Regression | Feed-forward network, hidden layer, ReLU, MSE loss, gradient descent, batching, learning rate | `q2_regression.tex` | Vinh |
| Q3 MNIST digit classification | Multi-class classification, logits, softmax + cross-entropy, epoch, train vs test accuracy, overfitting | `q3_digit_classification.tex` | Khang |
| Q4 Language identification (RNN) | RNN, hidden state qua từng ký tự, shared weights theo thời gian, input độ dài thay đổi, one-hot encoding ký tự | `q4_language_id.tex` | Khang |
| Q5 Convolution | 2D convolution (sliding window), kernel/filter, kích thước output `(H − kH + 1) × (W − kW + 1)`, feature map → dense classifier | `q5_convolution.tex` | Đức |
| Q6 Causal self-attention | Query/key/value, scaled dot product `QKᵀ / √d`, causal mask (lower-triangular), softmax | `q6_attention.tex` | Đức |
| Report: thông tin sinh viên | Tên nhóm, họ tên, MSSV, email các thành viên | `final_report.tex` (phần author) | Vinh |
| Report: giới thiệu tổng quan | Project 5 giải quyết vấn đề gì, tóm tắt các neural network (perceptron → feed-forward → RNN → CNN → attention) | `final_report.tex` (Giới thiệu) | Vinh |
| Họp nhóm: chốt output của bài toán | Thống nhất input/output, metric đánh giá (accuracy, loss) của từng question | Ghi kết quả vào Notes | Cả nhóm |
| Report: mô tả bài toán | Viết sau khi họp: input, output, điều kiện đạt của từng question | `final_report.tex` (Mô tả bài toán) | Vinh |

## Notes

**Thành viên:**

| MSSV | Họ tên | Vai trò |
| ---- | ------ | ------- |
| 23020715 | Lương Thế Vinh | Nhóm trưởng |
| 24020169 | Nguyễn Trọng Khang | Thành viên |
| 24022628 | Nguyễn Minh Đức | Thành viên |
