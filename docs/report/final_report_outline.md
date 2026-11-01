# Final report: dàn ý

Bản nháp cấu trúc để review trước khi gộp vào `final_report.tex`.

**Trạng thái**: `final_report.tex` đã có phần đầu, I và II; III–V mới có khung, phần chưa viết đánh dấu `\todo{...}` (chữ đỏ trong PDF).

**Ràng buộc** (từ `docs/sumary.md`): template IEEE conference, tiếng Việt, tối đa 7 trang, link GitHub đặt trong phần Tóm tắt. Report phải có ít nhất 4 mục:

| Mục bắt buộc | Nằm ở section |
|---|---|
| Bảng phân công công việc | Bảng I (ngay sau Tóm tắt) |
| Tổng quan và mô tả bài toán | I. Giới thiệu |
| Phương pháp | II. Kiến thức nền tảng + III. Phương pháp |
| Thực nghiệm và kết quả | IV. Thực nghiệm và kết quả |

---

## 0. Phần đầu

- **Tiêu đề**: *Báo cáo bài tập lớn: Học máy với PyTorch theo CS188 Project 5 (UC Berkeley)*
- **Tác giả**: 3 thành viên + MSSV (đã có).
- **Tóm tắt** (~150 từ): dự án làm gì (CS188 Project 5, 6 bài), các mô hình đã cài đặt, kết quả chính (25/25 điểm autograder, accuracy các bài), link GitHub.
- **Từ khoá**: perceptron, mạng nơ-ron, MNIST, RNN, tích chập, self-attention, PyTorch.
- **Bảng I: Phân công công việc**

  | Thành viên | MSSV | Công việc | Đóng góp |
  |---|---|---|---|
  | Lương Thế Vinh | 23020715 | Q3, Q5, tổng hợp final report, quản lý sprint/CoReach | 36% |
  | Nguyễn Trọng Khang | 24020169 | Q1, Q2 | 32% |
  | Nguyễn Minh Đức | 24022628 | Q4, Q6 | 32% |

## I. Giới thiệu

Mục đích: người đọc hiểu dự án giải quyết bài toán gì trước khi đi vào chi tiết.

- Bối cảnh: CS188 Project 5 (bản PyTorch), mục tiêu học cách xây dựng và huấn luyện mạng nơ-ron từ mô hình tuyến tính đến attention.
- Mạch phát triển qua 6 bài: tuyến tính (Q1) → phi tuyến (Q2) → phân loại đa lớp (Q3) → chuỗi độ dài thay đổi (Q4) → khai thác cấu trúc không gian (Q5) → attention (Q6).
- **Bảng II: Tổng quan các bài toán**: Q, nhiệm vụ, điều kiện đạt, điểm. Bỏ cột đầu vào → đầu ra để bảng vừa một cột; chi tiết đầu vào/đầu ra nằm ở III. Ngưỡng và tập được chấm lấy từ `autograder.py`.
- Cấu trúc bài báo cáo (1–2 câu).

## II. Kiến thức nền tảng

Mục đích: gom các kiến thức dùng chung, để phần III không phải giải thích lại ở từng bài.

- A. Tầng tuyến tính và mạng nhiều lớp (MLP): $Wx + b$, vì sao cần hàm phi tuyến.
- B. Hàm kích hoạt: sign/step, ReLU, softmax.
- C. Hàm mất mát: MSE (hồi quy), cross-entropy kết hợp softmax (phân loại), gradient $p - y$.
- D. Tối ưu: gradient descent, mini-batch, epoch, Adam; lan truyền ngược và autograd.
  - Câu nối với Q1: quy tắc cập nhật perceptron chính là SGD trên hàm mất mát $\max(0, -y\, w \cdot x)$ với bước học cố định, nên Q1 và Q2 dùng chung một khung tối ưu, chỉ khác hàm mất mát.
- E. Đánh giá: tập train/validation/test, overfitting, early stopping.

**Nguồn**: `q0_common.md`, cùng các đoạn chung đang lặp ở Q2 (MLP, ReLU), Q3 (softmax, CE), Q4 (Adam, overfitting).

## III. Phương pháp

Mỗi bài một subsection, cùng khung: **bài toán cụ thể** (1–2 câu) → **mô hình và công thức** → **cài đặt** (kiến trúc, số tham số, các hàm trong code). Kết quả **không** đặt ở đây mà chuyển sang IV.

| Subsection | Nội dung giữ lại | Nguồn |
|---|---|---|
| A. Q1 Perceptron | dự đoán bằng dấu của $w \cdot x$, quy tắc cập nhật $w \leftarrow w + yx$, điều kiện dừng | `q1_perceptron.tex` (viết lại) |
| B. Q2 Hồi quy phi tuyến | MLP $1 \to 100 \to 1$, MSE, xấp xỉ $\sin(x)$ | `q2_regression.tex` (viết lại) |
| C. Q3 Phân loại MNIST | MLP $784 \to 128 \to 10$, logits + CE | `q3_digit_classification.tex` |
| D. Q4 Nhận diện ngôn ngữ | RNN: $h_0$, $h_t$, chia sẻ tham số, BPTT, batch theo độ dài, `movedim` | `q4_language_id.tex` |
| E. Q5 Tích chập | công thức tích chập, kích thước đầu ra, `Convolve` bằng `tensordot`, kiến trúc conv + MLP | `q5_convolution.tex` |
| F. Q6 Causal self-attention | Q/K/V, chia $\sqrt{d_k}$, mặt nạ tam giác dưới trước softmax, bảng các bước của `forward` | `q6_attention.tex` |

## IV. Thực nghiệm và kết quả

- **A. Thiết lập thực nghiệm**
  - Môi trường (Python, PyTorch, CPU/GPU), cách chấm (`autograder.py`).
  - Dữ liệu và cách chia tập của từng bài (MNIST, lang_id, tập 200 ảnh của Q5).
  - **Bảng III: Siêu tham số**: Q, optimizer, learning rate, batch size, số epoch tối đa, điều kiện dừng.
- **B. Kết quả tổng hợp**
  - **Bảng IV: Kết quả**: Q, chỉ số (accuracy/loss), ngưỡng yêu cầu, kết quả đạt được, điểm. Dòng cuối: tổng 25/25.
- **C. Phân tích**
  - C1. Q3: so sánh cross-entropy và MSE (bảng hiện có ở Q3).
  - C2. Q4: ảnh hưởng của việc xáo batch và giữ trọng số tốt nhất (bảng hiện có ở Q4).
  - C3. Q5: quá trình huấn luyện (bảng 7 epoch) và lưu ý về con số 100% test.
  - C4. Q6: kiểm tra tính đúng (không có huấn luyện), hai tính chất của ma trận attention.
  - C5. So sánh chéo (ngắn, định tính): MLP và tích chập trên MNIST (Q3/Q5), RNN và attention trên dữ liệu chuỗi (Q4/Q6).

## V. Kết luận và hướng phát triển

- Tóm tắt kết quả đạt được.
- Hạn chế: `Convolve` dùng vòng lặp Python và chỉ có 1 kernel; RNN thuần; attention một đầu; cách chia dữ liệu của Q5.
- Hướng phát triển: vector hoá tích chập, CNN nhiều kernel/pooling, LSTM/GRU, multi-head attention, character GPT (`gpt_model.py`, `chargpt.py`).

## Tài liệu tham khảo

Không bắt buộc ≥ 7 tài liệu (chỉ áp dụng cho nhóm tự đề xuất đề tài), nhưng nên có để trích các phương pháp. Đề xuất:

1. CS188 Project 5: Machine Learning (UC Berkeley, Fall 2025)
2. PyTorch documentation
3. F. Rosenblatt, perceptron (1958)
4. D. Rumelhart, G. Hinton, R. Williams, backpropagation (1986)
5. Y. LeCun et al., MNIST và CNN (1998)
6. J. Elman, mạng hồi quy (1990)
7. D. Kingma, J. Ba, Adam (2015)
8. A. Vaswani et al., *Attention Is All You Need* (2017)

---

## Ngân sách trang (7 trang)

| Phần | Trang |
|---|---|
| Phần đầu + Bảng I | 0.5 |
| I. Giới thiệu | 0.5 |
| II. Kiến thức nền tảng | 0.75 |
| III. Phương pháp | 3.0 (Q1 0.3, Q2 0.4, Q3 0.5, Q4 0.6, Q5 0.6, Q6 0.6) |
| IV. Thực nghiệm và kết quả | 1.5 |
| V. Kết luận | 0.25 |
| Tài liệu tham khảo | 0.5 |
| **Tổng** | **7.0** |

## Vấn đề cần xử lý trước khi gộp

1. **Dung lượng**: 6 file Q hiện có khoảng 7 200 từ (tính cả markup), trong khi 7 trang IEEE hai cột chứa khoảng 5 000 từ cho toàn bộ báo cáo. Cần cắt khoảng 35%. Chỗ nên cắt nhiều nhất: Q5 (1 785 từ), chi tiết broadcast của mặt nạ ở Q6, và các đoạn kiến thức lặp lại (chuyển lên II).
2. **Q1, Q2 cần viết lại**:
   - Đang là document độc lập (`\documentclass{article}` + `fontspec`, chỉ biên dịch được bằng XeLaTeX/LuaLaTeX), không tương thích với `final_report.tex` (pdfLaTeX + gói `vietnam`).
   - Có mục không hợp với báo cáo (khung Thống nhất Output), văn phong hướng dẫn ("người học cần..."). Mục Ứng dụng thực tế đã bỏ.
   - Chưa có kết quả và siêu tham số thực tế.
   - Q2 ghi lớp ẩn $H = 64$ hoặc $128$, nhưng code dùng $1 \to 100 \to 1$, batch 10, 500 epoch.
   - Q2 dùng cú pháp markdown `**ReLU**` trong LaTeX.
   - Q1: công thức cập nhật $w \leftarrow w + y \cdot x$ thiếu hệ số `learning_rate = 0.5` mà code dùng (`train.py:58`).
   - Q2: ghi optimizer là "SGD hoặc Adam" và không có learning rate, trong khi code dùng `optim.SGD` với lr = 0.01 (`train.py:94`, `train.py:109`).
3. **Q6 nhắc tới "Q7"**: autograder chỉ có q1–q6. Nên bỏ, hoặc đổi thành phần mở rộng `gpt_model.py`/`chargpt.py`.
4. **Vị trí kết quả chưa thống nhất**: Q3, Q4 đặt kết quả trong mục "Cài đặt", Q5 có mục "Kết quả" riêng. Theo dàn ý này, tất cả chuyển sang IV.
5. **Thuật ngữ chưa thống nhất**: Q4 dùng "tập kiểm định / kiểm tra", Q5 dùng "validation / test". Cần chọn một cách viết, ví dụ tiếng Việt kèm tiếng Anh trong ngoặc ở lần đầu.
6. **Q3**: bảng có số gần đúng ("~33", "~8") nên chạy lại để lấy số chính xác; ghi rõ ngưỡng 97% là điều kiện dừng trên validation, còn autograder chấm trên test.
7. **Placeholder của template**: đã thay hết trong `final_report.tex`, chỉ còn câu tóm tắt kết quả trong phần Tóm tắt (viết sau khi xong IV).
8. **Q4 chọn mô hình bằng tập test**: `LanguageIDDataset.get_validation_accuracy()` gọi `_predict()` với mặc định `split='test'` (`backend.py:437`), nên `train_languageid` đang dừng sớm và giữ trọng số tốt nhất theo độ chính xác trên **tập test**, đúng tập autograder chấm. Hệ quả:
   - Con số 83.6% bị lạc quan, và câu "độ chính xác trên tập kiểm định" trong `q4_language_id.tex` không đúng.
   - `sumary.md` ghi Q4 chấm "Validation accuracy ≥ 81%", nhưng autograder thực tế chấm trên tập test.
   - Cách sửa: trong `train.py` dùng `dataset._predict('dev')` để chọn mô hình theo tập dev (không sửa `backend.py`), rồi chạy lại `q4`.
9. **Dấu thập phân**: `final_report.tex` đang dùng dấu chấm (0.02, 83.6%) cho khớp với code và output của autograder; Q3 đang dùng dấu phẩy ("97,2%"). Cần thống nhất một kiểu.
