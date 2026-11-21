# Sprint 2


## Goal



## Backlog

(Vinh)
- Cần review tên báo cáo.
- Mẫu view dataset (bảng mẫu khoảng 5 10 dòng mỗi q)
- Phần tóm tắt viết ngắn lại, cô đọng ý chính.
- Phần giới thiệu, viết ngắn lại, bỏ thông tin nhiễu, Q1, Q2, bỏ bullet Q1, Q2...

(Khang)
- Kiến thức nền tảng
    - Thêm section về các loại học (ngán)
    - Thêm section về experience, performance, task và MSE (Kiến thức chung) (ngắn)
    - Chỉ mô tả, liệt kê, bỏ công thức đi, và công thức sẽ thêm vào từng Q
- Thêm task coreach

(Vinh, Đức)
- Review lại cài đặt, có thể làm cách nào đó visualize được thay vì diễn dải bằng văn
- Review lại vị trí đặt bảng
- Review và viết lại phần phân tích, sao cho ngắn và đối chiếu tài liệu tham khảo sao cho khớp


## Notes

**Vấn đề còn mở (review báo cáo):**

- Q5: câu "84% lên 98% sau 5 epoch" lấy từ lần chạy autograder cũ, còn hình kernel lấy từ `scripts/figures/q5_kernel.py` (86% lên 98%, 6 epoch). Cần thống nhất một nguồn số liệu.
- Code: comment ở `train.py:244` ghi "same as Q3", nhưng Q3 đã dừng ở 97.5%. Docstring của `DigitConvolutionalModel` vẫn ghi "pretrained weights", không còn đúng.
- Q3: code không xáo dữ liệu (DataLoader không shuffle), báo cáo chưa nêu lý do.
- Nếu mỗi Q đều ghi siêu tham số ngay trong phần Cài đặt thì bảng siêu tham số (`tab:hparams`) bị thừa, cân nhắc bỏ.
- Kiến thức nền tảng (Khang): cân nhắc thêm pseudocode vòng huấn luyện chung, để phần Cài đặt của từng Q chỉ nêu điểm khác.
- Thuật ngữ: "mạng nơ-ron hồi quy" (recurrent) dễ nhầm với "hồi quy" (regression) ở Q2, cân nhắc dùng "hồi tiếp".
- Mẫu dữ liệu: Q4 cần bảng mẫu từ và ngôn ngữ; Q1, Q2 có lẽ không cần.
- Chưa build lại PDF sau khi sửa Q3, Q5, cần kiểm tra vị trí hình và bảng.

**Thành viên:**

| MSSV | Họ tên | Vai trò |
| ---- | ------ | ------- |
| 23020715 | Lương Thế Vinh | Nhóm trưởng |
| 24020169 | Nguyễn Trọng Khang | Thành viên |
| 24022628 | Nguyễn Minh Đức | Thành viên |
