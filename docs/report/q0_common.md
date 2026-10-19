### Kiến thức chung:

#### Tensor
Multi-dims array, sử dụng để lưu trữ và xử lý dữ liệu số trong ML và AI, Trong ML, AI đều xây dựng mô hình và phép tính dựa trên tensor

Phân loại tensor theo bậc
- Scalar (bậc 0): số đơn 
- Vector (bậc 1): mảng 1 chiều
- Matrix (bậc 2): mảng 2 chiều
- N-tensor... (Ví dụ, hình ảnh màu là mảng 3 chiều) h x w x c

### Input, Weight, Bias, Output
VD: Dự đoán khách hàng có mua xe không? 
- Có bảng dữ liệu lịch sử mua hàng (input: x): Khách hàng (tuổi, thu nhập, đã kết hôn, đã có xe) => x = [x1, x2, x3, x4]
- Đánh trọng số (weight: w): tuổi (x1) = w1, thu nhập (x2) = w2, đã kết hôn (x3) = w3, đã có xe (x4) = w4,
- Bias: đánh giá của thị trường ở thời điểm đó: ví dụ lãi xuất cho vay: b

=> công thức: z = sum(wi*x1) + b (output)

### Gradient descent, batch, iterations, epochs

#### Hàm kích hoạt
- Hàm kích hoạt (activition function): Khi mô hình tính toán, nó trả kết quả về là 1 số thực (z). Hàm kích hoạt là hàm lấy giá trị z này đưa về miền giá trị đã xác định.
    - VD: model ảnh -> z, f(z) -> Chó/Mèo
- Activation functions: 
    - Step-function: f(z) = [0 or 1]
    - Linear: f(z) = z                              (hồi quy, dự đoán giá nhà, ...)
    - Sigmoid: f(z) = 1 / 1 + e^-z, 0 <= f(z) <= 1  (classify đơn lớp, nhị phân)
    - Softmax: f(zi) = e^zi / sum(e^zi)             (classify đa lớp)

    - Tanh: f(z)                                    (trong hidden layer)
    - ReLU: f(z) = max(0,z)                         (trong hidden layer)

#### Hàm mất mát
- Hàm mất mát (loss function): Là hàm tính mức độ sai lệch giữa giá trị dự đoán và giá trị thực tế (so sánh dự đoán với ground truth)
- Loss functions: 
    - MSE   (mean square error)
    - MAE   (mean absolute error)
    - BCE   (binary cross-entropy)
    - CE    (cross-entropy)