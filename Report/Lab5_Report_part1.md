# Báo cáo Công Việc Đã Hoàn Thành - Lab 5 (Giới thiệu PyTorch) (tuần 1)

Phần Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM) (tuần 2) ở dưới

## 1. Mô tả công việc đã triển khai

Trong Lab này, chúng ta đã tìm hiểu các khái niệm cơ bản và thao tác cốt lõi của thư viện **PyTorch**, một framework học
sâu phổ biến.

Công việc tập trung vào việc thực hành các kỹ thuật nền tảng, bao gồm:

- Khởi tạo **Tensor** từ nhiều nguồn dữ liệu khác nhau (list, NumPy).
- Thực hiện các phép toán, slicing, và thay đổi hình dạng trên Tensor.
- Sử dụng hệ thống **autograd** để tự động tính toán đạo hàm.
- Xây dựng các mô hình mạng nơ-ron đơn giản bằng cách sử dụng `torch.nn`.

---

## 2. Các file/notebook chính đã được triển khai

**File:** `TrinhHaiDang_22001561_Lab06.ipynb`

Đây là file **Jupyter Notebook chính**, trình bày quy trình làm việc đầy đủ để khám phá các tính năng của PyTorch.  
Nội dung được chia thành các phần chính:

### Imports và Khởi tạo Tensor (In [1] - [6])

- Nhập các thư viện cần thiết (`numpy`, `torch`, `torch.nn`).
- Tạo tensor từ list, mảng NumPy, và sử dụng các hàm tiện ích như:
  `torch.ones_like`, `torch.rand_like`, `torch.rand`, `torch.ones`, `torch.zeros`.
- Kiểm tra các thuộc tính của tensor như `.shape`, `.dtype`, `.device`.

### Các phép toán trên Tensor (In [7] - [12])

- Thực hiện các phép toán cơ bản như cộng (+), nhân vô hướng (*).
- Thực hiện phép nhân ma trận (@) và chuyển vị (.T).
- Truy cập dữ liệu bằng **Indexing** và **Slicing** (lấy hàng, cột, giá trị).
- Thay đổi hình dạng tensor bằng `.reshape()`.

### Tự động tính Đạo hàm với autograd (In [13] - [17])

- Ví dụ lan truyền ngược (`.backward()`) trên một hàm toán học cơ bản `z = 3*y*y` để tính `x.grad`.
- Mô phỏng một bước huấn luyện mô hình: tính dự đoán (`pred`), mất mát (`loss`) bằng `binary_cross_entropy_with_logits`,
  rồi gọi `loss.backward()` để lấy đạo hàm cho `w.grad` và `b.grad`.

### Xây dựng Mô hình với `torch.nn` (In [18] - [21])

- Giới thiệu lớp `nn.Linear` (tầng tuyến tính) và `nn.Embedding` (tầng nhúng).
- Xây dựng mô hình tùy chỉnh `MyEmbeddingModel` kế thừa `nn.Module`, kết hợp `nn.Embedding`, `nn.Linear`, và `nn.ReLU`.
- Chạy dữ liệu đầu vào qua mô hình và kiểm tra `shape` của đầu ra.

---

## 3. Hướng dẫn chạy code

- Tải thư viện cần thiết: `pip install torch numpy`
- Chạy Jupyter Lab hoặc Notebook: `jupyter lab`
- Mở file .ipynb và chạy từng cell

---

## 4. Kết quả

Tất cả kết quả (output) của các lệnh đều được in trực tiếp bên dưới các ô code (cell) trong file notebook.  
Ví dụ: kết quả in tensor, kết quả các phép toán, và kết quả đạo hàm.

---

## 5. Phân tích và giải thích kết quả

### 5.1. Khởi tạo và Thao tác Tensor (In [2] - [12])

**Kết quả:**

Các tensor được tạo thành công với các kiểu dữ liệu và hình dạng mong muốn.  
Ví dụ: `x_data` có dtype là `torch.int64` và shape `[2, 2]`.  
Phép nhân ma trận `x_data @ x_data.T` cho ra:

tensor([[ 5, 11],
[11, 25]])

**Phân tích:**

Phần này cho thấy **PyTorch API** linh hoạt, cú pháp tương đồng với NumPy (ví dụ `torch.from_numpy`).  
Các thao tác `reshape`, `slicing` là thiết yếu để chuẩn bị dữ liệu cho mô hình.

---

### 5.2. Tự động tính Đạo hàm - autograd (In [13] - [17])

**Kết quả:**

- Ví dụ 1: Với `x=1 (requires_grad=True)`, `y=x+2`, `z=3*y*y`.  
  Sau khi gọi `z.backward()`, kết quả `x.grad = tensor([18.])`.

- Ví dụ 2: Mô hình tuyến tính đơn giản, sau `loss.backward()`, các tham số `w`, `b` có giá trị đạo hàm được cập nhật (
  `w.grad`, `b.grad`).

**Phân tích:**

Theo quy tắc chuỗi (chain rule):

z = 3*(x+2)^2
dz/dx = 6*(x+2)

Tại `x=1`, `dz/dx = 18`.  
Ví dụ 2 mô phỏng cốt lõi của quá trình huấn luyện mạng nơ-ron — **autograd** theo dõi các phép toán và tự động tính
gradient khi gọi `.backward()`.

---

### 5.3. Xây dựng Mô hình với torch.nn (In [18] - [21])

**Kết quả:**

- `nn.Linear` và `nn.Embedding` được khởi tạo thành công.
- Mô hình `MyEmbeddingModel` được định nghĩa và khởi tạo.
- `input_data` có shape `[1, 4]` (1 câu, 4 từ).
- `output_data` có shape `[1, 4, 2]`.

**Phân tích:**

- `nn.Embedding` hoạt động như một bảng tra cứu (*lookup table*), ánh xạ mỗi index thành vector đặc trưng.
- Mô hình `MyEmbeddingModel` minh họa cách kết hợp các lớp:

[1, 4] → Embedding → [1, 4, 16]
→ Linear + ReLU → [1, 4, 8]
→ Output layer → [1, 4, 2]


---

## 6. Các khái niệm cốt lõi đã học

| Khái niệm            | Mô tả                                                                                      | Ví dụ trong Lab                           |
|----------------------|--------------------------------------------------------------------------------------------|-------------------------------------------|
| **Tensor**           | Cấu trúc dữ liệu đa chiều, tương tự NumPy nhưng có hỗ trợ GPU và tính toán đạo hàm.        | `torch.tensor(data)`, `torch.rand(shape)` |
| **autograd**         | Hệ thống tự động tính đạo hàm. Theo dõi các phép toán trên tensor có `requires_grad=True`. | `x = torch.ones(1, requires_grad=True)`   |
| **.backward()**      | Bắt đầu quá trình lan truyền ngược, tính đạo hàm từ một tensor (thường là loss).           | `z.backward()`, `loss.backward()`         |
| **.grad**            | Lưu trữ giá trị đạo hàm tích lũy sau khi gọi `.backward()`.                                | `x.grad`, `w.grad`, `b.grad`              |
| **nn.Module**        | Lớp cơ sở để xây dựng mô hình mạng nơ-ron.                                                 | `class MyEmbeddingModel(nn.Module)`       |
| **Các lớp (Layers)** | Các khối xây dựng sẵn như `nn.Linear`, `nn.Embedding`, `nn.ReLU`.                          | `self.linear = nn.Linear(...)`            |
| **forward(self, x)** | Định nghĩa luồng dữ liệu trong mô hình.                                                    | `def forward(self, x):`                   |

---

## 7. Khó khăn gặp phải và cách giải quyết

**Khó khăn 1: Hiểu cơ chế hoạt động của autograd**

**Vấn đề:**  
Khó hiểu vì sao một số tensor có `grad_fn` còn các tensor khác thì không, và tại sao `.grad` chỉ được cập nhật cho
“tensor lá” (*leaf tensor*).

**Giải pháp:**

- Chỉ tensor có `requires_grad=True` mới được theo dõi.
- `grad_fn` chỉ tồn tại cho tensor trung gian (kết quả phép toán).
- `.grad` chỉ có ở tensor lá (ví dụ: `x`, `w`, `b`).
- Phải gọi `.backward()` trên một **tensor vô hướng (scalar)** để bắt đầu quá trình tính gradient.

---

## 8. Công cụ và tài liệu tham khảo

**Thư viện và Framework:**

- **PyTorch**: Framework học sâu mã nguồn mở chính được sử dụng.
- **NumPy**: Thư viện tính toán khoa học của Python.

**Các module PyTorch đã sử dụng:**

- `torch.Tensor` – cấu trúc dữ liệu cốt lõi.
- `torch.autograd` – tính toán đạo hàm tự động.
- `torch.nn` – xây dựng mạng nơ-ron (các lớp `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.ReLU`).
- `torch.nn.functional` – các hàm tiện ích, ví dụ `binary_cross_entropy_with_logits`.

------------------------------