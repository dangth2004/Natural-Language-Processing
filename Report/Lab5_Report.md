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

# Báo cáo Lab 5: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

## 1. Mô tả công việc đã triển khai

Trong Lab này, chúng ta đã triển khai và so sánh hiệu năng của 4 pipeline khác nhau cho bài toán phân loại ý định (
intent classification) trên bộ dữ liệu `hwu`.
Các pipeline được so sánh bao gồm:

- TF-IDF + Logistic Regression (Baseline 1)
- Word2Vec (vector trung bình) + Dense Layer (Baseline 2)
- Embedding Layer (pre-trained) + LSTM (Mô hình nâng cao 1)
- Embedding Layer (học từ đầu) + LSTM (Mô hình nâng cao 2)

Mục tiêu là để hiểu rõ hạn chế của các mô hình truyền thống (1, 2) và đánh giá sức mạnh của mô hình chuỗi (3, 4) trong
việc nắm bắt ngữ cảnh.

## 2. File/notebook chính đã được triển khai

File: `test/lab5_rnns_text_classification.ipynb`. Đây là file Jupyter Notebook chính, chứa toàn bộ mã nguồn để:

- Tải và tiền xử lý dữ liệu (từ cell 2, 3).
- Triển khai và huấn luyện Pipeline 1 (TF-IDF) (cell 4).
- Triển khai và huấn luyện Pipeline 2 (Word2Vec Avg) (cell 5).
- Triển khai và huấn luyện Pipeline 3 (LSTM Pre-trained) (cell 6).
- Triển khai và huấn luyện Pipeline 4 (LSTM Scratch) (cell 7).
- Tổng hợp kết quả và thực hiện phân tích (cell 8, 9).

## 3. Kết quả

### 3.1. So sánh định lượng

| Pipeline                       | F1-score (Macro) | Test Loss |
|--------------------------------|------------------|-----------|
| TF-IDF + Logistic Regression   | 0.828865         | NaN       |
| Word2Vec (Avg) + Dense         | 0.146507         | 3.083510  |
| Embedding (Pre-trained) + LSTM | 0.247700         | 2.637293  |
| Embedding (Scratch) + LSTM     | 0.827488         | 0.717818  |

### 3.2. Phân tích định tính

Kiểm tra dự đoán của các mô hình trên các câu "khó" (có yếu tố phủ định hoặc cấu trúc phức tạp), theo output của cell 8:
Câu 1: can you remind me to not call my mom (Kỳ vọng: `reminder_create`)

- TF-IDF+LR: `calendar_set`
- W2V(Avg)+Dense: `general_explain`
- LSTM + Pretrained Emb: `general_explain`
- LSTM + Scratch Emb: `calendar_set`

Câu 2: is it going to be sunny or rainy tomorrow (Kỳ vọng: `weather_query`)

- TF-IDF+LR: `weather_query`
- W2V(Avg)+Dense: `alarm_query`
- LSTM + Pretrained Emb: `social_query`
- LSTM + Scratch Emb: `weather_query`

Câu 3: find a flight from new york to london but not through paris (Kỳ vọng: `flight_search`)

- TF-IDF+LR: `transport_query`
- W2V(Avg)+Dense: `email_sendemail`
- LSTM + Pretrained Emb: `transport_ticket`
- LSTM + Scratch Emb: `transport_query`

## 4. Phân tích và giải thích kết quả

- TF-IDF + Logistic Regression hoạt động xuất sắc: Mặc dù mô hình này bỏ qua hoàn toàn thứ tự từ và các mối quan hệ ngữ
  pháp phức tạp (như "not through paris"), F1-score cao (0.828) và kết quả định tính chính xác cho thấy: việc phân loại
  ý định trong bộ dữ liệu này phụ thuộc chủ yếu vào các từ khóa (keywords) đặc trưng (ví dụ: "flight", "sunny", "remind
  me"). Yếu tố phủ định ("not") dường như không đủ trọng số để làm thay đổi ý định chính (transport_query).
- Word2Vec (Avg) + Dense thất bại hoàn toàn: Như giả định, việc lấy trung bình (averaging) các vector từ đã làm mất hoàn
  toàn cấu trúc chuỗi và thông tin ngữ nghĩa quan trọng. Một câu có ý nghĩa (find a flight...) và một câu vô nghĩa (a
  flight find...) sẽ có vector biểu diễn gần giống nhau, khiến mô hình không thể học được (F1=0.14).
- LSTM + Pretrained Embedding gây bất ngờ lớn: Trái với lý thuyết thông thường (rằng pre-trained embedding giúp tổng
  quát hóa tốt hơn khi dữ liệu ít), kết quả F1 (0.24) cho thấy sự không tương thích (mismatch) nghiêm trọng về miền dữ
  liệu (domain). Các vector từ được huấn luyện trước (ví dụ: trên Wikipedia, tin tức) không mang ngữ nghĩa chính xác cho
  các câu lệnh ngắn, đặc thù của bộ dữ liệu này.
- LSTM + Embedding (Scratch) là mô hình học sâu tốt nhất: Đây là kết luận then chốt. Khi được phép học embedding từ
  đầu (from scratch), mô hình LSTM đã xây dựng được một không gian vector từ vựng tùy chỉnh (custom), hoàn toàn phù hợp
  với domain dữ liệu. Nó đã học được các sắc thái ngữ nghĩa (và có thể cả thứ tự từ) cần thiết để đạt F1-score cao (
  0.827), ngang bằng với baseline TF-IDF mạnh nhất. Điều này cho thấy dữ liệu là đủ "phong phú" để tự học embedding mà
  không bị overfitting.

## 5. Các công cụ và tài liệu tham khảo

- Pandas: Tải và quản lý dữ liệu
- Scikit-learn:
    - `LabelEncoder`: Mã hóa nhãn.
    - `TfidfVectorizer`, `LogisticRegression`: Triển khai Pipeline 1.
    - `classification_report`, `f1_score`: Đánh giá mô hình.
- Gensim:
    - `Word2Vec`: Huấn luyện mô hình Word2Vec.
- TensorFlow / Keras:
    - `Sequential`, `Dense`, `Dropout`: Xây dựng mô hình.
    - `Tokenizer`, `pad_sequences`: Tiền xử lý văn bản cho RNN.
    - `Embedding`, `LSTM`: Các lớp mạng nơ-ron chính.
    - `EarlyStopping`: Ngăn ngừa overfitting.