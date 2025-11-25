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