# Báo cáo Công Việc Đã Hoàn Thành - Lab 5

## 1. Mô tả công việc đã triển khai

Trong **Lab 5**, các kỹ thuật **phân loại văn bản (text classification)** đã được triển khai để giải quyết bài toán *
*phân tích cảm xúc (sentiment analysis)**.  
Công việc tập trung vào việc **xây dựng và so sánh hiệu năng** giữa mô hình sklearn cơ bản và các pipeline xử lý dữ liệu
lớn bằng **PySpark**.

---

## 2. Các file chính đã được triển khai

### **File `src/models/text_classifier.py` và `test/lab5_test.py`**

- Xây dựng lớp `TextClassifier` cơ bản bằng Python, sử dụng `LogisticRegression` từ thư viện **scikit-learn**.
- Lớp này đóng gói quá trình `fit` (huấn luyện) và `predict` (dự đoán), cùng hàm `evaluate` để tính toán các độ đo *
  *accuracy**, **precision**, **recall**, **f1-score**.
- File `lab5_test.py` là script kiểm thử, sử dụng một tập dữ liệu rất nhỏ (6 câu) để xác minh tính đúng đắn của lớp
  `TextClassifier`.

### **File `test/lab5_spark_sentiment_analysis.py`**

- Triển khai một pipeline phân tích cảm xúc hoàn chỉnh bằng **PySpark** trên tập dữ liệu `sentiments.csv`.
- Pipeline bao gồm các bước tiền xử lý chuẩn:
    - `Tokenizer`: tách từ
    - `StopWordsRemover`: loại bỏ stop words
    - `HashingTF`: tạo vector đặc trưng thô
    - `IDF`: tính trọng số TF-IDF
- Sử dụng mô hình **LogisticRegression** của Spark ML để huấn luyện và phân loại.
- Đánh giá mô hình bằng `MulticlassClassificationEvaluator`.

### **File `test/lab5_improvement_test.py`**

- Một phiên bản **cải tiến** của pipeline PySpark, nhằm tăng độ chính xác.
- Giữ nguyên các bước tiền xử lý (`Tokenizer`, `StopWordsRemover`, `HashingTF`, `IDF`).
- Thay thế mô hình **LogisticRegression** bằng **MultilayerPerceptronClassifier (MLP)**.
- Cấu hình các tầng (**layers**) cho mạng nơ-ron để xử lý **10,000 đặc trưng đầu vào** từ `HashingTF`.

---

## 3. Hướng dẫn chạy code

```bash
# Tải thư viện cần thiết
pip install pyspark scikit-learn

# Chạy script kiểm thử cho lớp TextClassifier (sklearn)
python test/lab5_test.py

# Chạy script huấn luyện mô hình Logistic Regression bằng PySpark
python test/lab5_spark_sentiment_analysis.py

# Chạy script huấn luyện mô hình MLP (Neural Network) bằng PySpark
python test/lab5_improvement_test.py
```

## 4. Kết quả

Các file kết quả đầu ra (output) tương ứng với các script đã chạy:

- `results/lab5_test_output.txt`
- `results/lab5_spark_sentiment_analysis_output.txt`
- `results/lab5_improvement_test_output.txt`

---

## 5 Phân tích và giải thích kết quả

### **5.1. `lab5_test.py` (sklearn cơ bản)**

- **Kết quả:**
    - Accuracy: 0.5
    - Precision: 0.25
    - F1-Score: 0.33

- **Phân tích:**
    - Script này chạy trên một tập dữ liệu **rất nhỏ** (4 mẫu huấn luyện, 2 mẫu kiểm thử).
    - Mô hình dự đoán sai 1 trong 2 mẫu kiểm thử (`"I hate this film..."` dự đoán là 1 thay vì 0).
    - Các chỉ số thấp là điều dự kiến và **chỉ mang tính xác minh** rằng code chạy đúng, không phản ánh hiệu năng thực
      tế.

---

### **5.2. `lab5_spark_sentiment_analysis.py` (Spark + Logistic Regression)**

- **Kết quả:**
    - Accuracy: 0.6000
    - Precision: 0.7778
    - Recall: 0.6000
    - F1-Score: 0.5238
    - Thời gian huấn luyện: 2.76 giây

- **Phân tích:**
    - Đây là mô hình **baseline** trên Spark.
    - Hiệu năng thấp (Accuracy chỉ 60%).
    - Phân tích kết quả chi tiết cho thấy mô hình có xu hướng **dự đoán sai các câu tiêu cực** (label 0.0) thành **tích
      cực** (prediction 1.0).
    - Điều này chứng tỏ **LogisticRegression** với đặc trưng TF-IDF chưa đủ phức tạp để học được các cấu trúc ngữ nghĩa
      phủ định trong văn bản.

---

### **5.3. `lab5_improvement_test.py` (Spark + MLP Neural Network)**

- **Kết quả:**
    - Accuracy: 0.9167
    - Precision: 0.9259
    - Recall: 0.9167
    - F1-Score: 0.9132
    - Thời gian huấn luyện: 73.9 giây

- **Phân tích:**
    - Đây là một **sự cải thiện vượt bậc**.
    - Chỉ bằng cách thay thế **LogisticRegression** bằng **MultilayerPerceptronClassifier**, độ chính xác đã tăng **từ
      60% lên gần 92%**.
    - Mô hình mạng nơ-ron (với cấu trúc tầng `[10000, 1024, 512, 64, 32, 2]`) có khả năng **học các mối quan hệ phi
      tuyến phức tạp hơn** giữa các đặc trưng TF-IDF, giúp phân loại hiệu quả hơn.

---

## 6. So sánh các phương pháp

| Phương pháp       | Accuracy | Precision | F1-Score | Thời gian huấn luyện | Phân tích                                        |
|-------------------|----------|-----------|----------|----------------------|--------------------------------------------------|
| Sklearn (Local)   | 0.50     | 0.25      | 0.33     | Rất nhanh            | Chỉ để kiểm thử, tập dữ liệu quá nhỏ.            |
| Spark (Log. Reg.) | 0.6000   | 0.7778    | 0.5238   | 2.76 giây            | Nhanh nhưng không chính xác.                     |
| Spark (MLP)       | 0.9167   | 0.9259    | 0.9132   | 73.9 giây            | Tốt nhất. Chậm hơn nhưng độ chính xác vượt trội. |

**Kết luận:**  
Mô hình **LogisticRegression** quá đơn giản cho bài toán này.  
Mô hình **MultilayerPerceptronClassifier (MLP)** tuy tốn nhiều thời gian huấn luyện hơn (74 giây so với 3 giây) nhưng *
*đạt hiệu quả cao hơn hẳn**, thể hiện sự **đánh đổi giữa thời gian/tài nguyên tính toán và độ chính xác mô hình**.

---

## 7. Khó khăn gặp phải và cách giải quyết

### **Khó khăn 1: Hiệu năng mô hình baseline thấp**

- **Vấn đề:** LogisticRegression ban đầu chỉ đạt Accuracy ~60%, nhầm lẫn nhiều mẫu tiêu cực.
- **Giải pháp:** Thay thế bằng **MultilayerPerceptronClassifier** trong file `lab5_improvement_test.py`.  
  → Accuracy tăng lên **91.7%**.

---

### **Khó khăn 2: Cảnh báo (Warnings) khi chạy Spark**

- **Hiện tượng:**
    - `WARN Utils: Your hostname... resolves to a loopback address`
    - `WARN NativeCodeLoader: Unable to load native-hadoop library`
- **Nguyên nhân:** Các cảnh báo này phổ biến khi chạy Spark ở chế độ `local[*]`.
- **Giải pháp:** Có thể **bỏ qua**, vì không ảnh hưởng đến kết quả thực thi.

---

### **Khó khăn 3: Cảnh báo về kích thước tác vụ (Task binary)**

- **Hiện tượng:**  
  `WARN DAGScheduler: Broadcasting large task binary with size 82.6 MiB`
- **Phân tích:** Cảnh báo này cho biết mô hình MLP (với nhiều tầng và trọng số) có kích thước lớn, Spark cần **broadcast
  ** mô hình đến các worker.
- **Giải pháp:** Bình thường đối với các mô hình phức tạp — đây là lý do khiến **thời gian huấn luyện lâu hơn**.

---

## 8. Công cụ và tài liệu tham khảo

### **Thư viện và Framework**

- **Apache Spark (PySpark):** Framework tính toán phân tán dùng để xây dựng pipeline xử lý dữ liệu ML có khả năng mở
  rộng.
- **Scikit-learn (sklearn):** Thư viện học máy phổ biến cho Python, được dùng để xây dựng mô hình `TextClassifier` cơ
  bản.
- **Spark MLlib:** Thư viện học máy của Spark, cung cấp các công cụ:
    - **Feature Engineering:** `Tokenizer`, `StopWordsRemover`, `HashingTF`, `IDF`
    - **Classification Models:** `LogisticRegression`, `MultilayerPerceptronClassifier`
    - **Evaluation:** `MulticlassClassificationEvaluator`

---

