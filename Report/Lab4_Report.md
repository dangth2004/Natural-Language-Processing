# Báo cáo công việc đã hoàn thành của Lab 14

## Mô tả công việc đã triển khai

Trong Lab 4, các kỹ thuật về word embedding đã được triển khai bằng Python và PySpark để biểu diễn từ và văn bản dưới
dạng vector số học, nắm bắt các mối quan hệ ngữ nghĩa. Các task đã hoàn thành:

- Task 1 + 2: File `src/representations/word_embedder.py` và `test/lab4_test.py`.
- Task 3: File `test/lab4_embedding_training_demo.py`.
- Task 4: File `test/lab4_spark_word2vec_demo.py`
- Task 5: `TrinhHaiDang_22001561_Lab03.ipynb`

Các file chính đã được triển khai như sau:

1) File `src/representations/word_embedder.py`: Xây dựng lớp `WordEmbedder` để đóng gói các chức năng của mô hình word
   embedding đã được huấn luyện sẵn (pre-trained) từ `gensim` với các tính năng chính:
    - Tải mô hình: Tải một mô hình pre-trained (ví dụ: `glove-wiki-gigaword-50`) một cách tự động.
    - Trích xuất Vector: Lấy vector đặc trưng cho một từ cụ thể.
    - Đo lường độ tương đồng: Tính toán độ tương đồng cosine giữa hai từ.
    - Tìm từ đồng nghĩa: Tìm N từ gần nhất với một từ cho trước.
    - Vector hóa văn bản: Chuyển đổi một đoạn văn bản thô thành một vector đặc trưng duy nhất bằng cách lấy trung bình
      cộng các vector của từng từ trong văn bản đó.
2) File `test/lab4_test.py`: Script kiểm thử cho lớp `WordEmbedder`, thực hiện các tác vụ để xác minh tính đúng đắn của
   các phương thức đã triển khai, bao gồm lấy vector, tính độ tương đồng và vector hóa một câu hoàn chỉnh.
3) File `test/lab4_embedding_training_demo.py`: Script minh họa quy trình huấn luyện một mô hình `Word2Vec` từ đầu bằng
   `gensim` trên một tập dữ liệu tùy chỉnh (`en_ewt-ud-train.txt`). Các bước thực hiện:
    - Đọc và xử lý dữ liệu văn bản theo từng câu để tiết kiệm bộ nhớ.
    - Huấn luyện mô hình Word2Vec mới từ dữ liệu đã đọc.
    - Lưu mô hình đã huấn luyện ra file `results/word2vec_ewt.model`.
    - Sử dụng mô hình vừa tạo để tìm từ tương đồng và giải các bài toán loại suy (analogy).
4) File `test/lab4_spark_word2vec_demo.py`: Script sử dụng Apache Spark để huấn luyện mô hình `Word2Vec` trên một tập dữ
   liệu lớn hơn (`c4-train.00000-of-01024-30K.json`). Script này thể hiện khả năng xử lý dữ liệu phân tán:
    - Khởi tạo một Spark Session.
    - Đọc và tiền xử lý dữ liệu văn bản (làm sạch, tách từ) bằng các hàm tích hợp của Spark SQL.
    - Huấn luyện mô hình Word2Vec trên một DataFrame.
    - Sử dụng mô hình để tìm các từ đồng nghĩa.

## Hướng dẫn chạy code

```bash
# Tải thư viện cần thiết
pip install gensim pyspark

# Chạy script kiểm thử cho lớp WordEmbedder
python test/lab4_test.py

# Chạy script huấn luyện mô hình Word2Vec từ đầu bằng gensim
python test/lab4_embedding_training_demo.py

# Chạy script huấn luyện mô hình Word2Vec bằng PySpark
python test/lab4_spark_word2vec_demo.py
```

## Kết quả

Các file kết quả:

- File result: `results/lab4_test_output.txt`, `results/lab4_embedding_training_demo_output.txt`,
  `results/lab4_spark_word2vec_demo_output.txt`.
- File trọng số model: `results/word2vec_ewt.model`

## Phân tích và giải thích kết quả

Phân tích kết quả:

- Sử dụng mô hình Pre-trained (`src/representations/word_embedder.py`): mô hình pre-trained `glove-wiki-gigaword-50` thể
  hiện khả năng nắm bắt ngữ nghĩa rất tốt cho các tác vụ tổng quát. Cụ thể, độ tương đồng giữa `king` và `queen` (
  0.7839) cao hơn hẳn so với `king` và `man` (0.5309), cho thấy mô hình hiểu được mối quan hệ về vai trò và giới tính.
  Các từ tương đồng nhất với `computer` (như `software`, `technology`, `internet`,...) đều rất liên quan, chứng tỏ chất
  lượng của mô hình.
- Huấn luyện từ đầu (`test/lab4_embedding_training_demo.py`): Kết quả từ
  `results/lab4_embedding_training_demo_output.txt` cho thấy việc tự huấn luyện trên tập dữ liệu nhỏ (
  `en_ewt-ud-train.txt`) có những hạn chế rõ rệt. Mặc dù tìm ra được một số từ liên quan đến `computer` như `laptop` và
  `software`, nhưng cũng có những từ không liên quan như `soil` và `radio`. Đặc biệt, mô hình thất bại trong bài toán
  loại suy kinh điển (`king - man + woman`), cho ra kết quả là `tract` thay vì `queen`. Điều này chứng tỏ tập dữ liệu
  chưa đủ lớn và đa dạng để mô hình học được các mối quan hệ ngữ nghĩa phức tạp.
- Huấn luyện trên Spark (`test/lab4_spark_word2vec_demo.py`): `results/lab4_spark_word2vec_demo_output.txt` cho thấy sức
  mạnh của việc huấn luyện trên dữ liệu lớn. Các từ tương đồng nhất với `computer` (như `firewall`, `desktop`, `laptop`)
  đều rất chính xác và mang tính kỹ thuật cao, phản ánh ngữ cảnh của tập dữ liệu C4. Kết quả này vượt trội so với mô
  hình tự huấn luyện bằng gensim và cho thấy để có một mô hình embedding chất lượng, dữ liệu lớn là yếu tố then chốt.
  Thời gian huấn luyện (156 giây) cũng cho thấy quy mô xử lý lớn hơn đáng kể.

So sánh các phương pháp:

- Pre-trained: Lựa chọn tốt nhất cho các bài toán ngôn ngữ phổ thông, tiết kiệm thời gian và tận dụng được tri thức từ
  kho dữ liệu khổng lồ.
- Training from scratch (gensim): Hữu ích khi cần vector cho các từ vựng chuyên ngành không có trong mô hình
  pre-trained, nhưng chất lượng phụ thuộc rất nhiều vào quy mô và độ sạch của dữ liệu huấn luyện. Cần một lượng dữ liệu
  đủ lớn để có kết quả tốt.
- Training with Spark: Là giải pháp bắt buộc khi huấn luyện mô hình từ đầu trên các tập dữ liệu cực lớn (Big Data) để
  đảm bảo chất lượng và khả năng mở rộng.

## Khó khăn gặp phải và cách giải quyết

## Công cụ và tài liệu tham khảo

### Thư viện và Framework

- Gensim: Một thư viện Python mã nguồn mở mạnh mẽ cho xử lý ngôn ngữ tự nhiên, đặc biệt trong việc lập mô hình chủ đề
  không giám sát và word embedding. Được sử dụng để tải mô hình pre-trained và huấn luyện mô hình Word2Vec.
- Apache Spark: Một framework tính toán phân tán mã nguồn mở được sử dụng để xử lý dữ liệu lớn. Trong lab này, PySpark (
  API Python cho Spark) được dùng để huấn luyện mô hình Word2Vec trên tập dữ liệu C4.

### Mô hình Pre-trained

- GloVe (Global Vectors for Word Representation): Một thuật toán học không giám sát để có được các biểu diễn vector cho
  các từ. Mô hình `glove-wiki-gigaword-50` được huấn luyện trên kho dữ liệu Wikipedia và Gigaword.