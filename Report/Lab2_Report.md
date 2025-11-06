# Báo cáo công việc đã hoàn thành của Lab 2

## Mô tả công việc đã triển khai

File `Lab17_NLPPipeline.scala` đã được thiết kế để xây dựng một pipeline xử lý văn bản, chuyển đổi văn bản thô thành các
vector đặc trưng số học bằng thuật toán TF-IDF với các bước như sau:

- Tải dữ liệu (Read Dataset): Đọc 1000 bản ghi giới hạn từ tập dữ liệu văn bản lớn `c4-train.00000-of-01024-30K.json.gz`
- Tokenization: Sử dụng `RegexTokenizer` để tách câu thành các token, đồng thời loại bỏ các dấu câu và ký tự đặc biệt
  dựa trên biểu thức chính quy.
- Loại bỏ Stop Words: Sử dụng `StopWordsRemover` để loại bỏ các từ phổ biến không mang nhiều ý nghĩa ngữ cảnh (ví dụ: "
  the", "is", "a") nhằm giảm nhiễu và tăng hiệu suất.
- Tính TF: Sử dụng `HashingTF` để chuyển đổi các token còn lại thành vector tần suất thuật ngữ (TF) có kích thước cố
  định 20,000 chiều.
- Tính IDF: Sử dụng `IDF` để chuẩn hóa các vector TF, giảm trọng số của các từ xuất hiện quá thường xuyên trong toàn bộ
  tập dữ liệu (tăng trọng số cho các từ khóa hiếm, đặc trưng).
- Xây dựng và fit pipeline: Gộp các bước trên vào pipeline và huấn luyện trên dữ liệu đầu vào để học các trọng số IDF.

Các file biến thể theo yêu cầu đề bài:

- File `Lab17_NLPPipeline_normal_tokenizer.scala`: Thay vì sử dụng `RegexTokenizer` để tokenization, file này sử dụng
  `Tokenizer` cơ bản để tokenization, do đó file này chỉ tách được các từ cách nhau bởi dấu trắng, không tokenize được
  các ký tự hoặc dấu câu đặc biệt.
- File `Lab17_NLPPipeline_change_features.scala`: Đổi số chiều đầu ra của `HashingTF` từ 20000 -> 1000.
- File `Lab17_NLPPipeline_ML.scala`: Sau khi vetorize văn bản, thêm phần hồi quy logistic để phân loại (nhãn giả định
  dựa trên độ dài văn bản).
- File `Lab17_NLPPipeline_Word2Vec.scala`: Vectorize văn bản sử dụng `Word2Vec` thay vì sử dụng TF-IDF.

File `Lab17_NLPPipeline_similarity.scala`:

- Thêm `limitDocuments` cho phép người dùng nhập vào số văn bản được xử lý thay
  vì cố định sẵn.
- Thêm phần in ra thời gian chạy từng giai đoạn cụ thể vào file result. Thêm phần chuẩn hóa `Normalizer`
  vector đặc trưng TF-IDF vào pipeline.
- Thêm phần tìm văn bản tương đồng bằng độ đo cosine

## Hướng dẫn chạy code

```bash
sbt "runMain com.baro.spark.Lab17_NLPPipeline"
sbt "runMain com.dangth2004.spark.Lab17_NLPPipeline_normal_tokenizer"
sbt "runMain com.dangth2004.spark.Lab17_NLPPipeline_change_features"
sbt "runMain com.dangth2004.spark.Lab17_NLPPipeline_ML"
sbt "runMain com.dangth2004.spark.Lab17_NLPPipeline_Word2Sec"
sbt "runMain com.dangth2004.spark.Lab17_NLPPipeline_similarity"
```

## Kết quả

Các file kết quả:

- File result: `results/lab17_pipeline_output.txt`, `results/lab17_pipeline_output_change_features.txt`,
  `results/lab17_pipeline_output_normal_tokenizer.txt`, `results/lab17_pipeline_ml_output.txt`,
  `results/lab17_pipeline_word2vec_output.txt`.
- File log: `log/lab17_metrics.log`, `log/lab17_metrics_ml.log`, `log/lab17_metrics_change_features.log`,
  `log/lab17_metrics_normal_tokenizer.log`, `log/lab17_metrics_word2vec.log`

## Phân tích và giải thích kết quả

- Thời gian biến đổi dữ liệu ổn định quanh mức 0.61 - 0.73 giây cho thấy quá trình trích xuất đặc trưng có hiệu suất
  tương đối nhất quán sau khi pipeline đã được lắp.
- File `Lab17_NLPPipeline.scala` đã hoàn thành tương đối tốt nhiệm vụ vetorize văn bản.
- File `Lab17_NLPPipeline_change_features.scala` giảm số chiều còn 1000 không những giảm thời gian lắp pipeline mà còn
  tăng. Ngoài ra số chiều ít dẫn đến va chạm băm nghiêm trọng, có thể dẫn đến mất các thông tin của đặc trưng.
- File `Lab17_NLPPipeline_normal_tokenizer.scala` sử dụng phương pháp tokenize cơ bản, dẫn đến không xử lý được các ký
  tự đặc biệt hoặc các dấu câu, làm ảnh hưởng đến việc vectorize.
- File `Lab17_NLPPipeline_Word2Vec.scala` sử dụng Word2Vec để vectorize văn bản, đây là phương pháp vectorize mạnh hơn
  dẫn đến thời gian là lâu nhất.
- File `Lab17_NLPPipeline_ML.scala` sử dụng Logistic Regression để phân loại, tuy nhiên nhãn giả định (dựa vào độ dài
  văn bản) là quá đơn giản, dẫn đến hiệu suất cực tốt của mô hình học máy này.
- File `Lab17_NLPPipeline_similarity.scala`: đã in ra được 5 văn bản tương đồng nhất với văn bản cho trước bằng việc
  tính khoảng cách cosine giữa vector đặc trưng của 2 văn bản.

## Khó khăn gặp phải và cách giải quyết

- Khó khăn chính gặp phải không nằm ở logic xử lý NLP hay thuật toán, mà là vấn đề tương thích môi trường khi chạy
  Apache Spark 4.0.0 trên môi trường Java 9+ (hoặc Java 21).
- Cách giải quyết: Cập nhật lại file `build.sbt`