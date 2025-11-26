# Báo cáo Lab 5: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)

## 1. Mô tả công việc đã triển khai

Trong phần này, chúng ta mở rộng bài toán gán nhãn chuỗi sang Nhận dạng thực thể tên (Named Entity Recognition - NER).
Mục tiêu là xác định và phân loại các thực thể trong văn bản như tên người (PER), tổ chức (ORG), địa điểm (LOC) và các
thực thể khác (MISC) sử dụng bộ dữ liệu chuẩn CoNLL-2003.

Các bước triển khai cụ thể bao gồm:

- Tải dữ liệu từ thư viện `datasets`: Sử dụng bộ dữ liệu `conll2003` thay vì đọc file thủ công, giúp chuẩn hóa quá trình
  nạp dữ liệu.
- Xử lý nhãn BIO: Bài toán NER sử dụng lược đồ gán nhãn BIO (Beginning, Inside, Outside). Hệ thống cần phân biệt được
  điểm bắt đầu (B-) và phần tiếp theo (I-) của một thực thể.
- Xây dựng từ điển: Tương tự bài toán POS, tạo ánh xạ `word_to_ix` và sử dụng danh sách nhãn có sẵn từ metadata của
  dataset (`['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']`).
- Nâng cấp mô hình: Thay thế lớp RNN cơ bản bằng LSTM (Long Short-Term Memory) để xử lý tốt hơn các phụ thuộc xa trong
  câu, điều này rất quan trọng đối với việc nhận diện thực thể tên dài.
- Huấn luyện và Đánh giá: Thực hiện huấn luyện trên GPU (nếu có) và đánh giá độ chính xác trên tập validation.

## 2. File/notebook chính đã được triển khai

File: `test/lab5_rnn_for_ner.ipynb`. Notebook chứa mã nguồn thực hiện các bước sau:

- Tải dữ liệu & Xây dựng Vocab: Cell 1, 3 (Sử dụng `load_dataset` và duyệt token để tạo từ điển).
- Dataset & Dataloader: Cell 4 (Class `NERDataset` và hàm `collate_fn` xử lý padding cho cả sentence và tags).
- Mô hình: Cell 5 (Class `SimpleRNNForTokenClassification` nhưng sử dụng` nn.LSTM` thay vì `nn.RNN`).
- Huấn luyện: Cell 6 (Vòng lặp training qua 3 epoch).
- Dự đoán: Cell 7 (Hàm `predict_sentence` kiểm thử trên các câu thực tế).

## 3. Kết quả

### 3.1 Thông số cấu hình và dữ liệu:

- Bộ dữ liệu: `CoNLL-2003`.
- Kích thước từ điển (Vocab): 23,625 từ.
- Số lượng nhãn (Tags): 9 nhãn (bao gồm O và các cặp B/I cho 4 loại thực thể).
- Tham số mô hình:
    - Embedding dim: 100
    - Hidden dim: 256 (Tăng gấp đôi so với bài toán POS để tăng khả năng lưu trữ ngữ cảnh).
    - Learning rate: 0.001
    - Optimizer: Adam
    - Số Epoch: 3

### 3.2 Kết quả định lượng (Huấn luyện)

Mô hình LSTM hội tụ rất nhanh chỉ sau 3 epoch:

| Epoch | Train loss | Val loss | Train Acc | Val Acc | Training time | Validation time | Total time |
|-------|------------|----------|-----------|---------|---------------|-----------------|------------|
| 01    | 0.620      | 0.445    | 	0.841    | 	0.877  | 2.207s        | 0.151s          | 2.358s     |
| 02    | 0.324      | 0.291    | 	0.903    | 	0.916  | 2.040s        | 0.144s          | 2.184s     |
| 03    | 0.201      | 0.233    | 	0.939    | 	0.931  | 1.983s        | 0.142s          | 2.125s     |

Nhận xét: Độ chính xác trên tập Validation đạt 93.1%. Việc sử dụng LSTM và Hidden dim lớn (256) đã giúp mô hình học rất
hiệu quả các mẫu thực thể tên dù thời gian huấn luyện ngắn.

### 3.3 Kết quả định tính (Dự đoán thực tế)

Thử nghiệm 1: "VNU University of Science is located in Hanoi". Kết quả:

- VNU: `B-ORG` (Chính xác)
- University: `I-ORG` (Chính xác)
- of: `I-ORG` (Chính xác)
- Science: `I-ORG` (Chính xác)
- is: O (Chính xác)
- located: O (Chính xác)
- in: O (Chính xác)
- Hanoi: O (sai)

Thử nghiệm 2: "WHO is an organization based in Geneva". Kết quả:

- WHO: O (sai)
- is: O (Chính xác)
- an: O (Chính xác)
- organization: O (Chính xác)
- based: O (Chính xác)
- in: O (Chính xác)
- Geneva: `B-LOC` (Chính xác)

## 4. Phân tích và giải thích kết quả

- Khả năng nhận diện thực thể dài tốt: Trong câu 1, mô hình nhận diện rất chính xác cụm thực thể dài và phức tạp "VNU
  University of Science" (`B-ORG` theo sau bởi chuỗi `I-ORG`). Điều này cho thấy kiến trúc LSTM đã phát huy tác dụng
  trong
  việc ghi nhớ ngữ cảnh dài và sự phụ thuộc giữa các từ liên tiếp.
- Vấn đề "False Negative" với thực thể đơn lẻ:
    - Hanoi (O): Mô hình không nhận diện được "Hanoi" là địa điểm (`LOC`). Nguyên nhân có thể do từ này là OOV (
      Out-of-Vocabulary) trong bộ dữ liệu CoNLL-2003 (vốn dựa trên tin tức Reuters thập niên 90, có thể ít nhắc đến
      Hanoi), dẫn đến việc từ này bị gán vector `<UNK>` hoặc mô hình chưa học đủ ngữ cảnh để suy luận.
    - WHO (O): Mô hình bỏ sót tổ chức "WHO". Đây là một trường hợp Ambiguity (Đa nghĩa) điển hình. "Who" cũng là một đại
      từ phổ biến (được gán nhãn O). Mặc dù input viết hoa ("WHO"), nhưng nếu trong quá trình huấn luyện hoặc xây dựng
      từ điển, sự phân biệt giữa "WHO" (tổ chức) và "who" (đại từ) không đủ mạnh, mô hình có xu hướng chọn nhãn an toàn
      nhất là "O" (Outside).
- Kết luận: Mô hình có độ chính xác (Precision) cao khi đã phát hiện ra thực thể (như chuỗi VNU...), nhưng độ phủ (
  Recall) còn hạn chế đối với các thực thể hiếm gặp hoặc đa nghĩa (Hanoi, WHO). Để cải thiện, cần xem xét sử dụng
  Pre-trained Embeddings (như GloVe hoặc BERT) để xử lý tốt hơn các từ OOV và ngữ cảnh.

## 5. Các công cụ

Thư viện chính:

- Thư viện `datasets` (HuggingFace): Giúp tải và quản lý dữ liệu CoNLL-2003.
- `torch.nn.LSTM`: Module mạng hồi quy cải tiến, giải quyết vấn đề vanishing gradient.
- `pad_sequence`: Công cụ chuẩn hóa độ dài batch.
