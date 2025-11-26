# Báo cáo Lab 5: Xây dựng mô hình RNN cho bài toán Part-of-Speech Tagging

## 1. Mô tả công việc đã triển khai

Trong phần này của bài Lab, chúng ta tập trung giải quyết bài toán gán nhãn từ loại (Part-of-Speech Tagging - POS
Tagging) sử dụng kiến trúc mạng nơ-ron hồi quy (RNN) cơ bản. Đây là bài toán thuộc dạng Sequence Labeling (Gán nhãn
chuỗi), nơi mỗi từ trong câu đầu vào cần được gán một nhãn từ loại tương ứng.

Các bước triển khai cụ thể bao gồm:

- Xử lý dữ liệu CoNLL-U: Đọc và trích xuất thông tin (từ, nhãn UPOS) từ bộ dữ liệu chuẩn Universal Dependencies (
  UD_English-EWT).
- Xây dựng từ điển: Tạo ánh xạ từ $\to$ index (`word_to_ix`) và nhãn $\to$ index (`tag_to_ix`), xử lý các token đặc biệt
  như `<PAD>` và `<UNK>`.
- Pipeline dữ liệu: Xây dựng class `POSDataset` và `DataLoader` tùy chỉnh. Đặc biệt là hàm `collate_fn` để xử lý các câu
  có độ dài không đồng nhất (padding) trong cùng một batch.
- Xây dựng mô hình: Cài đặt mô hình `SimpleRNNForTokenClassification` gồm 3 lớp chính: Embedding, Vanilla RNN, và
  Linear (Fully Connected).
- Huấn luyện và Đánh giá: Sử dụng hàm loss `CrossEntropyLoss` (bỏ qua phần padding) và tính độ chính xác dựa trên các
  token thực tế.

## 2. File/notebook chính đã được triển khai

File: `test/lab5_rnn_for_pos_tagging.ipynb`. Notebook này chứa toàn bộ mã nguồn thực hiện các yêu cầu của Task 1 đến
Task 5
trong tài liệu hướng dẫn:

- Tải và tiền xử lý (Task 1): Cell 2, 3 (Hàm `load_conllu`, `build_vocab`).
- Dataset & DataLoader (Task 2): Cell 4, 5 (Class `POSDataset` và hàm `collate_fn` sử dụng `pad_sequence`).
- Mô hình RNN (Task 3): Cell 6 (Class `SimpleRNNForTokenClassification`).
- Huấn luyện (Task 4): Cell 7, 9 (Cấu hình tham số, vòng lặp training, tính loss).
- Đánh giá (Task 5): Cell 8, 10 (Hàm `evaluate`, `calculate_accuracy` và `predict_sentence`).

## 3. Kết quả

### 3.1 Thông số cấu hình và dữ liệu:

- Số lượng câu huấn luyện (Train): 12,544 câu.
- Kích thước từ điển (Vocab): 19,675 từ.
- Số lượng nhãn (Tags): 18 nhãn UPOS.
- Tham số mô hình:
    - Embedding dim: 100
    - Hidden dim: 128
    - Learning rate: 0.001
    - Optimizer: Adam
    - Số Epoch: 10

### 3.2 Kết quả định lượng (Huấn luyện)

Sau 10 epoch, mô hình đã hội tụ tốt với kết quả như sau:

| Epoch | Train loss | Train Acc | Dev Acc | Training time | 
|-------|------------|-----------|---------|---------------|
| 01    | 1.125	     | 0.658     | 	0.747  | 	~1.37s       |
| 05    | 0.303	     | 0.903     | 	0.848  | 	~1.06s       |
| 10    | 0.128	     | 0.960     | 	0.860  | 	~1.14s       |

Nhận xét: Độ chính xác trên tập Dev đạt 86%, một kết quả khả quan cho kiến trúc RNN đơn giản không sử dụng các kỹ thuật
phức tạp như Bidirectional LSTM hay CRF.

### 3.3 Kết quả định tính (Dự đoán thực tế)

Thử nghiệm dự đoán trên câu mới: "i love natural language processing"

- Input: i, love, natural, language, processing
- Output (Predicted Tags): `['PRON', 'VERB', 'ADJ', 'NOUN', 'NOUN']`

Mô hình đã nhận diện đúng cấu trúc ngữ pháp cơ bản của câu, phân biệt được tính từ bổ nghĩa cho danh từ ghép phía sau.

## 4. Phân tích và giải thích kết quả

- Hiệu quả của việc xử lý Padding: Một thách thức lớn trong bài toán này là các câu có độ dài khác nhau. Việc cài đặt
  `ignore_index=PAD_IDX` trong hàm `CrossEntropyLoss` và loại bỏ các token padding khi tính Accuracy (trong hàm
  `calculate_accuracy`) là yếu tố quyết định giúp mô hình học đúng các đặc trưng của từ thay vì học các giá trị đệm vô
  nghĩa.
- Hiện tượng Overfitting: Tại Epoch 10, Train Acc đạt 0.960 trong khi Dev Acc đạt 0.860. Có sự chênh lệch khoảng 10%,
  cho thấy mô hình bắt đầu có dấu hiệu overfitting nhẹ trên tập train. Tuy nhiên, Loss trên tập train vẫn giảm đều và
  Dev Acc vẫn tăng nhẹ qua các epoch, chứng tỏ mô hình vẫn đang học hiệu quả các quy luật chung.
- Hạn chế của Simple RNN: Mặc dù đạt kết quả tốt (86%), nhưng kiến trúc Simple RNN (Vanilla) thường gặp vấn đề Vanishing
  Gradient với các câu quá dài. Trong các bài toán phức tạp hơn, việc thay thế bằng LSTM hoặc GRU và sử dụng mô hình 2
  chiều (Bi-directional) sẽ giúp nắm bắt ngữ cảnh từ cả hai phía tốt hơn.

## 5. Các công cụ

Thư viện chính:

- `torch.nn.RNN`: Module cốt lõi để xử lý chuỗi.
- `torch.nn.Embedding`: Tạo không gian vector biểu diễn từ dày đặc (dense).
- `torch.nn.utils.rnn.pad_sequence`: Công cụ đệm chuỗi để tạo batch.
