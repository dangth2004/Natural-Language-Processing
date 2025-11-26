# Lab 6: Giới thiệu về Transformers

## 1. Mô tả công việc đã triển khai

Lab 6 tập trung vào việc làm quen với kiến trúc Transformer và thư viện `transformers` của Hugging Face thông qua việc
sử
dụng các mô hình tiền huấn luyện (pretrained models). Mục tiêu không phải là huấn luyện mô hình từ đầu mà là hiểu cách
inference và trích xuất đặc trưng.

Các nội dung thực hiện bao gồm 3 bài tập chính:

- Masked Language Modeling (MLM): Sử dụng mô hình Encoder-only (BERT) để khôi phục từ bị che khuất trong câu dựa trên
  ngữ cảnh hai chiều.
- Text Generation (Next Token Prediction): Sử dụng mô hình Decoder-only (GPT-2) để sinh văn bản tiếp nối từ một câu gợi
  ý (prompt).
- Sentence Representation: Sử dụng BERT để trích xuất vector đặc trưng (embedding) của một câu thông qua phương pháp
  Mean Pooling, thao tác trực tiếp trên các hidden states thay vì dùng pipeline có sẵn.

## 2. File/notebook chính đã được triển khai

File: `test/lab6_intro_transformers.ipynb`. Notebook chứa mã nguồn thực hiện các yêu cầu của Lab:

- Bài 1 (Masked Language Modeling): Cell 2, 3. Sử dụng `pipeline("fill-mask")` với mô hình `bert-base-uncased`.
- Bài 2 (Text Generation): Cell 4, 5. Sử dụng `pipeline("text-generation")` với mô hình `gpt2`.
- Bài 3 (Sentence Representation): Cell 6, 7. Tải `AutoModel`, `AutoTokenizer` và thực hiện tính toán vector trung bình
  có
  sử dụng `attention_mask`.

## 3. Kết quả

Bài 1:

- Input: "Hanoi is the [MASK] of Vietnam."
- Kết quả dự đoán (Top 1):
    - Token: `captital`
    - Độ tin cậy: 0.9991
    - Câu hoàn chỉnh: "hanoi is the capital of vietnam."

Bài 2:

- Prompt: "The best thing about learning NLP is"
- Văn bản sinh ra: "...that you learn to learn something, and you learn it through hard work and hard work. It's not
  like you learn all at once, but you learn at what is most important..."

Bài 3: Sentence Representation

- Input: "This is a sample sentence"
- Kích thước vector đầu ra: `torch.Size([1, 768])`
- 5 giá trị đầu tiên của vector: `[-0.2424, -0.3832, -0.0138, -0.2991, -0.2145]`

## 4. Phân tích và giải thích kết quả

Bài 1:

- Đánh giá: Mô hình dự đoán chính xác từ "capital" với độ tin cậy rất cao (~99.9%).
- Giải thích: Mô hình Encoder-only như BERT phù hợp cho tác vụ này vì chúng có khả năng nhìn hai chiều (bidirectional).
  Để điền đúng từ vào chỗ trống, mô hình cần hiểu ngữ cảnh từ cả phía trước ("Hanoi is the...") và phía sau ("...of
  Vietnam"). Kiến trúc Encoder cho phép mô hình tổng hợp thông tin toàn cục của câu tại mỗi vị trí từ.

Bài 2:

- Đánh giá: Văn bản sinh ra ngữ pháp đúng và liên quan đến chủ đề học tập, mặc dù nội dung hơi lặp lại ("hard work and
  hard work").
- Giải thích: Mô hình Decoder-only như GPT phù hợp cho tác vụ sinh văn bản vì chúng hoạt động theo cơ chế tự hồi quy (
  autoregressive) và chỉ nhìn một chiều (unidirectional). Khi sinh từ tiếp theo, mô hình chỉ được phép biết các từ đã
  xuất hiện trước đó (quá khứ) mà không được biết tương lai, điều này mô phỏng đúng quá trình viết hoặc nói tự nhiên của
  con người.

Bài 3:

- Kích thước vector: Kích thước (1, 768) tương ứng với tham số hidden_size của mô hình bert-base. Mỗi token trong BERT
  được biểu diễn bằng một vector 768 chiều.
- Vai trò của Attention Mask trong Mean Pooling:
    - Khi xử lý theo batch, các câu có độ dài khác nhau sẽ được đệm (padding) để bằng độ dài câu dài nhất.
    - Khi tính trung bình cộng (Mean Pooling) các vector token để ra vector câu, chúng ta bắt buộc phải sử dụng
      attention_mask để loại bỏ các token đệm (padding tokens).
    - Nếu không có mask, các vector của token đệm (thường là vector 0 hoặc vector rác) sẽ bị cộng vào tổng và chia trung
      bình, làm sai lệch ý nghĩa thực sự của câu. Cụ thể trong code, mask_expanded giúp triệt tiêu giá trị của padding
      token trước khi tính tổng.

## 5. Các công cụ sử dụng

- Thư viện Transformer:
    - `pipeline`: Công cụ high-level để thực thi nhanh các tác vụ NLP.
    - `AutoTokenizer`, `AutoModel`: Các lớp để tải kiến trúc và trọng số mô hình linh hoạt.
- Mô hình pretrained: BERT (Google), GPT-2 (OpenAI).