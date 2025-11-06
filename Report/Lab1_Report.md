# Báo cáo công việc đã hoàn thành của Lab 1 + 2

## Mô tả công việc

- Đã hoàn thành Task 1: Simple Tokenizer, tuy nhiên chưa xử lý trường hợp đặc biệt.
- Đã hoàn thành Task 2: Regex-based Tokenizer.
- Đã hoàn thành Task 3: Tokenization with UD_English-EWT Dataset.

## Kết quả chạy code

<img width="1847" height="717" alt="image" src="https://github.com/user-attachments/assets/e508cc5f-be69-4490-be7e-ac7a3f617418" />


## Phân tích kết quả

- Nếu tokenize văn bản bằng cách đơn giản là sử dụng dấu trắng thì không tách được những dấu câu đi kèm với từ. Ví dụ
  như "Hello, world!" sẽ được tokenize thành ['hello,', 'world!'].
- Tokenize văn bản bằng biểu thức chính quy giải quyết được nhược điểm trên. Ví dụ như trường hợp "Hello, world!" sẽ
  được tokenize thành ['hello', ',', 'world', '!'].
- Tuy nhiên em còn chưa giải quyết được dùng tokenize đơn giản mà xử lý được trường hợp đặc biệt như trên.

# Báo cáo công việc đã hoàn thành của Lab 2

## Mô tả công việc

- Đã hoàn thành Task 1: Vectorizer Interface.
- Đã hoàn thành Task 2: CountVectorizer Implementation

## Kết quả chạy code

<img width="1828" height="203" alt="image" src="https://github.com/user-attachments/assets/9c030c05-6270-4ca1-a850-1553f9e1d86c" />


## Phân tích kết quả

- Chương trình đã đưa ra 1 Vocabulary chứa các token và thứ tự của nó.
- Khi vectorize bằng CountVectorize, văn bản sẽ được mã hóa bằng 1 vector có độ dài bằng với độ dài từ điển (trong ví dụ
  của bài thực hành là 10).
- Nếu trong văn bản xuất hiện token nào thì vị trí tương ứng trong vector biểu diễn của văn bản sẽ có giá trị là 1,
  ngược lại là 0.