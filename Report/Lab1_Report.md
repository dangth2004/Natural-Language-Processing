# Báo cáo công việc đã hoàn thành của Lab 1

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
