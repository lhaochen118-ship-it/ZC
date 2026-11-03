# File nộp public test — hệ v19

Code: commit `a246b0b` = v18 + prompt liệt kê các bảng thô mà ánh xạ được chọn bỏ sót (chương trình đọc chúng qua `T`) +
lời dặn: xem các bảng này trước khi trả `unanswerable`.

`submission.jsonl` là kết quả bỏ phiếu 3 lần chạy (seed 0 / 1 / 2). Đã kiểm định dạng: 1000 id, 0 lỗi, **85 câu abstain**,
30 câu unanswerable (v18: 48).

- Train (một lần chạy): **PA 0.986** (dev 0.999, holdout 0.961), cao nhất từ trước.
- Public so với v18:
  - dev_014 (điện thoại): unanswerable 15 → 0 nhưng abstain 20 → 39;
  - dev_003: đồng thuận 3 lần chạy 100 % → 72 %; dev_009: 100 % → 57 %.
  - Các lần chạy chọn ánh xạ khác nhau ở cả v18 lẫn v19 (may rủi ánh xạ), nên chưa biết bản nào tốt hơn trên public.
  - Nộp cả hai sẽ biết hướng "bảng thô trong prompt" có lợi hay không.
