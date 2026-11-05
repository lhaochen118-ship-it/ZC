# File nộp public test — v30 (rule-first, không cần LLM cho public)

Tạo bằng stage 1 của luồng nộp (`src/rules_first.py`: automap + qnorm + qparse + toolkit), môi trường như image Docker
(pandas 2.3.3, numpy 2.1.3, calamine): 1000 / 1000 câu được luật trả lời, 0 câu chuyển cho mô hình; 143 s trên 2 worker CPU.
Kiểm định dạng (`automation/make_submission.py`): 0 lỗi. Chấm cục bộ nghiêm với đáp án public (chỉ dùng để chấm, code
không đọc): 1000 / 1000.

Lưu ý tổng quát hoá (không phải điểm public): trên câu hỏi viết lại bằng LLM (paraphrase, giữ nghĩa có kiểm tra) luật
một mình đạt PA ≈ 0.51 (public) / 0.67 (train, hoàn toàn chưa xem); phần còn lại chủ yếu là abstain → stage 2 (Qwen3.5-9B)
trả lời trong Docker. Chi tiết: `EXPERIMENT_LOG.md` entry 20–21.
