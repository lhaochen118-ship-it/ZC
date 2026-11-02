# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-02 13:05 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
(kể cả sau khi hết / reset token): đọc `CLAUDE.md` → file này → làm mục "Việc tiếp theo" theo thứ tự.*

## Kết quả mới nhất (Qwen3.5-9B FP8, train 3000 câu leave-one-out, 4×L4)

| Run (scratch `runs/`) | dev | holdout | all | wall |
|---|---|---|---|---|
| v2q35 round-based | 0.897 | 0.644 | 0.810 | 7174 s |
| p3q35 `--pipeline` (đã sửa) | 0.901 | 0.745 | 0.848 | 3234 s |
| t1q35-1 `--think-map` | 0.926 | 0.761 | **0.870** | 4065 s |
| t1q35-2 `--think --think-map` | 0.915 | 0.691 | 0.839 | 8124 s |

Chẩn đoán: DN có ánh xạ đúng đạt 0.90–1.00; sai số gần như toàn bộ do **chọn sai ánh xạ** ở vài DN (107, 110, 117,
123, 124, 126, 127): ánh xạ sai → trả lời sai mà tự tin (PA −1). Thinking cho câu trả lời giúp DN đã ánh xạ đúng.

## Việc đang dở

- `src/pipeline.py`: viết lại work stealing theo **chunk câu hỏi** (16 câu) thay vì theo DN (public: 4 tiến trình xong
  lúc 1174 / 1969 / 2401 / 3144 s). **Chưa test, chưa commit** → phải qua `tests/run_all.sh` + preflight trước khi dùng.

## Việc tiếp theo (thứ tự)

1. Module làm sạch câu hỏi + thư mục `question_cleaning/` (3000 câu train đã làm sạch + README) — user yêu cầu.
2. Test + commit chunk stealing.
3. Chọn ánh xạ tốt hơn: `src/replay.py` trên log các run (DN lỗi), thử bỏ phiếu nhiều ánh xạ / đồng thuận đáp án.
4. Run tiếp: pipeline + think-map (+ think cho câu trả lời) với cách chọn ánh xạ mới; public test → file nộp nếu tốt.
5. Docker 5090: kiểm tra cấu hình cuối, ước lượng thời gian.
