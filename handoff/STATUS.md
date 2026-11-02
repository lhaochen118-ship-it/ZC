# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-02 15:25 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
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

## Đã xong trong khung chat này (sau restart 12:53 UTC)

- `question_cleaning/` + `src/qnorm.py`: 3000 câu train đã làm sạch (khôi phục dấu 99.3 % từ / 93.6 % câu, LOBO).
- Chunk stealing (`src/pipeline.py`), đã qua test.
- Kiểm tra ánh xạ (`canon.plausibility`): 0 dương tính giả trên 36 tham chiếu; bắt được ánh xạ sai của 107/126/127/110.
- Ingest: cột ngày dạng số serial Excel → datetime (chỉ đổi biz_107 `ngay_ban`).
- Hậu kiểm câu trả lời (`agent.answer_flags`, 4/2930 cờ giả trên đáp án chuẩn) + **escalation** (`--escalate N`: thêm N
  mẫu thinking cho câu conf < 0.75 / unanswerable / bị cờ; thêm chương trình trên bảng thô nếu ánh xạ bỏ sót bảng).
- Vòng "đầy đủ" cho ánh xạ (`--complete-map`): hỏi lại một lần khi ánh xạ bỏ sót bảng thô.
- Câu đã làm sạch + kỳ hiển thị cạnh câu gốc trong prompt (`--qclean`).
- Routine tự tiếp tục: `trig_01U26KaZP68Dmg8JL2RGcbVS` (mỗi 2 giờ, vào phiên này).

## Đang chạy (15:25 UTC)

- **v4** (`runs/v4`, launched 14:59, đã qua preflight): phuongnehe zc-v4-1 (train), zc-v4-2 (public); hai1108 zc-v4-3
  (train escalate 0), zc-v4-4 (train think-all); mucnokieu zc-v4-5 (qnorm_llm: LLM thêm dấu 3000 câu + đo độ chính xác).
  Cấu hình: pipeline + think-map + kc 6 + qw 4 + escalate 4 + complete-map + qclean. Watcher `watch_done.py` chạy nền.
  v4 chạy với bản qnorm CŨ (trước khi sửa Max / SP / Lê) — ảnh hưởng nhỏ, chỉ phần gợi ý câu sạch.

## Làm sạch câu hỏi (15:25 UTC) — gần xong

- Đã sửa (luật chung + test): bảo vệ tên trong dữ liệu (bigram tên) và chữ dính số khỏi Telex / viết tắt / ngữ cảnh;
  entity linking chỉ đổi DẤU (giữ hoa thường của người dùng; từ người dùng gõ dấu là chuẩn; khớp cả ≥ 3 từ liên tiếp
  của tên; tên không dấu chỉ cho mượn từ không dấu có thật như 'lon'); `e` đầu câu → ê; `35.000d` → đ; `nhieu` không có
  từ hỏi → bao nhiêu; `có … hong` → không.
- LOBO (scratchpad `eval_qnorm_sig.py`, so khớp bỏ qua vị trí dấu thanh): **98.1 % câu / 99.79 % từ** (trước 94.5 %).
- Còn lại: ghép kết quả LLM (zc-v4-5) bằng `make.py --llm`, duyệt từng câu luật ≠ LLM, cập nhật README, commit.
## Việc tiếp theo (thứ tự)

0. User (02/11 15:00): làm sạch 3000 câu tới 100 % rồi mới phát triển tiếp mô hình; v4 vẫn chạy.
   → luật đã xong (commit 'question cleaning: name protection …'); còn bước ghép LLM khi zc-v4-5 xong.

1. `tests/run_all.sh` PASS → commit → preflight → chạy **v4** (pipeline + think-map + kc 6 + escalate 4 + complete-map +
   qclean) train + public, và **ablation** escalate 0 (acc khác) để đo phần đóng góp.
2. Chấm v4 (report.py), phân tích lỗi còn lại theo DN / loại câu; cập nhật EXPERIMENT_LOG.
3. Nếu tốt: tạo file nộp public từ run public v4 (`results/pub/submission.jsonl`) cho user.
4. Tiếp tục: chọn ánh xạ theo đồng thuận đáp án giữa nhiều ứng viên (replay), kiểm tra bảng phụ (kho, lô, nhập), Docker 5090.
