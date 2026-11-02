# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-02 19:35 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
(kể cả sau khi hết / reset token): đọc `CLAUDE.md` → file này → làm mục "Việc tiếp theo" theo thứ tự.*

## Kết quả mới nhất (Qwen3.5-9B FP8, train 3000 câu leave-one-out, 4×L4) — EXPERIMENT_LOG entry 4

| Run (scratch `runs/`) | dev | holdout | all | wall |
|---|---|---|---|---|
| t1q35-1 `--think-map` (cũ) | 0.926 | 0.761 | 0.870 | 4065 s |
| **v4b zc-v4b-1** escalate 4 | **0.976** | 0.913 | **0.955** | 6046 s |
| v4b zc-v4b-3 escalate 0 | 0.955 | 0.918 | 0.943 | 4774 s |

Còn yếu: biz_123 0.57, biz_111 0.86, biz_117 0.87, biz_121 0.88, biz_122 0.90; escalation làm hỏng kiểu `table`.
Public v4b (zc-v4b-2, 7599 s) đã có: ứng viên nộp `runs/v4b/zc-v4b-2_out/results/pub/`.

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

## Đang chạy (19:35 UTC)

- **v5** (`runs/v5`, launched 18:49; commit 2ea939c, compile-tokens 6144): zc-v5-1 train esc4, -2 public esc4,
  -3 train esc4 lặp lại (đo tất định), -4 train esc0, -5 public lặp lại (đo độ trùng đáp án). Xong khoảng 20:30–21:00.
- **v6-train** (`runs/v6`, launched 19:23 trên mucnokieu, commit 8793c4f: rule reader đọc câu đã làm sạch + sửa mẫu câu +
  toolkit).
- Sau v6 (chưa chạy GPU): f461753 (cảnh báo nhập không có mặt hàng, sổ kho coi như snapshot, quy đổi rỗng), c65757a
  (CANON_DOC đủ bảng, đổi tên cột alias: delivery_batch → trip_no sửa 10 câu biz_117; cảnh báo nợ biz_111).
  docker/predict.sh: cấu hình tốt nhất (kc 6, qw 4, think-map, escalate 4, compile 6144). → gom vào v7.
- Watcher `watch_done.py` nền; `active_runs.txt` = v5, v6.
- Oracle train (eval_lib --clean): 2754 đúng / 3 sai / 243 chưa đọc (sáng nay 2666 / 36 / 298). EXPERIMENT_LOG entry 5.
- Việc sau khi v5 / v6 xong: chấm, so v4b (0.955), đo |ΔPA| giữa 2 lần train giống nhau, độ trùng 2 lần public; nếu
  tốt → chạy public bản v6 làm ứng viên nộp; còn lại: chọn ánh xạ cho DN mới (dev_012 sửa xe, dev_014 điện thoại).

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
