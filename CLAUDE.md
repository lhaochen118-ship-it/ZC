# CLAUDE.md — bộ nhớ dự án (Claude Code tự đọc file này mỗi phiên)

Zalo AI Challenge 2026, track DataMate. Branch làm việc **`claude/datamate-v3-0dny90`** (repo `lhaochen118-ship-it/ZC`).
Đọc theo thứ tự: `README.md` → `docs/ZAIC_REQUIREMENTS.md` (yêu cầu user) → `docs/00_TASK.md` (đề) →
`docs/04_DECISION_V3.md` (kiến trúc v3) → `docs/03_WORKFLOW.md` (quy trình, acc Kaggle) → `docs/05_GPU_OPTIMIZATION.md`
→ `EXPERIMENT_LOG.md`.

## Nguyên tắc user (đã chốt, không hỏi lại)

- Tự động hoá tối đa; chỉ hỏi việc chỉ user làm được (token, nộp portal, accept rules Kaggle trên web).
- Mục tiêu: PA cao nhất trên private test **và nhanh nhất có thể**; tổng tham số mô hình ≤ 15B, offline, ≤ 6 giờ trên
  2×RTX 5090. Tổng quát, không vá theo DN / câu của tập test. Chỉ tạo file nộp khi kết quả tốt.
- Token Kaggle chỉ ở `~/.kaggle/tok_<username>` (600); container mất khi sang khung chat mới → xin lại user.
- Theo dõi kernel bằng watcher nền, không poll; tìm tài liệu: đọc văn bản / paper, không đọc code repo người khác.

## Playbook tốc độ GPU (user yêu cầu ghi nhớ, 2026-11-02) — chi tiết: `docs/05_GPU_OPTIMIZATION.md`

1. **Đo trước khi tối ưu.** Mọi kernel chạy qua `src/run_kaggle.py` → `gpu_monitor.csv` + `gpu_summary.json`
   (util trung bình, tỉ lệ mẫu bận, VRAM đỉnh). Không kết luận "đã tận dụng GPU" khi chưa có số này.
2. **4×L4 = 4 GPU 24 GB độc lập, không NVLink** (không phải 1 GPU 96 GB). Mô hình vừa 1 GPU → **data parallel**
   (`--dp 4`, mỗi GPU một bản); chỉ dùng TP khi không vừa. Lượng tử hoá (AWQ / FP8 `--quant fp8`) để dư VRAM cho KV.
3. **VRAM cho KV cache:** `--gpu-util 0.92`; đọc dung lượng KV thật từ vLLM (dòng `[engine] ... KV cache N tokens`);
   prefix budget = ½ KV (tự động).
4. **Prefix cache:** phần chung (rulebook + README + card) đặt **trước**, phần riêng mỗi request (`# EXAMPLES`,
   `# QUESTION`, `# TASK`) đặt **sau**; gom request theo DN; không trộn nhiều DN trong một đợt.
5. **Không có rào chắn giữa các vòng:** `--pipeline` (`src/pipeline.py`) — request vào vLLM từng cái
   (`add_request`/`step`), chương trình chạy CPU song song khi GPU vẫn sinh, sửa lỗi đưa vào hàng đợi ngay.
6. **Cân bằng tải động:** các tiến trình GPU tự nhận DN tiếp theo (claim file nguyên tử, DN nhiều câu trước); không chia
   tĩnh.
7. **Đầu ra ngắn là đòn bẩy lớn nhất:** việc của LLM phải hẹp (chương trình toolkit ~100 token, ánh xạ ~1.5k token);
   ngữ nghĩa nằm trong code đã kiểm chứng (`src/dm.py`).
8. Thử thêm khi cần: `--max-num-seqs`, `--max-batched-tokens`, `--spec-mtp N` (MTP của Qwen3.5); chỉ giữ khi đo nhanh hơn
   mà PA không giảm.
9. **Kaggle:** dồn mọi luồng lên một acc (2 phiên GPU + 5 phiên CPU) rồi mới sang acc khác; L4 cần acc đã accept rules
   `gemma-4-developer-agent`; probe acc bằng kernel 1 dòng; QUEUED > 8 phút (L4) → huỷ, đẩy lại (tối đa 5 lần).
10. **Bẫy công cụ:** watcher `automation/watch_done.py` chạy `run_in_background` với `timeout` 7200000 ms (mặc định
    30 phút sẽ bị kill); `A && B && nohup C &` đưa cả chuỗi vào subshell (mất biến) → dùng `(…) &`.

## Kiểm tra trước khi push (CPU, phải PASS)

`tests/README.md`: `test_ingest_robust.py`, `test_engine_batching.py`, `test_agent_v2.py`, `test_pipeline.py`,
`eval_lib.py` (oracle mapping train: 2666 đúng / 36 sai / 298 chưa đọc, 02/11).
