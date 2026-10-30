# Experiment log — DataMate

Mỗi entry: ngày, thí nghiệm (mã E# trong `docs/01_PLAN.md`), giả thuyết, cấu hình (mô hình, prompt, K, ngưỡng), dữ
liệu (split, số câu), kết quả (PA tổng và theo type / loại câu, % abstain, thời gian/câu), phân tích lỗi, quyết định.

| Entry | Ngày | Thí nghiệm | PA (dev) | PA (holdout) | PA public LB | s/câu | Quyết định |
|---|---|---|---|---|---|---|---|

## Entry 0 — 2026-10-30: khởi tạo

- Đọc đề trên portal (Description, Evaluation, Rule, Submission Guidance, Timeline) → `docs/00_TASK.md`.
- Viết `src/score.py` theo luật khớp của BTC; self-test các ví dụ trong bảng Evaluation đều qua.
- Chưa có dữ liệu (`data/` trống) và chưa có token Kaggle trong container → E0/E1 chờ.

## Entry 1 — 2026-10-30: dữ liệu, bộ làm sạch tổng quát, agent v0

- Dữ liệu tải từ `nhatlong1103/zc-2026`; tóm tắt ở `docs/00_TASK.md` §3.
- `src/ingest.py`: làm sạch tự động, chỉ dựa trên quy ước chung (không theo tên DN/cột cụ thể). Audit "cột chỉ gõ kiểu
  được một phần" giảm từ 78 → 0 trên train; public còn 2 cột IMEI (đúng là chữ). Làm sạch toàn bộ: train 57 s, public
  40 s (4 nhân).
- Kiểm chứng: 3 chương trình viết tay trên bảng đã làm sạch (`biz_101`: số đơn hôm qua, doanh thu hôm nay, ngày doanh
  thu cao nhất tháng) ra **đúng tuyệt đối** đáp án BTC.
- Kaggle: kernel T4 có internet; `pip install vllm` (0.32, torch 2.13 + CUDA 13) mất ~6 phút và xung đột torchaudio →
  gỡ torchaudio; L4 grant không cho internet; `yuan1103` hết quota GPU tuần.
- E1 đang chạy: Qwen3-14B-AWQ, không thinking, K = 4, 2 vòng sửa lỗi, DP trên 2 T4, 300 câu train (10/DN).
