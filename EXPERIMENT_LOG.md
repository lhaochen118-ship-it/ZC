# Experiment log — DataMate

Mỗi entry: ngày, thí nghiệm (mã E# trong `docs/01_PLAN.md`), giả thuyết, cấu hình (mô hình, prompt, K, ngưỡng), dữ
liệu (split, số câu), kết quả (PA tổng và theo type / loại câu, % abstain, thời gian/câu), phân tích lỗi, quyết định.

| Entry | Ngày | Thí nghiệm | PA (dev) | PA (holdout) | PA public LB | s/câu | Quyết định |
|---|---|---|---|---|---|---|---|

## Entry 0 — 2026-10-30: khởi tạo

- Đọc đề trên portal (Description, Evaluation, Rule, Submission Guidance, Timeline) → `docs/00_TASK.md`.
- Viết `src/score.py` theo luật khớp của BTC; self-test các ví dụ trong bảng Evaluation đều qua.
- Chưa có dữ liệu (`data/` trống) và chưa có token Kaggle trong container → E0/E1 chờ.
