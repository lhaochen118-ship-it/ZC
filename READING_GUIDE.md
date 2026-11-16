# READING_GUIDE — cách đọc repository và việc tiếp theo (cập nhật 16/11/2026 10:15 UTC)

Cho Agent nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm. **Hạn nộp 16/11 23:59 GMT+7 (16:59 UTC).**

## 0. Mười lăm phút đầu

1. **Xin user token Kaggle** (container mới mất `~/.kaggle/tok_*`; 29 acc, danh sách trong `handoff/environment_summary.pdf` §6–§7);
   lưu `~/.kaggle/tok_<username>` (chmod 600), không ghi vào repo.
2. `pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf pandas openpyxl pypdf` (CPU torch trước nếu cần: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
3. Dữ liệu: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_nhatlong1103) kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip`;
   đáp án public (chỉ để chấm): `cp "score 1.jsonl" data/raw/public_gt.jsonl`.
4. `git lfs pull` nếu cần adapter cục bộ (`Process/adapters/`: WL1, WJR1, MoE `moe_E88/{MS2,ML1,MJ2}`).
5. Đọc `handoff/STATUS.md` rồi **PDF 3** (`handoff/03_HANDOFF.pdf`) §3 (việc đang chạy) và §5 (việc tiếp theo).
6. Tiến trình nền đã mất khi đổi container: `python3 automation/resume.py Process/runs/p5` (gắn lại theo dõi kernel), rồi chạy các dòng
   trong `Process/kaggle/p5_chains.txt` (chuỗi gộp → đánh giá, resume-safe) — chỉ khi kernel tương ứng chưa xong.

## 1. Bắt buộc

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | quyết định user (mới nhất 16/11 09:00: **nộp framework 0.952** + submission của nó); GPU / quota / 12 h; cổng kiểm tra |
| 2 | `handoff/03_HANDOFF.pdf` | mục tiêu, quyết định, tiến độ, vị trí, việc tiếp theo |
| 3 | `handoff/01_SCIENTIFIC_FINDINGS.pdf` §10 | kết quả P5 (zero-shot, MoE, ghép component) và bằng chứng |
| 4 | `handoff/02_ENGINEERING_OPS_LESSONS.pdf` §11 | lỗi đã gặp + quy tắc tránh (template suy nghĩ, STALE CODE, đầy đĩa, khởi động lại worker…) |
| 5 | `Docker_submission/README.md`, `0.994_Docker_submission/README.md` | build / chạy thử / xuất image nộp |

## 2. Khi cần chi tiết

| File | Nội dung |
|---|---|
| `docs/19_PLAN_P5_ZS.md` | nhật ký P5 theo thời gian (mọi lượt chạy, điểm, quyết định) |
| `docs/21_COMBO_PLAN.md` | kế hoạch tổ hợp trên framework 0.952 (vai trò S / L / J) |
| `handoff/research_workflow.pdf` §10 | quy trình thí nghiệm "chỉ thay đúng thành phần" |
| `handoff/environment_summary.pdf` §7 | acc, mô hình, dataset, bẫy môi trường |
| `Process/README.md` | những gì đã lưu để không phải dựng lại |
| `Process/scores_p5zs.csv`, `Process/tools/p5/jscore.py` | bảng điểm, chấm ghép cặp |

## 3. Việc tiếp theo (tóm tắt — chi tiết PDF 3 §5)

1. Thu kết quả các lượt đang chạy (α 0.5 / 0.35, MS3, MS2C, MS3C, pver) và ghi lại (chỉ là findings; bản nộp không đổi).
2. User build + chạy thử Docker (`Docker_submission/`), xuất image, nộp trước hạn.
3. Sau nộp: xoá routine `trig_013dELF9kwndZnzmL5KAZ6GS`.
