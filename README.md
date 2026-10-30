# ZC — Zalo AI Challenge 2026, track DataMate

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng (CSV, Excel, SQLite) của hộ kinh doanh nhỏ, biết từ chối khi dữ liệu
không đủ. Điểm: Penalized Accuracy (+1 đúng, 0 abstain, −1 sai). Đề đầy đủ: `docs/00_TASK.md`.

Branch làm việc: **`datamate`**. Branch này là bản sao của `hmm-navigation-2` (repo `lhaochen118-ship-it/hmm-develop`),
**giữ nguyên toàn bộ lịch sử commit**. Hai commit đầu trên nền đó:

1. Bỏ nội dung nghiên cứu VPR/HMM (code `analysis/`, kết quả, paper, literature, handoff, script dump dataset riêng).
2. Thêm đề bài, kế hoạch, quy trình và công cụ cho DataMate.

File cũ vẫn khôi phục được: `git checkout de4360e -- <path>` (`de4360e` là commit cuối của `hmm-navigation-2`).

## Đọc theo thứ tự

| # | File | Để biết |
|---|---|---|
| 1 | File này | Repo có gì, trạng thái, việc tiếp theo |
| 2 | `docs/00_TASK.md` | Đề bài: input/output, 9 kiểu đáp án và luật khớp, công thức PA, luật kỹ thuật (≤ 15B, offline, 6 giờ trên 2 × RTX 5090, tất định ≤ 0.03), lịch nộp |
| 3 | `docs/01_PLAN.md` | Kế hoạch A–Z: tổng quan paper, kiến trúc đề xuất, fine-tune, vòng thí nghiệm E0–E9, lịch đến 16/11, rủi ro |
| 4 | `docs/03_WORKFLOW.md` | Nguyên tắc làm việc, vòng lặp thí nghiệm, tối ưu thời gian chạy và token, git |
| 5 | `docs/02_KAGGLE_OPS.md` | Tài khoản Kaggle, cách chạy fanout, dataset public cần tạo, bài học vận hành Kaggle/container |
| 6 | `EXPERIMENT_LOG.md` | Nhật ký thí nghiệm (mỗi vòng một entry) |
| — | `docs/RESEARCH_WORKFLOW_original.pdf` | Prompt quy trình nghiên cứu gốc của user (Generate → Critique → Verify → Improve), nền của `03_WORKFLOW.md` |

## Bản đồ repo

| Path | Nội dung |
|---|---|
| `docs/` | Tài liệu đề bài, kế hoạch, vận hành, quy trình; prompt quy trình gốc (PDF) |
| `src/` | Code chạy trên Kaggle (upload thành dataset `<acc>/zc-code`). Hiện có `score.py`: bộ chấm cục bộ theo luật BTC (đã self-test). `ZC_CODE_MARKER` để runner tìm thư mục code |
| `automation/` | `fanout.py` (chia job ra mọi acc/slot GPU, failover), `make_kernel.py` (dataset code + kernel), `run_kernel.py` (push/poll/kéo output, tự huỷ kernel kẹt hàng đợi), `make_meta.py` (tier T4/L4/RTX/CPU), `publish_dataset.py` (dataset public của `nhatlong1103`), `watch_done.py` (báo ngay khi một kernel xong), `resume.py` + `reattach.py` (phục hồi sau khi container khởi động lại), `md2pdf.py` |
| `data/` | Chỗ đặt `data.txt` và 2 file zip của BTC (`data/raw/` cho bản giải nén, không commit) |
| `results/` | Kết quả kéo về từ Kaggle |

## Trạng thái (2026-10-30)

- Đã đọc toàn bộ đề trên portal, trừ tab Data (chờ `data/data.txt`).
- Đã có kế hoạch, bộ chấm cục bộ, và hạ tầng chạy Kaggle tổng quát hoá cho DataMate.
- **Đang chờ user:**
  1. Đẩy `data.txt` và 2 file zip vào `data/` của branch này (repo ZC hiện chưa có các file đó).
  2. Gửi token Kaggle (thứ tự như bảng trong `docs/02_KAGGLE_OPS.md` §2). Container mới không có token.
  3. Cho biết dữ liệu BTC có được đăng **public** trên Kaggle không (điều khoản cuộc thi).

## Việc tiếp theo

1. **E0:** đọc `data.txt` và cấu trúc zip (chỉ vài dòng), điền `docs/00_TASK.md` §3, sửa `src/score.py::ref_forms`
   theo định dạng đáp án thật, tách train theo doanh nghiệp thành dev/holdout.
2. Đăng dữ liệu, wheels vLLM, trọng số mô hình thành dataset dùng chung (`docs/02_KAGGLE_OPS.md` §4).
3. **E1** baseline với 3–4 mô hình ≤ 15B, rồi lần lượt E2–E9 theo `docs/01_PLAN.md` §5, ghi `EXPERIMENT_LOG.md`.

## Môi trường tối thiểu

```bash
pip install kaggle markdown xhtml2pdf
```

Token: `~/.kaggle/tok_<username>` cho từng acc, `~/.kaggle/access_token` của `nhatlong1103`. Không commit token, không ghi
vào `~/.bashrc`.
