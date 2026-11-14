# Process/ — mọi thứ cần để tiếp tục mà không nằm sẵn trong code

Khung chat mới chạy trong container mới: scratch (log lần chạy, cache, worktree) và `training/data/out/` (dữ liệu huấn luyện,
`.gitignore`) **không còn**. Thư mục này giữ phần chuyển giao được (nén gzip); phần còn lại dựng lại theo cuối file.

| Đường dẫn | Nội dung | Dùng khi |
|---|---|---|
| `scores.csv` | điểm (PA theo đáp án cuối) của **188 lần chạy** kế hoạch 1–2 (kế hoạch 3–4: `scores_p3.csv`) trên mọi tập: pub_full, pub_ho, tr_ho, pub_ho400, tr_ho400, tr_dev_all, tr_dev_para, d3sel, pub_k0 + held-out gộp | so ứng viên mới với mốc cũ mà không tải lại gì |
| `runs/<run>/<result>/` (lưu bằng `automation/archive_runs.py`; 14/11 chỉ giữ: P (`finala/b`, `p2r17p*`, `p3pbase`), kế hoạch 3 (`rq5_*`, `lrl0*`, `pj`, v3a, `c1a/b`, `q3*`, `plrl*`), kế hoạch 4 (`e0h*`, `g0h*`, `g1*`, `pv*`, `pva3k16` = hệ nộp, huấn luyện v3b / v4 / WR30); log kế hoạch 1–2 đã xoá, còn trong git) | `logs.jsonl.gz` (mọi mẫu: chương trình, đáp án, cờ, phiếu), `submission.jsonl.gz`, `result.json`; `arms.json` / `train_metrics.json` của lần huấn luyện | phát lại phiếu / bộ chọn (`tests/pool_gens.py`), quét tín hiệu sạch cho kiểm tra mới, liệt kê câu sai (`tests/list_wrong.py`), dữ liệu RFT / DPO |
| `train_data/wg/` | dữ liệu huấn luyện kiểu WG: `d1_full` (2,119 câu train → chương trình đã kiểm), `d2_full` (3,818 paraphrase), `d2n_full` (5,419 câu nhiễu), `rft_hard` (710 mẫu đúng của câu khó) — prompt full | huấn luyện nhánh mới (`automation/jobs/wg.jobs`, `wg2r.jobs`: `--train-data wg`) |
| `train_data/p3/` | kế hoạch 3 (13/11): `v3a_mix` (SFT v3a thuần: D1 + RFT + phong cách kỳ / ViLexNorm, nhãn `<spec>`), `v3b_mix` (+ thẻ luật trong prompt, DR, phiên âm), `d1_all` / `d2_all` (30 DN, WR30), đầu vào sinh thẻ (`cases`, `alias_values`, `newterm_in`), thẻ quy nạp ứng viên | `training/data/build_v3b.sh`, `docs/16_PLAN3_LEARNED_RULES.md` |
| `train_data/v4/` | kế hoạch 4 (13/11): `v4_mix` (UD dùng hàm + RFT + DR + replay v3a), `ud`, `rft` — v4 không hơn v3a trên public | tái lập v4 |
| `kaggle/kernels.csv` | lần chạy → kernel id (`<acc>/zc-<slug>`) → trạng thái | tải lại output: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <dir>` |
| `kaggle/plans/*.json` | plan `launch_many.py` đã dùng (acc, job, est_h, datasets) | mẫu khởi chạy lần sau |
| `kaggle/datasets_by_account.txt` | dataset `zc-*` trên từng acc (zc-para, zc-final-adapter, zc-wg-adapters …) | chọn acc có sẵn dữ liệu / adapter |
| `kaggle/quota_2026-11-13.txt` (cũ: `_11-11`) | quota lúc bàn giao 13/11 (làm mới 14/11 00:00 UTC) | lập kế hoạch (số thật: `python3 automation/quota.py`) |
| `tools/` | công cụ phân tích của phiên (sửa `$S` = scratch): `score_sets.py` (PA theo tập), `cmp_pure_p.py` (lỗi thuần so với P theo kỳ / thực thể / hàm), `period_share.py`, quét tín hiệu sạch `scan_*.py`, `publish.sh` (đăng dataset có `info/`) | phân tích lỗi, đăng adapter / checkpoint |

## Khôi phục nhanh

```bash
S=<scratchpad>                                            # thư mục scratch của phiên mới
# log của một lần chạy -> thư mục như cũ (pool_gens / list_wrong / train_selector đọc logs.jsonl)
mkdir -p $S/runs/p2r15o && gunzip -c Process/runs/p2r15o/p2r15o/logs.jsonl.gz > $S/runs/p2r15o/logs.jsonl
# dữ liệu huấn luyện
mkdir -p training/data/out/wg && for f in Process/train_data/wg/*.gz; do gunzip -c $f > training/data/out/wg/$(basename ${f%.gz}); done
mkdir -p training/data/out/v3a && gunzip -c Process/train_data/p3/v3a_mix.jsonl.gz > training/data/out/v3a/v3a_mix.jsonl   # mô hình nộp (v3a)
```

## Không chuyển giao được (dựng lại)

| Thứ | Cách lấy lại |
|---|---|
| Token Kaggle (24 acc) | xin user → `~/.kaggle/tok_<username>` (600); username: `KAGGLE_API_TOKEN=... kaggle config view` |
| Dữ liệu BTC + đáp án public | `kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip`; `data/raw/public_gt.jsonl` do user gửi (`score_1.jsonl`) |
| Cache ingest / bảng chuẩn | `python3 src/ingest.py <split> <S>/cache_{train,pub} 4`; `python3 tests/build_canon.py data/raw/train/train <S>/cache_train <S>/canon` |
| Adapter (LoRA, 320 MB / nhánh) | trên Kaggle: `<acc>/zc-wg-adapters` (WG1–4), `nhatlong1103/zc-final-adapter` (WG1 nộp), output kernel `loinguyentran/zc-wg-1`, `bichthuy93/zc-wg2r-1` (WG5–8) |
| Paraphrase đánh giá | `<acc>/zc-para` (`para_pub.jsonl`, `para_train.jsonl`) |
| Log của lần chạy không có ở `runs/` | `kaggle kernels output` theo `kaggle/kernels.csv` (output kernel còn trên Kaggle) |
| PDF tooling | `pip install markdown xhtml2pdf` |
