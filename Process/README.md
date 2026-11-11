# Process/ — mọi thứ cần để tiếp tục mà không nằm sẵn trong code

Khung chat mới chạy trong container mới: scratch (log lần chạy, cache, worktree) và `training/data/out/` (dữ liệu huấn luyện,
`.gitignore`) **không còn**. Thư mục này giữ phần chuyển giao được (nén gzip); phần còn lại dựng lại theo cuối file.

| Đường dẫn | Nội dung | Dùng khi |
|---|---|---|
| `scores.csv` | điểm (PA theo đáp án cuối) của **188 lần chạy** trên mọi tập: pub_full, pub_ho, tr_ho, pub_ho400, tr_ho400, tr_dev_all, tr_dev_para, d3sel, pub_k0 + held-out gộp | so ứng viên mới với mốc cũ mà không tải lại gì |
| `runs/<run>/<result>/` | `logs.jsonl.gz` (mọi mẫu: chương trình, đáp án, cờ, phiếu), `submission.jsonl.gz`, `result.json`; `arms.json` / `train_metrics.json` của lần huấn luyện | phát lại phiếu / bộ chọn (`tests/pool_gens.py`), quét tín hiệu sạch cho kiểm tra mới, liệt kê câu sai (`tests/list_wrong.py`), dữ liệu RFT / DPO |
| `train_data/wg/` | dữ liệu huấn luyện kiểu WG: `d1_full` (2,119 câu train → chương trình đã kiểm), `d2_full` (3,818 paraphrase), `d2n_full` (5,419 câu nhiễu), `rft_hard` (710 mẫu đúng của câu khó) — prompt full | huấn luyện nhánh mới (`automation/jobs/wg.jobs`, `wg2r.jobs`: `--train-data wg`) |
| `train_data/m2/`, `m4/`, `w7f_*` | RFT (m2a–d), cặp DPO (871), dữ liệu WF1–4 | tái lập M2 / M4 / WF |
| `kaggle/kernels.csv` | lần chạy → kernel id (`<acc>/zc-<slug>`) → trạng thái | tải lại output: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <dir>` |
| `kaggle/plans/*.json` | plan `launch_many.py` đã dùng (acc, job, est_h, datasets) | mẫu khởi chạy lần sau |
| `kaggle/datasets_by_account.txt` | dataset `zc-*` trên từng acc (zc-para, zc-final-adapter, zc-wg-adapters …) | chọn acc có sẵn dữ liệu / adapter |
| `kaggle/quota_2026-11-11.txt` | quota lúc bàn giao (làm mới 14/11 00:00 UTC) | lập kế hoạch (số thật: `python3 automation/quota.py`) |
| `m0_worktree/` | code M0 đóng băng = commit trong `BASE_COMMIT.txt` + `tracked.patch` + `untracked.tgz` | chạy tiếp hướng M (cần đúng code M0) |
| `history/STATUS_full_until_2026-11-11.md` | bảng tiến độ đầy đủ WP + 16 vòng R + hướng M (số từng lần chạy, kernel, quyết định) | tra lịch sử; trạng thái sống ở `handoff/STATUS.md` |

## Khôi phục nhanh

```bash
S=<scratchpad>                                            # thư mục scratch của phiên mới
# log của một lần chạy -> thư mục như cũ (pool_gens / list_wrong / train_selector đọc logs.jsonl)
mkdir -p $S/runs/p2r15o && gunzip -c Process/runs/p2r15o/p2r15o/logs.jsonl.gz > $S/runs/p2r15o/logs.jsonl
# dữ liệu huấn luyện
mkdir -p training/data/out/wg && for f in Process/train_data/wg/*.gz; do gunzip -c $f > training/data/out/wg/$(basename ${f%.gz}); done
mkdir -p training/data/out/m2 && for f in Process/train_data/m2/*.gz; do gunzip -c $f > training/data/out/m2/$(basename ${f%.gz}); done
# worktree M0 (hướng M)
git worktree add $S/wt_m0 $(cut -d' ' -f1 Process/m0_worktree/BASE_COMMIT.txt) && (cd $S/wt_m0 && git apply /home/user/ZC/Process/m0_worktree/tracked.patch && tar xzf /home/user/ZC/Process/m0_worktree/untracked.tgz)
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
