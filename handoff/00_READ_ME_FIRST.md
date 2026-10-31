# 00 — READ ME FIRST (khung chat mới, DataMate)

*Repo `lhaochen118-ship-it/ZC`, branch `datamate`. File này chỉ cách đọc repo và việc tiếp theo; chi tiết ở các file
được trỏ tới. Mỗi file `handoff/0x_*.md` có bản PDF cùng tên.*

## Thứ tự đọc

| # | File | Để biết |
|---|---|---|
| 0 | `handoff/04_TOOLKIT_V2.md` | **Pipeline v2 (31/10) — trạng thái hiện tại, kết quả, cách chạy, việc tiếp theo** |
| 1 | File này | Repo có gì, việc tiếp theo (bản 30/10, trước v2) |
| 2 | `handoff/03_HANDOFF.md` | Bối cảnh, quyết định user đã chốt, trạng thái, **các bước tiếp theo chi tiết** |
| 3 | `handoff/RUNNING_KERNELS.md` | Kernel còn chạy lúc bàn giao, cách kéo kết quả |
| 4 | `handoff/01_SCIENTIFIC_FINDINGS.md` | Đóng góp đã kiểm chứng, kết quả tích cực / tiêu cực, số liệu, hạn chế |
| 5 | `handoff/02_ENGINEERING_OPS_NOTES.md` | Kaggle, môi trường, lỗi đã gặp và cách sửa, bài học |
| 6 | `docs/00_TASK.md` | Đề bài đầy đủ: kiểu đáp án, luật khớp, PA, luật kỹ thuật, dữ liệu, lịch |
| 7 | `docs/03_WORKFLOW.md`, `docs/02_KAGGLE_OPS.md` | Quy trình, nguyên tắc dùng Kaggle của user; tài khoản, assets, lệnh |
| — | `docs/01_PLAN.md`, `EXPERIMENT_LOG.md`, `docker/SETUP_DOCKER.md` | Kế hoạch ban đầu và tài liệu tham khảo, nhật ký thí nghiệm, hướng dẫn Docker cho user |

## Việc tiếp theo (tóm tắt; chi tiết ở `03_HANDOFF.md` §3)

1. **Khởi động:**
   - Xin token Kaggle từ user, lưu vào `~/.kaggle/tok_<acc>`.
   - Cài gói, tải dữ liệu `nhatlong1103/zc-2026`, chạy 2 test không GPU.
   - Chạy `automation/pull_kernels.py` rồi phân tích các run bằng `src/analyze.py`.
2. **Khắc phục hết lỗi đã mắc:**
   - helper ngày an toàn;
   - chặn đọc file;
   - `.astype(str)` trước `.str`;
   - compile: A/B rồi giữ hoặc bỏ;
   - chọn τ trên DN dev;
   - xác nhận thông lượng.
3. **Chạy lại trên GPU đã cấp:**
   - `nguyenhoa99` (2 × 4 L4): toàn bộ train, chia dev / holdout.
   - T4 của `mxuanquy`, `anderson1202`: các biến thể A/B.
   - **Luôn thử nhỏ (`--subset 8`) trước khi chạy full.**
4. Chỉ tạo file nộp public khi PA holdout tốt.

## Lệnh hay dùng

```bash
python3 automation/fanout.py <runs>/<run> zc-<run> "run_kaggle.py train OUT --model Qwen/Qwen3-14B-AWQ --compile --k 4 --kc 4 --dp 4 --cache /tmp/zc_cache --max-len 24576 --subset 8" 1 --accounts nguyenhoa99 --l4 --ks nhatlong1103/zc-assets
```

```bash
python3 automation/watch_done.py <runs> --max 110
```

```bash
python3 src/analyze.py <out>/results/logs.jsonl data/raw/train/train/answers.jsonl --show 10
```

- `watch_done.py` phải chạy bằng `run_in_background`, và ghi tên run vào `<runs>/active_runs.txt` trước.
- `<runs>` đặt trong scratchpad, không commit.
