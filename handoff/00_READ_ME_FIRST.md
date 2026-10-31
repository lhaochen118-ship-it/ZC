# 00 — READ ME FIRST (khung chat mới, DataMate)

*Repo `lhaochen118-ship-it/ZC`, branch **`claude/peaceful-carson-0dny90`**. Mỗi file `handoff/0x_*.md` có bản PDF cùng tên.*

## Thứ tự đọc

| # | File | Để biết |
|---|---|---|
| 1 | File này | Repo có gì, bắt đầu từ đâu |
| 2 | `handoff/03_HANDOFF.md` | Mục tiêu, quyết định đã chốt, trạng thái, vị trí, **cấu hình tốt nhất**, **việc tiếp theo** |
| 3 | `handoff/RUNNING_KERNELS.md` | Kernel còn chạy lúc bàn giao, cách kéo kết quả |
| 4 | `handoff/01_SCIENTIFIC_FINDINGS.md` | Đóng góp, kết quả (PA train dev +0.47 / holdout +0.44), kết quả âm, lỗi còn lại, hạn chế |
| 5 | `handoff/02_ENGINEERING_OPS_NOTES.md` | Môi trường, Kaggle (acc / quota), code, tổ chức thử nghiệm, lỗi đã sửa, bài học |
| 6 | `docs/00_TASK.md` | Đề bài đầy đủ |
| — | `EXPERIMENT_LOG.md`, `results/chat2_runs/README.md`, `docs/02_KAGGLE_OPS.md`, `docker/SETUP_DOCKER.md` | Nhật ký thí nghiệm, log các run, tham khảo Kaggle, Docker |

## Bắt đầu nhanh

```bash
pip install kaggle markdown xhtml2pdf tokenizers python-calamine
```

```bash
kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip
```

```bash
python3 tests/test_ingest_robust.py
```

```bash
python3 tests/test_agent_local.py data/raw/train/train <scratch>
```

```bash
python3 automation/pull_kernels.py <scratch>/pulled
```

```bash
python3 automation/watch_done.py <runs> --max 110
```

- `watch_done.py` chạy bằng `run_in_background`; ghi tên run vào `<runs>/active_runs.txt` trước.
- `<runs>` đặt trong scratchpad, không commit.
