# Kernel còn chạy lúc bàn giao (2026-10-31 09:00 UTC)

**Không còn kernel nào chạy.** Theo yêu cầu của user, khung chat 2 dừng sau khi các run cuối kết thúc.

Kết quả cuối:

| Run | Kernel | Kết quả |
|---|---|---|
| p1 | `lelynhan/zc-p1-1` | public test, dừng ở 12 giờ, 764/1000 câu → `results/p1_submission.jsonl` |
| e11 | `lelynhan/zc-e11-1` | thinking trên holdout: PA +0.51 vs +0.25 (e8) |
| e3h-4, e3b-2 | `linhmai24/zc-e3h-4`, `thnliu/zc-e3b-2` | dừng ở 12 giờ, không có output (chỉ để thêm shots) |

Log mọi run (nén): `results/chat2_runs/` (README trong đó). Kéo lại output một kernel đã xong:
`KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <dir>`.
