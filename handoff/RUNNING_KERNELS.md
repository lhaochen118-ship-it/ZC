# Kernel còn chạy lúc bàn giao (2026-10-31 01:10 UTC)

Runner cục bộ của khung chat 2 mất khi sang khung chat mới; kernel vẫn chạy trên Kaggle. Kéo output khi xong:

```bash
python3 automation/pull_kernels.py <scratch>/pulled
```

Cần token `~/.kaggle/tok_<acc>` của từng acc. Kernel chưa xong được báo `not finished`.

| Run | Kernel | GPU | Cấu hình | Dữ liệu | Ghi chú |
|---|---|---|---|---|---|
| p1 | `lelynhan/zc-p1-1` | 4×L4 | cấu hình tốt nhất (handoff 03 §5), shots v2 | **public test 1000 câu** | → file nộp; đẩy 20:57 UTC |
| e10l | `lelynhan/zc-e10l-1` | 4×L4 | tốt nhất + `--think --max-tokens 8192`, shots v3 | train biz_114,116,117,125 | đang **QUEUED** (pool L4 bận); so với e7 |
| e10 | `anderson1202/zc-e10-1` | 2×T4 | như e10l, `--tp 2 --subset 40` | train biz_114,116,117,125 (40 câu) | so với e7 |
| e10b | `mxuanquy/zc-e10b-1` | 2×T4 | như e10 | train biz_121,124,129,130 (40 câu) | so với e8 |
| e3h | `linhmai24/zc-e3h-4` | 4×L4 | `--helpers --compile --compile-scope sales`, không shots | train biz_101,103,112,121,124,127,129 | chạy > 7 giờ (nghi kẹt); chỉ để thêm shots |
| e3b | `thnliu/zc-e3b-2` | 4×L4 | `--helpers`, BBBB | train nửa sau | chạy > 7 giờ (nghi kẹt); chỉ để thêm shots |

**Cần user ngắt tay** (công cụ tự động bị chặn lệnh xoá): `mxuanquy/zc-e1b-1`, `anderson1202/zc-e1c-1` — baseline cũ của khung
chat 1, code cũ có lỗi, vô ích, chỉ tốn quota T4.

Đã xong và đã phân tích (log trong `results/chat2_runs/`): smoke2, e3h-1/2/3, e3b-1, e3t-1/2, e4-1/2, e5-1/2, e6a, e7, e7b, e8, e9.
