# Kernel còn chạy lúc bàn giao (2026-10-31 04:45 UTC)

Runner cục bộ của khung chat 2 mất khi sang khung chat mới; kernel vẫn chạy trên Kaggle. Kéo output khi xong:

```bash
python3 automation/pull_kernels.py <scratch>/pulled
```

Cần token `~/.kaggle/tok_<acc>` của từng acc. Kernel chưa xong được báo `not finished`.

| Run | Kernel | GPU | Cấu hình | Dữ liệu | Ghi chú |
|---|---|---|---|---|---|
| p1 | `lelynhan/zc-p1-1` | 4×L4 | cấu hình tốt nhất (handoff 03 §5), shots v2 | **public test 1000 câu** | → file nộp; đẩy 20:57 UTC 30/10; trước SafePool, > 7.5 giờ (nghi kẹt) |
| e11 | `lelynhan/zc-e11-1` | 4×L4 | tốt nhất + `--think --max-tokens 8192`, shots v4 | train holdout biz_121,124,129 | **RUNNING** từ ~04:30 UTC (sau nhiều lần xếp hàng); phép thử quyết định cho thinking; so với e8 |

`mxuanquy/zc-e1b-1`, `anderson1202/zc-e1c-1` (baseline cũ của khung chat 1) **đã không còn** (kiểm 31/10 01:20 UTC: không có trong
danh sách kernel, truy vấn trạng thái trả về giống kernel đã xoá).

`linhmai24/zc-e3h-4`, `thnliu/zc-e3b-2` bị Kaggle dừng ở giới hạn 12 giờ, không có output (chỉ dùng để thêm shots: không mất gì
quan trọng; nghi kẹt do chương trình treo, trước SafePool).

Đã xong và đã phân tích (log trong `results/chat2_runs/`): smoke2, e3h-1/2/3, e3b-1, e3t-1/2, e4-1/2, e5-1/2, e6a, e7, e7b, e8, e9, e10, e10l, e10b.
