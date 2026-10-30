# Kernel còn chạy lúc bàn giao (2026-10-30 16:40 UTC)

Các tiến trình theo dõi cục bộ của khung chat cũ đã mất; kernel vẫn chạy trên Kaggle. Kéo output khi xong:

```bash
python3 automation/pull_kernels.py runs_pulled
```

(cần token `~/.kaggle/tok_<acc>` của từng acc). Kernel chưa xong được báo `not finished`.

| Run | Kernel | GPU | Cấu hình | Code | Ghi chú |
|---|---|---|---|---|---|
| e2b | `nguyenhoa99/zc-e2b-1` | 4×L4 | `--compile --k 4 --kc 4 --dp 4`, 300 câu train | có tắt FlashInfer sampler; **chưa** có bảo vệ độ dài prompt / sinh theo nhóm prefix / chép model cục bộ | nếu 1 shard chết vì prompt > 24576 token thì thiếu ~1/4 câu (abstain) |

Đã huỷ 30/10 ~20:40 UTC: `mxuanquy/zc-e1b-1`, `anderson1202/zc-e1c-1` (E1 baseline trên T4, code cũ có lỗi, chạy > 8 h, có nguy cơ chạm trần 12 h mà không ghi được kết quả vì code cũ chỉ ghi ở cuối; giải phóng quota T4 cho lần chạy lại). Không có số liệu E1.

Đã xong (kết quả đã phân tích, xem `handoff/01_SCIENTIFIC_FINDINGS.md`): `nguyenhoa99/zc-e2c-1` (E2c public, code mới nhất: 1000/1000 câu, không có đáp án; **không tạo file nộp** theo yêu cầu user), `nguyenhoa99/zc-e2b-2` (E2b public: 746/1000
câu do 1 shard chết), `nguyenhoa99/zc-e2-1`, `zc-e2-2` (E2 lần 1: hỏng toàn bộ vì FlashInfer JIT).

Phân tích một run train: `python3 src/analyze.py <out>/results/logs.jsonl data/raw/train/train/answers.jsonl --show 10`.
