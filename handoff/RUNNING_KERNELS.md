# Kernel còn chạy lúc bàn giao (2026-10-30 16:40 UTC)

Các tiến trình theo dõi cục bộ của khung chat cũ đã mất; kernel vẫn chạy trên Kaggle. Kéo output khi xong:

```bash
python3 automation/pull_kernels.py runs_pulled
```

(cần token `~/.kaggle/tok_<acc>` của từng acc). Kernel chưa xong được báo `not finished`.

| Run | Kernel | GPU | Cấu hình | Code | Ghi chú |
|---|---|---|---|---|---|
| e1b | `mxuanquy/zc-e1b-1` | 2×T4 | Qwen3-14B-AWQ, TP=2, path B only, K=4, repairs 2, 300 câu train (10/DN) | trước mọi bản sửa (có lỗi `1,134tr`, thrash prefix, chưa có bảo vệ độ dài prompt) | baseline; chạy từ 12:04 UTC, rất chậm |
| e1c | `anderson1202/zc-e1c-1` | 2×T4 | như e1b + `--think`, K=2, max-tokens 6144 | như e1b | baseline có thinking |
| e2b | `nguyenhoa99/zc-e2b-1` | 4×L4 | `--compile --k 4 --kc 4 --dp 4`, 300 câu train | có tắt FlashInfer sampler; **chưa** có bảo vệ độ dài prompt / sinh theo nhóm prefix / chép model cục bộ | nếu 1 shard chết vì prompt > 24576 token thì thiếu ~1/4 câu (abstain) |
| e2c | `nguyenhoa99/zc-e2c-1` | 4×L4 | `--compile --k 4 --kc 4 --dp 4`, 1000 câu public test | **mới nhất** (mọi bản sửa đến commit lúc 16:20 UTC) | không có đáp án → chỉ xem tốc độ, tỷ lệ chương trình chạy được, độ đồng thuận |

Đã xong (kết quả đã phân tích, xem `handoff/01_SCIENTIFIC_FINDINGS.md`): `nguyenhoa99/zc-e2b-2` (E2b public: 746/1000
câu do 1 shard chết), `nguyenhoa99/zc-e2-1`, `zc-e2-2` (E2 lần 1: hỏng toàn bộ vì FlashInfer JIT).

Phân tích một run train: `python3 src/analyze.py <out>/results/logs.jsonl data/raw/train/train/answers.jsonl --show 10`.
