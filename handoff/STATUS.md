# STATUS — trạng thái hiện tại (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-05. Chi tiết khung 03–05/11: `handoff/04_SESSION_0411.pdf`. Đang chờ user nộp **v28**.*

## Điểm portal (các file chấm cùng đáp án hiện tại, trừ v23)

| File | Portal | Dev set 37 câu (`goldkey/devset.py`) |
|---|---|---|
| `submissions/v23` | 0.87 (đáp án cũ; theo X2 ≈ 0.75 hiện tại) | — |
| `submissions/v25` | 0.825 (bỏ phiếu 5 lần chạy, train 0.9893) | 19 / 37 |
| `submissions/v26` | 0.825 (consensus lần 1) | ~15 / 37 |
| `submissions/diag/X1–X3` | 0.793 / 0.756 / 0.768 | — |
| `submissions/v27` (bản thay thế) | chưa | 36 / 37 |
| **`submissions/v28`** | **chưa — bản đề xuất** | **37 / 37** |

`submissions/diag/G1–G5` (v27 với một nhóm DN để abstain) đã tạo, chưa nộp: điểm(v27) − điểm(Gi) = đúng − sai của nhóm.

## Kết quả Kaggle (tải lại: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <dir>`)

| Bản | Kernel | Job |
|---|---|---|
| v25 (98d601f) | loinguyentran/zc-v25-1, -2; anderson1202/zc-v25-3, -4; phatle1578/zc-v25-5, -6 | train, pub, pub_s1 … pub_s4 |
| v28 (code 05/11) | loinguyentran/zc-v28-1 (train 0.9897), -2 (pub_s5); phatle1578/zc-v28-5 (pub_s8), -6 (pub_s9) | seed 6 / 7 không chạy |

v28 = `src/consensus.py` trên 8 thư mục `results/pub*` ở trên (≈ 85 phút CPU, `--workers 3`) → `automation/make_submission.py`.
Token cũ ở `~/.kaggle/tok_*` mất khi đổi container → xin lại user.

## Việc tiếp theo

1. Điểm portal v28 (> 0.825 → bật `CONSENSUS=1` mặc định trong `docker/predict.sh`, đo thời gian trên 2×RTX 5090).
2. Tìm ~80 câu sai còn lại: G1–G5 trên portal, phân tích 8 câu sai / 15 abstain của train v28,. Sửa bằng quy tắc chung → chạy lại consensus (CPU).
3. Docker cuối trước **16/11/2026 23:59 GMT+7**.
