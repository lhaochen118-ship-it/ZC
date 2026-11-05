# STATUS — trạng thái hiện tại (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-05 (chiều). Chi tiết: `EXPERIMENT_LOG.md` entry 20. Bàn giao cũ (03–05/11): `handoff/04_SESSION_0411.pdf`.*

## Kiến trúc hiện tại: rule-first (`docker/predict.sh` → `src/predict_flow.py`)

| Stage | Thành phần | Kết quả |
|---|---|---|
| 1 (CPU, phút) | `src/rules_first.py` = `automap` (bảng thô → C) + `qnorm` (làm sạch câu) + `qparse` (câu → chương trình) + toolkit `dm` / `industry` | train **3000 / 3000**, public **1000 / 1000** (chấm nghiêm; cả pandas 2.3.3 + calamine của image) |
| 2 (GPU) | `agent.py` (Qwen3.5-9B FP8, mô hình duy nhất) chỉ cho câu stage 1 bỏ lại, ánh xạ ưu tiên `fmt:automap` | đang đo trên Kaggle (`automation/jobs/v30.jobs`) |

Kiểm tra tổng quát hoá (`S=<scratch> WORKERS=4 bash tests/eval_all.sh`): README xoá trắng train 3000 / 3000, public 0.991;
câu không dấu 0.981; nhiễu chat 0.95–0.96 (phần thiếu chủ yếu là abstain → stage 2).

## Kaggle

- Acc phiên này (token ở `~/.kaggle/tok_<acc>`, mất khi đổi container): L4 quynhtran35, yuan1103, phatle1578; T4 yeshenhi,
  mxuanquy. Lấp đầy 1 acc (2 GPU + 5 CPU) rồi mới sang acc khác.
- Trọng số: Kaggle Model `danbth/qwen3-5-9b/transformers/default/1` (`--ms`), `run_kaggle.py` kiểm sha256 với Hugging Face
  (`src/model_hashes.json`). Wheels: `--ks nhatlong1103/zc-assets`. Dữ liệu: `nhatlong1103/zc-2026`. Dataset trọng số cũ
  (`nhatlong1103/…`, `lelynhan/…`) là private → 403 với acc mới.
- v30 (GPU, quynhtran35): luồng nộp trên câu public / train biến đổi; v30cpu: luật trên môi trường Kaggle + seed nhiễu mới.

## Việc tiếp theo

1. Đọc kết quả v30: độ chính xác của stage 2 trên câu luật bỏ lại (> 50 % mới có lợi; nếu không → `LLM=0` hoặc ngưỡng).
2. Paraphrase bằng LLM (`paraphrase_llm.py`) → kiểm độ bền của bộ đọc câu với cách hỏi mới; sửa bằng quy tắc chung.
3. Tạo file nộp (`automation/make_submission.py`), build Docker, đo trên GPU trước **16/11/2026 23:59 GMT+7**.
