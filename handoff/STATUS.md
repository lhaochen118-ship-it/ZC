# STATUS — trạng thái hiện tại (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-05 17:45. Chi tiết: `EXPERIMENT_LOG.md` entry 20–21. Bàn giao cũ (03–05/11): `handoff/04_SESSION_0411.pdf`.*

## Kiến trúc hiện tại: rule-first (`docker/predict.sh` → `src/predict_flow.py`)

| Stage | Thành phần | Kết quả |
|---|---|---|
| 1 (CPU, phút) | `src/rules_first.py` = `automap` (bảng thô → C) + `qnorm` (làm sạch câu) + `qparse` (câu → chương trình) + toolkit `dm` / `industry` | train **3000 / 3000**, public **1000 / 1000** (chấm nghiêm; cả pandas 2.3.3 + calamine của image) |
| 2 (GPU) | `agent.py` (Qwen3.5-9B FP8, mô hình duy nhất) chỉ cho câu stage 1 bỏ lại, ánh xạ ưu tiên `fmt:automap` | đang đo trên Kaggle (`automation/jobs/v30.jobs`) |

Kiểm tra tổng quát hoá (`S=<scratch> WORKERS=4 bash tests/eval_all.sh`): README xoá trắng train 3000 / 3000, public 0.993;
câu không dấu 0.989; nhiễu chat 0.972–0.978. **Paraphrase bằng LLM (ước lượng thật cho cách hỏi mới), chỉ luật:** public
held-out 0.506, train held-out 0.669 (abstain ~23–30 % → stage 2). File nộp public: `submissions/v30/` (1000 / 1000).
Routine tự tiếp tục khi hết quota: `trig_01Vrvp6RpbNVxYYfUB1YGeJW` (mỗi giờ, :42) — xoá khi xong việc.

## Kaggle

- Acc phiên này (token ở `~/.kaggle/tok_<acc>`, mất khi đổi container): L4 quynhtran35, yuan1103, phatle1578; T4 yeshenhi,
  mxuanquy. Lấp đầy 1 acc (2 GPU + 5 CPU) rồi mới sang acc khác.
- **Quota GPU tính theo GPU:** một phiên 4×L4 tiêu ~4 giờ quota mỗi giờ chạy (30 h / tuần / acc ≈ 7.5 h phiên 4×L4).
  Tuần 2026-W45: quynhtran35, phatle1578 hết; yuan1103 đang chạy v32 (có thể hết giữa chừng). T4 không chạy được Qwen3.5-9B FP8.
- Trọng số: Kaggle Model `danbth/qwen3-5-9b/transformers/default/1` (`--ms`), `run_kaggle.py` kiểm sha256 với Hugging Face
  (`src/model_hashes.json`). Wheels: `--ks nhatlong1103/zc-assets`. Dữ liệu: `nhatlong1103/zc-2026`. Paraphrase / câu chờ:
  dataset private `<acc>/zc-para`, `phatle1578/zc-pend` (mỗi acc một bản: private không chia sẻ được giữa acc).
- Kết quả: v30 (câu nhiễu, toàn luồng 0.989–0.996), v30d (CPU, môi trường Kaggle 1000/1000, 3000/3000), v31 (paraphrase:
  public 0.66 / 0.59 / 0.57, train 0.768; phần giữ kỳ 0.79 / 0.71 / 0.82), v32 (code mới nhất: paraphrase public 0.68 dev / 0.58 held-out, train held-out 0.780; giữ kỳ 0.81 / 0.71 / 0.83), v33 (stage 1.5
  `canon_llm.py`, chưa chạy: chờ quota).

## Việc tiếp theo

1. (xong) v32 tốt hơn v31 trên cả dev và held-out (entry 25).
2. Khi quota reset: v33 (stage 1.5) → chỉ bật `--canon 1` nếu luật trên câu viết lại chính xác hơn mô hình.
3. **Việc của user:** build Docker (`docker/SETUP_DOCKER.md`: `docker/fetch_assets.sh` → `docker/build.sh` →
   `docker/run_public_test.sh` → `docker/export_image.sh`), nộp link + checksum trước **16/11/2026 23:59 GMT+7**.
   File nộp public hiện tại: `submissions/v30/` (1000 / 1000 khi chấm cục bộ).
