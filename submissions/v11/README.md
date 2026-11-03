# File nộp public test — hệ v11

Commit code: `e73a151` (branch `claude/datamate-v3-0dny90`). Qwen3.5-9B FP8 trên 4×L4 (Kaggle), mỗi lần chạy khoảng 77 phút
cho 1000 câu.

| File | Nội dung | Abstain |
|---|---|---|
| **`submission.jsonl`** | **Bản nộp v11**: bỏ phiếu 3 lần chạy độc lập (seed 0 / 1 / 2) bằng `src/vote.py`, giống hệt cách `docker/predict.sh` làm trên máy chấm. Một đáp án cần ≥ 2/3 lần chạy đồng ý | 103 |
| `submission_single_run.jsonl` | Một lần chạy v11 (seed 0, kernel zc-v11-2) | 42 |

- Định dạng theo tab Data của BTC: mỗi dòng `{"id", "type", "answer"}`, `unanswerable` có `"answer": null`, câu không trả
  lời là `{"id", "type": "abstain"}`.
- Đã kiểm bằng `automation/make_submission.py`: đủ 1000 id, đúng id của public test, 0 lỗi định dạng.
- Ước lượng trên train (leave-one-out, 3000 câu):
  - một lần chạy v11: PA 0.975 (dev 0.990, holdout 0.946);
  - bỏ phiếu 3 lần chạy: 0.976 – 0.982.
- Trên public, hai lần chạy cùng cấu hình chỉ trùng 85 % đáp án, nên bản bỏ phiếu bỏ qua nhiều câu hơn bản một lần chạy.
  Nộp cả hai sẽ cho biết trên public nên ưu tiên ít sai hay ít bỏ qua.
