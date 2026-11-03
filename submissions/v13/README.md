# File nộp public test — hệ v13

Code: commit `69c0d8d` = v11 + mẫu câu "lô sắp hết hạn" / "hết hàng". `submission.jsonl` = bỏ phiếu 3 lần chạy (seed 0 / 1 / 2)
bằng `src/vote.py`. Định dạng đã kiểm bằng `automation/make_submission.py`: 1000 id, 0 lỗi.

| | v11 | v13 |
|---|---|---|
| Train một lần chạy (PA) | 0.975 (dev 0.990, holdout 0.946) | 0.955 (dev **0.996**, holdout 0.876) |
| Public: số câu bỏ qua của bản bỏ phiếu 3 lần | 103 | 67 |
| Public: độ trùng giữa các lần chạy | ≈ 85 % | 84 – 89 % |

- Holdout của v13 thấp vì biz_123 và biz_124 chỉ có 1 ứng viên ánh xạ trong lần chạy đó (may rủi khi LLM viết ánh xạ).
- Trên dev, v13 tốt hơn v11.
