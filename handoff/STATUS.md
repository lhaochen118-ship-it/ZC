# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-14 05:20 UTC. Dự án **đang dừng** (user 14/11 02:30): kế hoạch 4 xong, PDF 1 / 2 / 3 cập nhật, repo đã dọn. Chờ user đưa
chiến lược mới hướng tới public = 1. Routine tự tiếp tục đã tắt. Lịch sử chi tiết: `docs/17_PLAN4_SWV.md` §7, PDF 1 / 2; bản STATUS cũ trong git.*

## 1. Hệ nộp

| Thành phần | Giá trị |
|---|---|
| Luồng | `configs/flow_final.json` = S–W, không luật viết tay: S viết chương trình mọi câu, W viết hàm luật từ rulebook (5 mẫu, 1 vòng tự sửa, trần 45 phút), thẻ luật do mô hình biên dịch, k_fast 4 / k_max 16, bỏ phiếu thường θ 0.5 |
| Mô hình | Qwen3.5-9B + v3a × 0.70 gộp; `nhatlong1103/zc-final-adapter` (bản 14/11 04:35), sha256 `src/adapter_hashes.json` |
| Public gốc | **0.938** (eval, pva3k16) / **0.922** (đường container f2a3, 39 phút trên 4×L4; `final_submission/swv/`) |
| Dự phòng P | `configs/flow_p_final.json` + WG1 × 0.35 (bản cũ của dataset, `src/adapter_hashes_p.json`): public 0.994; `final_submission/submission.jsonl` vẫn là file P (PA cao nhất) |
| Docker | `docker/` cập nhật (README_SOLUTION, assets.env, import check); user build trước hạn 16/11 23:59 GMT+7 |

## 2. Kết quả chính kế hoạch 4 (public, `Process/scores_p3.csv`)

v3a k 8 / k 16: 0.936 / 0.938; + V4S1 0.925 / 0.926; + V4S2 0.920 / 0.932; soup 0.928 / 0.932; v3b 0.926 / 0.925 (bỏ phiếu thường). W ở ngành mới
(614 câu, ẩn hàm ngành): +0.119 / +0.078 (z 3.8 / 3.3). V không lợi trên public. Lỗi còn lại (39): 17 có mẫu đúng, 22 không mẫu nào đúng.

## 3. Tài nguyên

Token 29 acc ở `~/.kaggle/tok_*` (mất khi đổi container). Quota 14/11 ~05:00: phần lớn acc còn 25–30 h (làm mới 21/11). Adapter trên Kaggle:
`<acc>/zc-v3a-adapter` (≥ 13 acc), `zc-v4s{1,2}-adapter`, `zc-v3b-adapter`; WR30 (30 DN, dự phòng) ở output `anderson1202/zc-wr30p4-1`.
Không còn kernel nào chạy.

## 4. Việc tiếp theo

User duyệt chiến lược mới → kế hoạch chi tiết (luồng, batch, thời gian) → chạy. Trước hạn: user build Docker và nộp.
