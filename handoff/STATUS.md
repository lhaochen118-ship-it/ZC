# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-14 15:30 UTC. Kế hoạch 5 (`docs/18_PLAN5_TWO_LORA.md`) **xong phần thử nghiệm, đang DỪNG** chờ user đưa chiến lược mới (khung chat
Fable, kế hoạch dạng docx). Lỗi vận hành + quy tắc tránh: PDF 2 §9 (`handoff/02_ENGINEERING_OPS_LESSONS.md`).*

## Tiến độ v2 (kế hoạch chạy `handoff/05_RUN_PLAN.md`; T0 = 17:05 UTC 14/11, user duyệt 17:00)

| Bước | Acc / kernel | Trạng thái | Kết quả |
|---|---|---|---|
| Chế độ H không rò rỉ (thẻ W1 / link / phán xử / W1 bind) | CPU | xong | 14 / 125 thẻ public từng gợi ý hàm `industry.py` → đã lọc; `tests/test_mode_h.py` |
| H0a `pub_full_H` (container) | quynhle07/zc-h0a-1 | chạy | |
| H0b held-out H | quynhle07/zc-h0b-1 | chạy | |
| H0c held-out thường | quynhtran35/zc-h0c-1 | chạy | |
| D1 container `flow_p5_final` | quynhtran35/zc-d1-1 | chạy | |
| TG0 thuật ngữ mới (Qwen gốc) | nguyenhoa99/zc-tg0-1 | xong | 1,446 / 2,681 thuật ngữ (`training/data/out/tg0`) |
| A1 phán xử ràng buộc (4 log × 4 biến thể) | nguyenhoa99/zc-a1-1 | chạy | |
| F1 `pub_full_H` + F | yuan1103/zc-f1-1 | chạy | F2 (held-out) đẩy sau |
| TG1 thầy WT p0–p3 | phatle1578, mucnokieu (2 phiên mỗi acc) | chạy | |
| TG1n thầy cho thuật ngữ mới p0 (p1 sau) | huyenco | chạy | |
| TG2 cặp tương phản | loinguyentran | chạy | |

## 0. Kế hoạch 5 — kết quả (public, đáp án chuẩn)

| Hệ | Public |
|---|---|
| Kế hoạch 4 (S + W unit, bỏ phiếu) | 0.938 |
| F5 container × 2 (S + W1 bind + W2 linker WL1 + W3 phán xử WJB1) | 0.943 / 0.941 (`final_submission/swv/`) |
| **Cấu hình cuối `configs/flow_p5_final.json`** (W3 = WJR1, tổ hợp 3 biến thể, ngưỡng 0.5) — phán xử lại offline trên 4 log S | **0.949–0.958, TB 0.952** |

Adapter (LoRA, học chồng trên V3A × 0.70): WL1 (`synchan5704/zc-wl1b-1` output; dataset `{yuan1103,synchan5704}/zc-w-adapters`), WJB1
(`nguyenhoa99/zc-wjb1-1`; trong `zc-w-adapters`), **WJR1** (`quynhtran35/zc-wjr1-1` output); cả 3 adapter ở **`Process/adapters/` (Git LFS)** và Kaggle `nhatlong1103/zc-w-final`. Log: `Process/runs/f5a`,
`f5b`, `training/data/out/jlogs/` (không commit), dữ liệu W: `training/data/out/{wj,wl}` (dựng lại: `training/data/w_judge.py --aug 1 --rows
training/data/out/wj/p_rows.jsonl --tag _r`, `w_link.py --max-per-shop 60`, `p_rows.py`).

**Việc còn lại trước hạn (16/11 04:00 UTC đóng băng):** D1 Docker cho cấu hình cuối (đăng WJR1 + WL1 vào dataset adapter cuối; `docker/` thêm
`training/merge.py` + 2 adapter W; chạy `docker/run_public_test.sh`). Quota 14/11 15:00: các acc còn ~14–25 h.

**Đề xuất chiến lược mới v2 (14/11 17:00 UTC, chờ user duyệt):** `proposed plan.pdf` (nguồn `handoff/04_PROPOSED_PLAN.md`, bản 2) + kế hoạch chạy `handoff/05_RUN_PLAN.md` (`run plan.pdf`). Thay đổi so với bản 1: đo trung thực chế độ H (`hide_api: industry` — `flow_p5_final` hiện vẫn cho S gọi `industry.py` viết tay), cổng held-out cho mọi quyết định, E sinh có hệ thống (không lấy trục từ câu public sai), C thêm định nghĩa ngoài toolkit, giải pháp mới G = LoRA viết tool WT (huấn luyện độc lập, thầy = Qwen gốc có suy nghĩ). Token 14 acc đã lưu `~/.kaggle` (14/11 16:44); quota 334 h.

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

**Kế hoạch 5 (user duyệt 14/11 06:30): `docs/18_PLAN5_TWO_LORA.md`** — S đóng băng + LoRA-W ba vai huấn luyện riêng; bước kế: P0 (code CPU +
smoke) → Z (zero-shot) → T1 (huấn luyện W) → E1 / G1. Đáp án public: `python3 automation/rebuild_public_gt.py` (cần `git fetch --unshallow`).
Token đang có (14/11): 10 acc L4 + `yeshenhi` (T4). Đóng băng 16/11 04:00 UTC; sau đó user build Docker và nộp.
