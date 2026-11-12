# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-11 13:00 UTC (bàn giao sang khung chat mới). Lịch sử đầy đủ từng vòng (16 vòng R + hướng M):
`Process/history/STATUS_full_until_2026-11-11.md`; điểm mọi lần chạy: `Process/scores.csv`.*

## 1. Bản nộp hiện tại (chốt 12/11 07:40 UTC)

| Thành phần | Giá trị |
|---|---|
| Luồng | `configs/flow_final.json` = **P** = O + `count_check` + `size_use_check` + `ticket_avg_check` (vòng 17, `docs/12_PLAN2.md`); O cũ ở `configs/flow_o.json` |
| Mô hình | Qwen3.5-9B + **WG1 × 0.35** (`nhatlong1103/zc-final-adapter`, sha256 `src/adapter_hashes.json`) — không đổi |
| Public gốc (eval, 3 seed) | **0.993 / 0.994 / 0.993** (tb 0.9933; O 0.986); đường container (`predict_flow.py`) **0.994** (995 / 1 / 4), 1,000 câu: gộp 90 s + trả lời 1,168 s trên 4×L4 |
| Held-out | gộp 0.7718 / 0.7743 / 0.7784 (tb 0.7748; O 0.7784, −0.36 — quy về nguyên nhân ≈ −0.01, phần còn lại là nhiễu chạy lại; ngoại lệ ghi ở `docs/12_PLAN2.md` vòng 17) |
| `final_submission/submission.jsonl` | **0.994** (995 / 1 / 4, P seed 1) |
| Docker | `docker/` sẵn sàng (README_SOLUTION cập nhật P); user build 15/11, nộp trước **16/11 23:59 GMT+7** |

## 2. Đang chạy lúc bàn giao

| Kernel | Việc | Khi xong |
|---|---|---|
| `bichthuy93/zc-wg2r-1` | huấn luyện WG5–8: **xong** (66 bước / nhánh, loss tham chiếu 0.0001–0.003); adapter tải về scratch, WG7 + WG8 đăng thành `huuhien95/zc-wg78-adapters` | — |
| `zc-sc3wg5…8-1` (bichthuy93, huuhien95) | sàng lọc WG5–8: **xong** — public 0.979 / 0.985 / 0.973 / 0.978, held-out tập con 0.7575 / 0.7463 / 0.7588 / 0.75 (mốc WG1 0.9855 / 0.746) → không nhánh nào hơn WG1 ở public, giữ WG1 (`docs/12_PLAN2.md` vòng 17) | — |
| `caohuyen74/zc-ptr2-1` | paraphrase train mới: **xong** (12,000 → 11,292 giữ; `Process/train_data/ptr2/`) → **D2b** 3,742 bản ghi (k 2–3; `Process/train_data/wg/d2b_full.jsonl.gz`) | nhánh kiểu WG1 trên D1 + D2 + D2b sau khi quota làm mới 14/11 00:00 UTC |
| P seed 0 / 1 / 2, O seed 2, đường container P (`hieuho669`, `dinhthinh`, `caohuyen74`, `loinguyentran`, `bichthuy93`) | **xong** — số ở §1 | — |
| `hieuho669/zc-wg9f-1`, `dinhthinh/zc-wg10f-1`, `caohuyen74/zc-wg11f-1`, `huuhien95/zc-wg12f-1` (12/11 09:05 UTC) | huấn luyện **WG9–12 song song, mỗi nhánh một acc, FSDP 4 GPU** (`training/full_job.py`, ~2k tok/s; r 32 α 64 lr 2e-4, mb 4 × 4 GPU × ga 4 = 64; WG9 / WG10 D1 + d2mix seed 31 / 32, WG11 D1 + D2 + D2b 21M token, WG12 D1 + D2 + D2b 14.7M) — user 12/11: không tiết kiệm acc, ra kết quả nhanh nhất; thay chuỗi checkpoint 1 GPU / nhánh (~8 h) | ~2.5–3.5 h → adapter ở `results/WG*/adapter` → đánh giá **đủ tập** seed 0 ngay (× 0.35, cấu hình P) trên bichthuy93 / linhmai24 / loinguyentran; seed 1 sau quota 14/11 |

## 3. Tiến độ kế hoạch

- Kế hoạch 1 (`docs/08_MASTER_PLAN.md`, WP0 → WP12): **xong** (WP8 phần bộ kiểm 0.8B / truy hồi embedding không làm — thay bằng bộ chọn M6).
- Nhiệm vụ sau kế hoạch 1 (`docs/13_TASK_AFTER_PLAN1.md`): vòng phân tích lỗi (`docs/10`) → tài liệu (`docs/11`) → kế hoạch 2
  (`docs/12_PLAN2.md`) → triển khai; **điều kiện dừng: public gốc 100 % bằng cải tiến tổng quát, held-out không giảm** — chưa đạt
  (tb 0.9855; 3 câu sai ổn định với WG1 sau `mask_check`: bỉm size M theo bịch, bình quân phiếu thu, nhập về N viên — chọn sai hàm;
  còn lại là phiếu tách thay đổi theo seed).
- Hướng R (luật tổng quát, 16 vòng) và hướng M (mô hình, từ M0 = 0.939): kết quả và quyết định ở `handoff/01_SCIENTIFIC_FINDINGS.pdf`.

## 3b. Quyết định user 12/11

Không chờ 100 % mới dừng: làm hết §4 (đến khi chốt bản nộp + Docker), rồi cùng user phân tích và lập kế hoạch mới (mô hình → score = 1).

## 4. Việc tiếp theo (theo thứ tự; cập nhật 12/11 04:40 UTC)

1. ~~Cấu hình P~~: **xong, P là cấu hình nộp** (§1).
2. **14/11 00:06 UTC** (đã hẹn `send_later`): quota làm mới → huấn luyện **WG9–12** (`automation/jobs/wg3r.jobs`, plan `plan_wg3r.json`, ~7.5 h):
   WG9 / WG10 = D1 + `d2mix_full` (1,909 D2 + 1,909 D2b, cùng tỉ lệ WG1) seed 31 / 32; WG11 = D1 + D2 + D2b, ngân sách 21M token; WG12 = D1 + D2 + D2b,
   14.7M token. Rồi sàng lọc (× 0.35, `pub_full,pub_ho400,tr_ho400`) → xác nhận đủ tập 2 seed nhánh hơn WG1.
3. Trước 15/11: chốt bản nộp, đường container lần cuối, `docker/README_SOLUTION.md`, báo user build.

Đã thử 12/11 (không giữ): WG5–8 (seed 25 / 26, r 64, lr 1e-4) — không nhánh nào hơn WG1 ở public. Ý tưởng chưa làm: tài liệu API thiếu
`purchased_qty_base` ("nhập về bao nhiêu viên"; đổi mọi prompt → cần đo riêng); bộ chọn học trên log WG1 (bộ chọn J từng bị loại: public −0.1…−0.8).

## 5. Kaggle lúc bàn giao (`python3 automation/quota.py`; snapshot `Process/kaggle/quota_2026-11-11.txt`)

- Token: `~/.kaggle/tok_<username>` (600) — **mất khi đổi container → xin lại user** (24 acc; danh sách tên ở `Process/kaggle/datasets_by_account.txt`).
- Còn nhiều giờ: mxuanquy 30 h (chưa accept rules → không L4), yeshenhi 29.9 h (chỉ T4), bichthuy93 17 h (đang huấn luyện), phuongnehe 13.9 h,
  caohuyen74 13.7 h, hieuho669 12.3 h, huuhien95 10.8 h, dinhthinh 10.3 h, loinguyentran 8.7 h, linhmai24 7.3 h; còn lại < 6 h hoặc 0.
- Dataset riêng mỗi acc (zc-para, zc-final-adapter, zc-wg-adapters): `Process/kaggle/datasets_by_account.txt`; acc không có L4:
  `automation/accounts.json`.
- Quy định user: acc < 6 h phải kiểm đủ giờ trước khi giao (tự động trong `launch_many.py`); phiên ≤ 12 h (tự động trong runner); 4×L4 ≈
  1.5–2.2 h quota / giờ tường.

## 6. Quyết định user còn hiệu lực

Xem `CLAUDE.md` (bắt buộc) và `handoff/03_HANDOFF.pdf` §2. Tóm tắt: hybrid framework (mô hình viết mọi chương trình trả lời, luật chỉ
hỗ trợ); không train trên public / private; không vá theo DN / câu; báo cả public và held-out ở mọi mốc; chạy tự động không dừng (routine
mỗi giờ); `final_submission/` luôn là file public tốt nhất; kiểm quota và giới hạn 12 h; chạy nhanh nhất mà không ảnh hưởng nhiều độ chính
xác (sàng lọc tập con song song → xác nhận đủ tập).
