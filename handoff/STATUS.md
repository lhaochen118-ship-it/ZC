# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-11 13:00 UTC (bàn giao sang khung chat mới). Lịch sử đầy đủ từng vòng (16 vòng R + hướng M):
`Process/history/STATUS_full_until_2026-11-11.md`; điểm mọi lần chạy: `Process/scores.csv`.*

## 1. Bản nộp hiện tại (chốt 11/11 04:55 UTC)

| Thành phần | Giá trị |
|---|---|
| Luồng | `configs/flow_final.json` = **O** = J + `mask_check` (hybrid, prompt full, k 2 → 8, round 2 think, review 2, period / named-periods / threshold flag, size / unit / len_mask check, fmt_fix, hint_counts, trần 1536 token) |
| Mô hình | Qwen3.5-9B + **WG1 × 0.35** (LoRA trên D1 + toàn bộ D2 paraphrase) — `soup.json` trong dataset `nhatlong1103/zc-final-adapter`, sha256 ở `src/adapter_hashes.json` |
| Public gốc (eval, 2 seed) | **0.986 / 0.985** (tb 0.9855); đường container (`predict_flow.py`, lối vào Docker) **0.986** (992 / 6 / 2), 1,000 câu trong 111 s gộp + 1,172 s trả lời trên 4×L4 |
| Held-out | pub_ho 0.644 / 0.636, tr_ho 0.882 / 0.881 → gộp tb **0.778** |
| `final_submission/submission.jsonl` | file public PA cao nhất từng đạt: **0.991** (994 / 3 / 3, J + WG1 seed 0) — luật user 10/11: chỉ thay khi PA cao hơn (`automation/update_final_submission.py`) |
| Docker | `docker/` sẵn sàng; user tự build 15/11, nộp trước **16/11 23:59 GMT+7** (`docker/SETUP_DOCKER.md`) |

## 2. Đang chạy lúc bàn giao

| Kernel | Việc | Khi xong |
|---|---|---|
| `bichthuy93/zc-wg2r-1` | huấn luyện WG5–8: **xong** (66 bước / nhánh, loss tham chiếu 0.0001–0.003); adapter tải về scratch, WG7 + WG8 đăng thành `huuhien95/zc-wg78-adapters` | — |
| `bichthuy93/zc-sc3wg5-1`, `zc-sc3wg6-1` (12/11 03:37 UTC) | sàng lọc WG5 / WG6 × 0.35, cấu hình O, `pub_full,pub_ho400,tr_ho400` (adapter mount thẳng từ output `zc-wg2r-1`) | so với O + WG1: public 0.9855, tập con 0.746 → nhánh hơn thì xác nhận đủ tập 2 seed |
| `huuhien95/zc-sc3wg7-1`, `zc-sc3wg8-1` (12/11 03:39 UTC) | sàng lọc WG7 / WG8 (như trên; `huuhien95/zc-wg78-adapters`) | như trên |
| `caohuyen74/zc-ptr2-1` (12/11 03:41 UTC) | paraphrase câu train mới (`paraphrase_llm.py --n 4`; giữ k = 2, 3: kiểu "lịch sự" / "đổi cấu trúc", D2 cũ chỉ có k = 0, 1) | dựng D2b (`training/data/d2_paraphrase.py`) → nhánh kiểu WG1 trên D1 + D2 + D2b sau khi quota làm mới 14/11 |

## 3. Tiến độ kế hoạch

- Kế hoạch 1 (`docs/08_MASTER_PLAN.md`, WP0 → WP12): **xong** (WP8 phần bộ kiểm 0.8B / truy hồi embedding không làm — thay bằng bộ chọn M6).
- Nhiệm vụ sau kế hoạch 1 (`docs/13_TASK_AFTER_PLAN1.md`): vòng phân tích lỗi (`docs/10`) → tài liệu (`docs/11`) → kế hoạch 2
  (`docs/12_PLAN2.md`) → triển khai; **điều kiện dừng: public gốc 100 % bằng cải tiến tổng quát, held-out không giảm** — chưa đạt
  (tb 0.9855; 3 câu sai ổn định với WG1 sau `mask_check`: bỉm size M theo bịch, bình quân phiếu thu, nhập về N viên — chọn sai hàm;
  còn lại là phiếu tách thay đổi theo seed).
- Hướng R (luật tổng quát, 16 vòng) và hướng M (mô hình, từ M0 = 0.939): kết quả và quyết định ở `handoff/01_SCIENTIFIC_FINDINGS.pdf`.

## 4. Việc tiếp theo (theo thứ tự)

1. **WG5–8** (khi `zc-wg2r-1` xong): đăng adapter thành dataset riêng trên acc đánh giá (như `zc-wg-adapters`), **sàng lọc** mỗi nhánh × 0.35
   với cấu hình O trên `pub_full,pub_ho400,tr_ho400` (một nhánh / kernel, song song nhiều acc, ~40 phút; mốc O + WG1: public 0.9855,
   tập con 0.746) → nhánh nào hơn → **xác nhận** đủ tập (`pub_full,pub_ho,tr_ho`) 2 seed → nếu public tb ↑ ≥ 0.3 và held-out gộp không
   giảm > 0.3 (§4 `docs/12_PLAN2.md`): đổi mô hình nộp bằng `automation/set_final_model.py --publish` + chạy đường container
   (`automation/jobs/p2r16pred.jobs` mẫu) + cập nhật README_SOLUTION / STATUS.
2. **Quota làm mới 14/11 00:00 UTC** (24 acc × 30 h): đợt lớn cuối — thêm seed / nhánh huấn luyện kiểu WG1 (dữ liệu D1 + D2 là đòn bẩy đã
   chứng minh), đánh giá nhiều seed của ứng viên tốt nhất; mọi lần chạy có public → `update_final_submission.py` + commit.
3. Ý tưởng chưa thử còn lại (ưu tiên): bộ chọn huấn luyện trên log của WG1 (cần log WG1 trên `tr_dev_all,d3sel,tr_dev_para`); thêm
   paraphrase train (D2 mới bằng `src/paraphrase_llm.py`) cho nhánh mới; kiểm tra tổng quát cho 3 câu chọn sai hàm (chỉ nếu tín hiệu sạch
   trên mọi log, như `len_mask`).
4. Trước 15/11: chốt bản nộp, chạy đường container lần cuối, cập nhật `docker/README_SOLUTION.md`, báo user build.

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
