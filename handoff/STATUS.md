# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-15 03:40 UTC. **Kế hoạch v2 đang chạy** (user 14/11 17:00: chế độ H, không luật viết tay; user 15/11 01:15: **dừng
held-out, chỉ thử nhiều giải pháp trên public**, chọn điểm cao nhất). Mốc chế độ H `pub_full_H` **0.863** (H0a = H0a2). Lưới public:
`handoff/06_PUBLIC_GRID.md`; data augmentation S″: `handoff/07_DATA_AUG_PLAN.md`. Đóng băng 16/11 04:00 UTC. Lỗi vận hành + quy tắc
tránh: PDF 2 §9 (`handoff/02_ENGINEERING_OPS_LESSONS.md`).*

**Đang chạy (03:40):** lưới A/B còn lại (pwts, pwtbk24, ps1b, pwtbw1r, pwtbu24, pwtbma1, pbase, pwtbv2 — đều dùng WT-b, dự kiến kém mốc);
B′ bộ viết hàm = S: phu16 (caotienminh1), phnou (mucnokieu), phut (chờ test CPU); S″-A huấn luyện (duylnng/zc-s3a-1); kiểm ngược UQ
uqchk_a/b (lelynhan) → S″-A′. Docker: dataset W adapter `nhatlong1103/zc-w-v2` (WL1, WJR1) đã đăng, sha256 khớp `src/adapter_hashes_w.json`.

**Kết quả public chế độ H tới giờ:** mốc 0.863 / 0.863; F1 0.847; S′ 0.861 / 0.856; WT-b 0.855; WT-b8 0.846; S′ + WT-b 0.830. Hàm WT-b sai
nghĩa trên public (mẫu gọi hàm đúng 0.149, hàm của S 0.514) → G1 loại.
## Tiến độ v2 (kế hoạch chạy `handoff/05_RUN_PLAN.md`; T0 = 17:05 UTC 14/11, user duyệt 17:00)

| Bước | Acc / kernel | Trạng thái | Kết quả |
|---|---|---|---|
| Chế độ H không rò rỉ (thẻ W1 / link / phán xử / W1 bind) | CPU | xong | 14 / 125 thẻ public từng gợi ý hàm `industry.py` → đã lọc; `tests/test_mode_h.py` |
| H0a `pub_full_H` (container) | quynhle07/zc-h0a-1 | xong | **pub_full_H 0.863** (907 / 44 / 49; point_think riêng 0.868); pub_ind 0.606, pub_noind 0.125; 110 phút. F1 − H0a = −0.016 (SE 0.011, z −1.5) |
| H0b held-out H | quynhle07/zc-h0b2-1 | xong | bỏ phiếu (chưa phán xử): **pub_ho400c 0.581, tr_ho400c 0.901, pub_hoind 0.455, tr_newterm 0.852** |
| H0c held-out thường | quynhtran35/zc-h0c2-1 | xong | bỏ phiếu: pub_ho400c 0.644, tr_ho400c 0.912, **pub_hoind 0.704** (H: 0.455), tr_newterm 0.815 |
| D1 container `flow_p5_final` | quynhtran35/zc-d1-1 | xong | **public 0.952** (973 / 21 / 6, file ensemble; còn `industry.py`) → `final_submission/p5_container_d1/` |
| TG0 thuật ngữ mới (Qwen gốc) | nguyenhoa99/zc-tg0-1 | xong | 1,446 / 2,681 thuật ngữ (`training/data/out/tg0`) |
| A1 phán xử ràng buộc (4 log × 4 biến thể) | nguyenhoa99/zc-a1-1 | xong — **A loại** | đọc được 100 % (point+c 1375/1375 vs 479) nhưng PA không tăng: list_c+point_c+point_think m0.5 TB 0.946; list_c+point_think 0.950–0.952 vs mốc chạy lại list+point+point_think 0.951 (0.944 / 0.956 / 0.958 / 0.947); có log giảm → không qua G-A |
| F1 `pub_full_H` + F | yuan1103/zc-f1-1 | xong | **pub_full_H 0.847** (903 / 56 / 41); pub_ind 0.550, pub_noind 0.000; 14 hàm W (709 s), 97 phút; F2 held-out → f2b |
| F2b held-out H + F | yuan1103/zc-f2b-1 | xong | pub_ho400c 0.603 (+0.023, z 1.4), tr_ho400c 0.899 (−0.003), **pub_hoind 0.499 (+0.044, z 2.3)**, tr_newterm 0.852 (0) so với H0b2; G-F chờ H0a2 (nhiễu giữa 2 lần chạy) |
| EWTS2 mốc WT: W = S (H, 614 câu ngành) | phuongnehe/zc-ewts2-1 | xong | pub_ind 0.576, pub_hoind 0.475 (15 hàm; kế hoạch 4: 0.595 / 0.493) |
| H0a2 `pub_full_H` seed 2 | synchan5704/zc-h0a2-1 | xong | **0.863** (909 / 46 / 45) = H0a → nhiễu nhỏ; F1 − TB(H0a, H0a2) = −0.016 (z −1.6) → **G-F: F loại** |
| EWBT2 W = Qwen gốc có suy nghĩ (H, 614 câu) | phuongnehe/zc-ewbt2-1 | xong | pub_ind 0.539, pub_hoind 0.455; chỉ 2 hàm qua kiểm (1,552 s) → kém W = S; suy nghĩ lúc suy luận không phải lời giải |
| bl1 B-lite: phán xử xem lại câu đồng thuận (4 log public, WJR1) | anderson1202/zc-bl1-1 | xong — **B-lite loại** | câu đồng thuận bị bác chắc chắn: point+think 6 đúng / 1 sai (f5a), 14 / 0 (f5b), 6 / 1, 13 / 3; point+c 35 / 1 … → ghi đè chỉ mất điểm; bộ phán xử không thấy lỗi đồng thuận |
| jh0b phán xử WJR1 trên log held-out H0b2 | nguyenhoa99/zc-jh0b-1 | xong | **mốc held-out H có phán xử: pub_ho400c 0.609, tr_ho400c 0.909, pub_hoind 0.470, tr_newterm 0.852** (bỏ phiếu cùng log: 0.563 / 0.901 / 0.426 / 0.852) |
| TG1 thầy WT (suy nghĩ) | phatle1578, mucnokieu | **hỏng → huỷ** | p0 giữ 7 / 243 dòng: Qwen gốc suy nghĩ lặp vòng quá 7–8k token (304 / 447 mẫu) — lỗi G0b kế hoạch 4 |
| TG1b thầy WT (không suy nghĩ, JSON ràng buộc, 8 mẫu, đồng thuận ≥ 3) | phatle1578, mucnokieu ×2 | xong | 2,654 dòng → giữ 782 hàm (train_rule 143, biến thể 145, mới 18, thuật ngữ mới 476) |
| TG1c lượt thầy 2 (chỉ dòng tổng hợp, seed 777) | phatle1578, mucnokieu ×2 | xong | +204 hàm (168 biến thể, 36 mới) cho WT v2 |
| Hàm từ chương trình đã kiểm (`wt_from_programs.py`, CPU) | CPU | xong | 1,249 / 1,446 thuật ngữ → hàm của `p`, kiểm lại bằng thực thi (cân bằng còn 917) |
| Dữ liệu WT (`wt_build.py`) | CPU | xong | 864 bản ghi (3.5M token) + 265 dòng đo (6 DN held-out) |
| WT-b / WT-s huấn luyện | nhatlong1103 ×2 (wtb2, wts3) | xong | 861 bản ghi × 2 epoch, loss 0.256→0.101 / 0.223→0.111, 1.85k tok/s, 11.6 GB (mb 2); lần 1–2 OOM |
| S′ huấn luyện seed 1 / 2 | nguynduylm ×2 | xong | 7,077 bản ghi, 24.1M token, 4.0 h mỗi seed; loss 0.025→0.026 / 0.026→0.018 |
| WT0B (238 / 543 bản ghi do max-len) → EWT0 (614 câu H) | anderson1202 | xong | **tích hợp chạy đúng** (WT0B gộp vào trọng số gốc, 14 hàm, có 'Dòng máy', 'hãng' cho dev_014); pub_ind 0.528, pub_hoind 0.438 (< mốc W = S 0.576 / 0.475; dữ liệu quá nhỏ) |
| ewu_ref: độ chính xác hàm W = S, W = Qwen gốc (265 dòng held-out) | synchan5704 | xong | **W = S 0.517** (giữ 221, đúng 137; nhãn chương trình đã kiểm 67 / 157); Qwen gốc 0.423 (41 / 157) |
| TG1n (thuật ngữ mới, suy nghĩ) | huyenco | huỷ | gộp vào TG1b |
| TG2 cặp tương phản | nguyenhoa99/zc-tg2d-1 | xong (lần 4) | **1,005 cặp** (729 tương phản, kiểm ngược mù 97 %; 276 cách viết); lỗi trước: 2 engine vLLM, HTML-escape `&quot;` |
| WJR2D (LoRA chồng V3A×0.70 + WJR1; 1,738 bản ghi tương phản + 1,000 WJR1) | quynhle07 (ej2) | xong — **G-J: loại** | public tổ hợp TB 0.9425 (0.934 / 0.946 / 0.950 / 0.940) vs WJR1 0.951; held-out pub_ho400c 0.595 (WJR1 0.609), tr_ho400c 0.912 (0.909), pub_hoind 0.464 (0.470) |
| Đo S′ trên public H (held-out huỷ theo user 15/11) | nguynduylm ×2 | xong — **S′ riêng loại** | seed 1 0.861, seed 2 0.856 (mốc 0.863) |
| ewuwtb: độ chính xác hàm WT-b (265 dòng held-out) | synchan5704 | xong | **0.596** (giữ 244, đúng 158; nhãn chương trình đã kiểm 98 / 157) vs W = S 0.517: **+0.079, z 2.82** (39 tốt hơn / 18 kém hơn) |
| ewuwts: độ chính xác hàm WT-s | synchan5704 | xong | 0.566 (đúng 150; chương trình đã kiểm 100 / 157) < WT-b 0.596 → **WT-b là nhánh chính** (LoRA trên Qwen gốc giữ kỹ năng pandas tổng quát) |
| Đo WT đầu-cuối (614 câu H: ewtb, ewts_wt) | nhatlong1103 ×2 | chạy | |

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
