# Kế hoạch P5 zero-shot (15/11/2026) — sổ thí nghiệm

Nguồn: `DataMate_Execution_Plan_P5_ZeroShot_v2.pdf` (user 15/11; ID B00/M/P/V/S/H/A/T/I/D/R/G/L/Q). Bản chi tiết trước đó: `docs/execution_plan_0.md`.
User 15/11 08:20: xong toàn bộ trong 7–8 h (tới ~16:30 UTC), dùng hết 2 phiên GPU + 5 phiên CPU mỗi acc trước khi sang acc khác; kiểm tra lỗi /
tối ưu trước khi chạy; chạy bản zero-shot trước fine-tune để so sánh.

## 1. Quy ước

- **Tập đo D1** = public gốc 1,000 câu, **đường container** (`training/soup_job.py --runner predict` → `src/predict_flow.py`), điểm *container*
  (file Docker ghi) và *ensemble* (file phán xử). Đáp án public chỉ để chấm (`data/raw/public_gt.jsonl` = `score 1.jsonl`; kiểm lại: P 0.994,
  D1 0.952 khớp `final_submission`).
- **B00-S (ZERO_SHOT_P5_STRICT)** = `configs/flow_p5_zs_strict.json`: framework P5 ở **chế độ H** (ẩn `industry.py` — luật nghiệp vụ viết tay
  cho ngành public, vi phạm §1.2 của kế hoạch), Qwen3.5-9B gốc (`--adapters NONE`), WL1 / WJR1 tắt, **không dùng thẻ luật do mô hình
  tuned viết** (bỏ `compile_public_b`, `alias_*`, `induced`, `translit`; `lrl_regen 1`: W1 bind + aliases do chính Qwen gốc viết lúc suy luận;
  hàm cửa hàng (units) và link cũng do Qwen gốc viết). Mọi biến thể zero-shot so với B00-S trên cùng D1 (ghép cặp từng câu).
- **Mốc tuned**: TUNED_REFERENCE 0.952 (`flow_p5_final`, D1 `quynhtran35/zc-d1-1`, còn `industry.py`); cùng giao thức chế độ H: 0.863 / 0.860
  (H0a / H0a2 / pfix, `flow_p5_final_h`). B00-I (zero-shot nhưng **có** `industry.py`) đo để so trực tiếp với 0.952.
- Toolkit trung lập (`dm.py`: hàm toán / truy vấn, kỳ, định dạng) giữ nguyên ở mọi biến thể (kế hoạch §1.2 cho phép công cụ trung lập).

## 2. Thí nghiệm (acc, kernel)

| ID | Nội dung | Config / mô hình | Acc (phiên) | Trạng thái |
|---|---|---|---|---|
| B00-S s0 / s1 | zero-shot strict, 2 seed | `flow_p5_zs_strict` / `_s1` (seed_base 1) | hieuho669 ×2 | chạy 08:38 |
| B00-D | ablation backbone: Qwen gốc + thẻ luật cũ của mô hình tuned | `flow_p5_zs_abl` | caohuyen74 | chạy 08:39 |
| B00-I | zero-shot cùng giao thức tuned 0.952 (có `industry.py`, chẩn đoán) | `flow_p5_zs_ind` | caotienminh1 | chạy 09:00 |
| M01 | Qwen3-14B (FP8 chính thức, Kaggle Model) mọi pha, DP2×TP2, KV fp8 | `flow_p5_zs_m01` | caohuyen74 | chạy 09:10 |
| M02 | Qwen3.5-4B (staged HF) mọi pha | `flow_p5_zs_m02` | duckzuykobip | chạy 08:56 |
| P01 | Qwen3.6-35B-A3B cắt còn 64 expert theo keep-list REAP-64 (khớp router 40/40 lớp), từ FP8 chính thức, 10.93B | `flow_p5_zs_p01` | namhunhh | chạy 08:56 |
| P02 / P03 | REAP tự tính trên prompt framework (DN train): 88 / 92 expert | `training/moe_saliency.py` → `src/stage_hf.py --keep` | namhunhh (saliency, chạy 09:10) / dinhthinh (cắt + chạy) | saliency chạy |
| V01 / V02 | Qwen3.5-4B phán xử mọi câu / chỉ câu bất đồng, trên log B00-S s0 | `--runner judge` | duckzuykobip | sau B00 |
| S01 | 9B gốc tự kiểm (phán xử mọi câu) trên log B00-S s0 | `--runner judge` | bichthuy93 | sau B00 |
| H01 / H02 | probe trạng thái ẩn (logistic / MLP) học trên log DN train (`tr_dev1k`), GroupKFold theo DN | `training/probe_job.py` | bichthuy93 (dữ liệu) | dữ liệu chạy 08:59 |
| A01 / A02 | K cố định 2 / 4; K thích nghi (phát lại offline từ log) | `Process/tools/p5/replay_k.py` | CPU | sau B00 |
| T01 | ví dụ in-context khớp schema DN (Jaccard bảng/cột, chữ ký DN train) | `flow_p5_zs_t01` | huuhien95 | chạy 09:22 |
| I01 | phủ IR trên chương trình train đã kiểm: 79.3 % một lời gọi toolkit (≈ 89 % thêm growth / đếm) | `Process/tools/p5/ir_coverage.py` | CPU | xong |
| I02 | chương trình có kiểu, giải mã ràng buộc grammar toolkit + fallback suy nghĩ | `flow_p5_zs_i02` | nguyenmanh0404 | chạy 09:22 |
| Q01 | bộ giải BF16 thay FP8 (DP2×TP2) | `flow_p5_zs_q01` | huuhien95 | chạy 09:10 |
| L02 | WiSE-FT α của V3A ∈ {0.25, 0.5, 1.0} (+WL1, WJR1; α 0.70 = 0.863 đã đo), TUNED | `flow_p5_final_h` | hai1108 ×2, nguyenmanh0404 | chạy 08:40 |

## 3. Nhật ký

- 08:05 token 18 acc (+10 acc 08:20); quota: tổng ≈ 377 h L4 (10 acc mới 20–30 h; acc cũ 0–19 h), yeshenhi 30 h T4.
- 08:30 keep-list REAP-64 suy ra bằng đọc range HTTP router của REAP-64 và FP8 chính thức: khớp chính xác 64 hàng / lớp × 40 lớp.
- 08:35 staging CPU: Qwen3.5-4B (9.34 GB) và REAP-64 FP8 (12.43 GB, 10.933B tham số = đúng model card) xong trong ~11 phút.
- 08:45 sự cố: runner tải output staging (≈ 18 GB) về container → hết hạn mức đĩa, hỏng test CPU / smoke đang chạy → xoá, thêm `ZC_OUT_PATTERN`
  cho `run_kernel.py` (output lớn ở lại Kaggle), chạy lại test.
- 09:05–09:20 smoke CPU (mô hình giả) qua: B00-S, M01 (DP2×TP2: 2 shard), T01 (ví dụ đổi theo schema), đường phán xử V01/S01, dữ liệu probe
  (eval_flow train); `test_grammar` PASS (2681/2681). I02: grammar nay gồm cả hàm cửa hàng do mô hình viết (`spec_u`).
- 09:10 saliency P02/P03 (namhunhh, 4×L4, bf16 72 GB device_map, sha256 so với HF): 340 prompt framework của DN train (240 bộ giải + 100 link).
- 09:22 đã khởi chạy toàn bộ P0 / P1 không phụ thuộc: 15 phiên GPU trên 10 acc. Phụ thuộc: V01 / S01 / A01 / A02 / H01 sau B00-S s0 (~10:50),
  P02 / P03 sau saliency (~10:30 → cắt CPU → chạy ~11:00–13:30).
- P3 (D01 / D02 / R01 / G01 / G02 / L01 / S02, huấn luyện 6–48 GPU-h): theo cổng của kế hoạch chỉ mở sau khi P1 có kết quả (~11:00–13:30);
  huấn luyện + đánh giá không kịp trong khung 7–8 h user đặt → chưa khởi chạy (đề xuất kèm báo cáo cuối).
- 09:31 **H01 dữ liệu xong** (bichthuy93, 31 phút; 1000 câu train trong 800 s trên 4×L4): B00-S trên `tr_dev1k` PA **0.955** (971 / 16 / 13; DN train
  có hàm toolkit riêng nên dễ hơn public chế độ H). Phát lại K: K1 0.917, K2 0.922, K4 0.923, K thích nghi (chạy) 0.955. 1,127 mục probe
  (12 % sai). Lưu `Process/runs/p5_results/h01tr/`.
- 10:00 **L02 α = 1.0** (V3A × 1.0 + WL1 + WJR1, chế độ H): container **0.827** (894 / 67 / 39), pub_noind 0.949; 74 phút, util 39 %.
  So α 0.70: 0.860 container / 0.863 ensemble → α 1.0 kém hơn (lặp lại kết quả WiSE-FT trước: 0.70 tốt hơn trọng số đầy đủ).
- 10:15 saliency P02/P03 xong (namhunhh, 2015 s; 1.14M token, REAP đủ (tính lại đầu ra expert), util GPU ~10 % vì device_map tuần tự);
  keep-list trùng REAP-64 cộng đồng 0.66 (E64) → 0.80 (E92). Cắt E92: 15.96 GB, 14.459B. E88 lần 1 lỗi STALE CODE (2 lần đẩy song song cùng
  acc ghi đè phiên bản code dataset) → đẩy lại.
- 10:13 **P02 / P03 chạy** (dinhthinh ×2): E88 15.46 GB / 13.955B, E92 15.96 GB / 14.459B tham số (gồm vision; < 15B), cắt từ FP8 chính thức theo
  keep-list saliency của tác vụ.
- 10:20 **L02 α = 0.5**: container **0.804** (875 / 71 / 54). Đường α: 0.5 → 0.804, 0.70 → 0.860, 1.0 → 0.827 (đỉnh gần 0.70).
- 10:25 bộ test CPU: 30 test PASS trong `run_all.sh` (chạm trần 3000 s vì máy bận) + `test_flow_hybrid` chạy riêng PASS → **31 / 31 PASS** trên code đã đẩy.
- 10:30 **L02 α = 0.25**: container **0.798** (861 / 63 / 76). Đường α đủ: 0.25 → 0.798, 0.5 → 0.804, 0.70 → 0.860, 1.0 → 0.827 (TUNED; đỉnh 0.70).
- 10:35 **B00-D** (ablation, Qwen gốc + thẻ luật cũ của mô hình tuned, chế độ H): container **0.758** (834 / 76 / 90; pub_noind 0.817), 93 phút,
  util 58 %. Phán xử (Qwen gốc, list / point / point+think) **bị cắt ở trần 1800 s** → file = bỏ phiếu. Chỉ 7 hàm cửa hàng qua kiểm (tuned: ~28).
  Phát lại K: K1 0.630, K2 0.637, K4 0.674, thích nghi 4→8 0.727, 4→16 (chạy) 0.758 → với Qwen gốc, nhiều mẫu hơn giúp rõ (+0.08 từ K4 lên 4→16).
- 10:40 **B00-S seed 0 = 0.753** (container = bỏ phiếu: phán xử Qwen gốc bị cắt ở trần 1800 s; 832 / 79 / 89; pub_ind 0.595, pub_noind 0.811;
  110 phút). Pha W: W1 bind 891 s + aliases **1427 s chạy trên 1 GPU** (3 GPU rảnh: chỗ chưa tối ưu, ghi PDF 2), 11 hàm cửa hàng qua kiểm, link 431 s.
  **A01 / A02** (phát lại offline log này): K1 0.632, **K2 0.648, K4 0.688** (cố định); thích nghi 2→8 0.713 (3.06 mẫu / câu), 4→8 0.730 (5.05),
  4→16 đồng thuận 0.75 0.733 (5.75), **4→16 (cấu hình) 0.753** (7.16) → với Qwen gốc, nhiều mẫu hơn vẫn có lợi; K thích nghi theo bất đồng tốt hơn K cố định.
- 10:42 đẩy S01 (9B tự kiểm, phán xử câu bất đồng — 261 / 1000 câu — **không trần thời gian**; hai1108), V02 (Qwen3.5-4B phán xử cùng 261 câu;
  duckzuykobip), H01 / H02 (probe: 1,127 mục train + 1,637 mục public từ log B00-S s0; bichthuy93). V01 (4B phán xử cả 1000 câu, list + point)
  đẩy khi M02 xong (cùng acc có Qwen3.5-4B).
- 10:50 **B00-S seed 1 = 0.755** (831 / 76 / 93; 12 hàm; phán xử bị cắt ở trần) → **BASELINE_ZS_P5_D1 = 0.754** (TB 2 seed, container; ghép cặp s0–s1
  +0.002, z 0.2: nhiễu nhỏ). B00-D − B00-S = +0.004 (z 0.5): thẻ luật của mô hình tuned gần như không giúp Qwen gốc.
  Đẩy held-out B00-S (`pub_ho400c`, `tr_ho400c`, eval_flow, không thẻ luật) làm mốc cho cổng held-out của finalist (hieuho669).
- 10:55 **B00-I = 0.818** (zero-shot, cùng giao thức với tuned 0.952, **có** `industry.py`; 881 / 63 / 56; pub_ind 0.829): so zero-shot ↔ tuned
  trên cùng luồng: **0.818 → 0.952 (+0.134)** khi có `industry.py`; **0.754 → 0.860 (+0.106)** ở chế độ H. Luật ngành viết tay đóng góp +0.064 cho
  zero-shot (ghép cặp vs B00-S, z 6.3) — chỉ để chẩn đoán, không dùng (vi phạm §1.2).
- 11:05 **M02 (Qwen3.5-4B)**: xem bảng `Process/scores_p5zs.csv`.
- 11:50 **V01 / V02** (Qwen3.5-4B phán xử log B00-S s0): V01 (mọi câu) ensemble **0.715** (Δ −0.038, z −3.9), list 0.650, point 0.602; V02 (chỉ 261 câu bất đồng) ensemble **0.715** (Δ −0.038, z −3.9). **Loại**: verifier 4B zero-shot làm hỏng câu đúng của bỏ phiếu. Cũng: **T01 0.764** (Δ +0.010, z 1.0, nhiễu), **M01 0.709** (Δ −0.045, z −3.6; phán xử 14B không khởi tạo được trên 1 GPU), **M02 0.717** (Δ −0.037, z −3.0).
- 11:55 **V02 với point+think** (4B suy nghĩ, chấm từng ứng viên, chỉ câu bất đồng): **0.769** (Δ +0.016 so với bỏ phiếu 0.753); ensemble 0.750, list 0.701, point 0.692.
- 12:00 **B00-S held-out** (eval_flow, không thẻ luật): pub_ho400c **0.529** (240/56/52 trên 348), tr_ho400c **0.851** (344/25/6). Tham chiếu tuned chế độ H (H0b, bỏ phiếu): 0.581 / 0.901.
- 12:05 **S01** (9B gốc tự phán xử 261 câu bất đồng, list / point / point+think, **không trần thời gian**): ensemble **0.776** (Δ +0.023 vs bỏ phiếu, **z 2.75**, 46 tốt / 24 kém); point+think 0.771 (z 1.8); list 0.727, point 0.729 (z −2.4). → phán xử của chính framework có lợi khi được chạy hết; trong B00 nó bị trần 1800 s cắt. Ứng viên mạnh nhất tới giờ (cần seed 2).
- 12:10–13:20 **P01 (REAP-64 cộng đồng) = 0.429** (sụp: 321 bỏ; pub_ind 0.18) — **P02 (REAP tác vụ, 88 expert, 13.96B) = 0.781** (Δ +0.027, **z 2.26**),
  **P03 (92 expert, 14.46B) = 0.773** (Δ +0.019, z 1.6); cả hai phán xử bị cắt ở trần → file = bỏ phiếu. Hiệu chỉnh tỉa trên prompt của chính
  framework (tiếng Việt + toolkit, DN train) quyết định: cùng kiến trúc, keep-list cộng đồng (code tiếng Anh) mất 0.33.
- **H01 / H02** (probe): AUC theo DN (GroupKFold) logistic **0.775**, MLP 0.763 (cổng AUC ≥ 0.70 qua), train PA 0.955 → 0.965 (logistic);
  nhưng **public không tăng** (bỏ phiếu 0.757 → 0.754 logistic, 0.723 MLP): probe học trên DN train (có hàm toolkit riêng) không chuyển sang
  DN mới chế độ H → **dừng**.
- 13:20 worker khởi động lại (mọi watcher / runner chết) → `resume.py` gắn lại 4 kernel. **User 13:20: chỉ tối ưu trên public, không chia tập khác.**
  Đẩy: P02 seed 2 + P02 phán xử không trần (dinhthinh), P02 + T01 + trần phán xử 5400 s (namhunhh) và thêm k_max 24 (hieuho669) sau khi cắt lại E88.
- 13:40 **Q01 (bộ giải BF16, DP2×TP2) = 0.764** (Δ +0.010, z 1.3; 243 phút so với ~110 của FP8) → FP8 đủ, BF16 không đáng thời gian.
  **S01 seed 2** (9B tự phán xử trên log B00-S s1): ensemble 0.762 (Δ +0.007, z 0.8); gộp 2 seed Δ +0.015 (89 tốt / 54 kém, z ≈ 2.5).
- 13:40 user: ghép các giải pháp tốt vào framework 0.952, thử mọi tổ hợp (vd. thay 9B tuned bằng MoE E88 + fine-tune 2–3 LoRA). Ràng buộc: LoRA
  V3A / WL1 / WJR1 chỉ áp được cho Qwen3.5-9B; E88 + 9B > 15B (không hợp lệ). Đẩy: C1 = 0.952 + T01, C2 = C1 + k24 (9B tuned, có
  `industry.py`), C3 = E88 zero-shot + `industry.py` + T01 + trần phán xử 5400 s, C4 = E88 + `industry.py`; chuẩn bị fine-tune LoRA S trên E88 bf16
  (cắt từ bf16 chính thức trong kernel, FSDP 4×L4, LoRA attention + linear-attention + shared expert; ghép LoRA vào FP8 để phục vụ).
- 13:45 **V02 seed 2** (4B phán xử, log B00-S s1): point+think 0.750 (Δ −0.005), ensemble 0.738 → gộp 2 seed point+think ≈ +0.005: **không xác nhận**, loại.
- 14:05 **C5** (framework 0.952, phán xử WJR1 thay bằng Qwen3.5-4B, log D1): ensemble 0.930 (−0.022, z −4.7), point+think 0.938 (−0.014, z −3.0) — kém WJR1 (0.952) → **loại**.
- 14:40 **P02 + phán xử không trần** (E88 tự phán xử câu bất đồng, 4618 s): ensemble **0.788** (Δ +0.007 vs bỏ phiếu P02 0.781, z 1.1); point+think 0.781, list 0.775, point 0.768. Tốt nhất zero-shot tới giờ: **0.788** (Δ +0.034 vs mốc 0.754). Phán xử MoE chậm (77 phút) → trong luồng cần trần ≥ 5400 s (P02x đang đo).
- 15:10 **C2** (framework 0.952 + T01 + k24): **0.913** (953 / 40 / 7; −0.039 vs 0.952, z −5.7) → kém; chờ C1 (chỉ + T01) để tách T01 với k24.
- 15:20 **C1** (framework 0.952 + T01): **0.905** (947 / 42 / 11; −0.047, z −6.6). Trước khi kết luận T01 hại bản tuned: chạy **C0 = `flow_p5_final` trên code hiện tại** (neo EXP-003) để loại trừ trôi code từ D1 (14/11).
- 15:25 **I02** (giải mã ràng buộc grammar toolkit): **0.675** (Δ −0.079, z −7.4), **337 phút** (ngân sách 20,400 s gần hết, phán xử bị bỏ), util GPU 8 % (mask grammar tính trên CPU) → **loại** (sai cả điểm lẫn tốc độ).
- 15:30 **New7** (gói user `New7_P5_Package.zip`): kế hoạch cập nhật theo thời gian đo `docs/20_NEW7_PLAN_UPDATED.md`. Cài phase 1 (N4-A cờ tắt mặc
  định). Độ phủ graph N4-A: 11 / 14 DN public có cạnh → N4-A 2 seed chạy (loinguyentran). **N1-A** (hoán vị dòng, phát lại CPU 2,838 ứng viên
  B00-S s0 trong 144 s, trung thực 0.995): **0 ứng viên đổi đáp án** → không tác dụng; N1-B đóng theo cổng. N7-A / N3-A / N6-A: prompt dựng
  offline (`new7_build.py`), một lượt vLLM chung (`src/new7_gen.py`), chấm CPU (`new7_score.py`).
- 16:05 **P02 seed 2 = 0.777** (Δ +0.023 vs B00-S, z 2.0) → **P02 xác nhận qua 2 seed**: TB 0.779 (Δ +0.025; z 2.3 / 2.0). Backbone zero-shot tốt nhất: MoE tỉa theo tác vụ 88 expert (13.96B).
- 16:05 **C3** (MoE E88 zero-shot + `industry.py` + T01 + phán xử 5400 s, framework 0.952 không LoRA) = **0.870** (928 / 58 / 14) vs B00-I (9B zero-shot + industry) 0.818. E4 lỗi (failover sang caotienminh1 trước khi chặn → STALE CODE) → chạy lại E4 + E3 seed 2 trên acc mới mxuanquy (30 h) sau khi ghép 3 LoRA tại đó.
- 16:25 **New7 lượt GPU (n7g, 4,240 item, ~25 phút trên 4×L4) — chấm ghép cặp trên cùng ứng viên của B00-S s0 / s1:**
  N7-A (bỏ nhóm có chứng chỉ đơn vị sai) 0.738 / 0.739 (z −2.5 / −2.8; Qwen gốc đặt tên đơn vị không nhất quán: 60 % chứng chỉ bị cờ);
  N3-A (phản ví dụ mẫu con + Qwen chọn) 0.722 / 0.726 (z −3.0 / −2.7); N6-A vote (thêm đáp án DuckDB của Qwen) 0.751 / 0.752 (z −0.4 / −0.7),
  N6-A agree 0.745 / 0.729. → **cả 3 loại** trên cả 2 seed; các nhánh sâu N7-B / N3-B / N6-B đóng theo cổng.
- 16:30 **C4** (MoE zero-shot + industry, phán xử 5400 s, không T01): xem bảng; so C3 (có T01).
- 16:45 **C0 (flow_p5_final trên code hiện tại) = 0.921, không phải 0.952** → tìm ra: template Qwen3.5 **mặc định suy nghĩ** khi không truyền
  `enable_thinking`; các lượt 0.952 chạy trên thư mục mô hình gộp (đường dẫn không có "qwen3") nên **mọi mẫu, kể cả 'direct', được suy nghĩ**;
  sửa nhận diện Qwen3 theo `model_type` (15/11 sáng, cho M01) đã tắt việc đó (bỏ phiếu D1 0.943 → C0 0.904). Thêm cờ `think_kw`
  (`legacy` = hành vi cũ, mặc định; `explicit`; `default` = để template suy nghĩ cho mọi mô hình). Hệ quả: C1 / C2 (−0.047 / −0.039 so 0.952) bị
  lẫn với lỗi này (so C0 cùng code: −0.016 / −0.008); mọi lượt zero-shot / MoE trước giờ dùng mẫu direct **không suy nghĩ** → có thể bị đánh
  giá thấp. Đẩy: C0L, C1L (lelynhan; xác nhận sửa), C3T (dinhthinh), P02T (hieuho669), B00T (phuongnehe) với `think_kw default`.
- 16:55 Đọc lại PDF 1 / PDF 2 (user): **dùng** — phán xử chỉ câu bất đồng; dòng rulebook trong prompt phán xử (công thức WJR1 → MJ1 / MJ2);
  nhiều mẫu (k16); dữ liệu paraphrase > kỹ thuật huấn luyện phức tạp (MS2 dùng toàn bộ dữ liệu V3A đã có paraphrase); so ghép cặp, ≥ 2 log;
  đẩy tuần tự từng acc (lỗi STALE CODE). **Tránh** — dòng rulebook trong prompt S; phán xử mọi câu; bộ chọn học được V; súp / DPO / RFT;
  k > 16 cho bản tuned. Phân tích offline: gộp 2 seed (≈ k ×2) chỉ +0.005 (P02: 0.786; B00: 0.759) dù oracle 942–946 / 1000 → nút thắt là
  **chọn ứng viên**, đúng PDF 1 §8 → ưu tiên phán xử có huấn luyện (MJ1 / MJ2) cho backbone MoE.
- 17:00 Chạy lại các lượt bị lỗi template (đường dẫn không có 'qwen3', code sau 09:05): C0 → C0L, C1 → C1L (đã đẩy), **C2 → C2L** (nguyenmanh0404), **E1 / E2 → E1L / E2L** (think mặc định như code cũ; caohuyen74, tự đẩy khi E1 / E2 xong). Không bị ảnh hưởng: B00, P02, C3 / C4, E3–E6 (đường dẫn có 'qwen3'), S01, V01 / V02, Q01, L02 (đẩy trước khi sửa).
- 17:25 **E1** (MoE E88 + LoRA MS1 4M token, framework 0.952 có industry, mẫu direct không suy nghĩ) = **0.857** ≈ C4 (zero-shot, 0.861) → MS1 chưa giúp ở chế độ không suy nghĩ; chờ E1L (suy nghĩ như code cũ) và E3 (đủ 3 LoRA).
- 17:30 **P02x** (P02 chế độ H + T01 + phán xử 5400 s, không luật viết tay) = **0.800** (887 / 87 / 26; Δ +0.046 vs B00-S, z 3.7) → **hệ chế độ H zero-shot tốt nhất** (vs P02 0.781, P02 + phán xử offline 0.788). 174 phút trên 4×L4 (phán xử 75 phút).
- 19:20 (worker khởi động lại ~17:15–19:05; đã `resume.py` + watcher + đẩy lại chuỗi bị mất). Kết quả: **P02xk** (+k24) 0.794 < P02x 0.800 (k24 không
  giúp, như PDF 1); **E3** (MoE + MS1 / ML1 / MJ1, industry) 0.856, **E5** 0.0 (link nạp mô hình từ mount 27 phút → bị cắt), **E6** (α 0.7, chế độ H)
  0.784. **Hai lỗi hạ tầng phát hiện** (đã sửa, commit FIX): (1) `lrl_gen` không khởi tạo được vLLM với MoE 15.5 GB (mặc định 256 seq / util 0.9)
  → **mọi lượt E88 không có thẻ W1 lẫn hàm cửa hàng** (P02, P02x, P03, C3, C4, E1, E3…); (2) mô hình W đọc thẳng từ mount. → Chạy lại
  P02xF, C3F (namhunhh), E3F, E4F (bichthuy93) với code đã sửa.
- 19:35 **Suy nghĩ (think_kw default) là đòn bẩy lớn:** B00T (9B zero-shot strict) **0.805** vs 0.754 (+0.051); C3T (MoE zero-shot + industry + T01,
  vẫn thiếu hàm do lỗi lrl_gen) **0.902** vs 0.870; C0L (framework 0.952, legacy) **0.942** ≈ D1 (khôi phục được); C1L (+T01) 0.943, C2L 0.935.
  → mọi tổ hợp tiếp theo: sửa lỗi + suy nghĩ: P02xFT / C3FT (caotienminh1, sau khi cắt E88), E3FT / E4FT (bichthuy93). Quota là ràng buộc:
  namhunhh 5.3 h (từ chối), caotienminh1 18.6 h, duckzuykobip 15.2 h; mxuanquy 30 h nhưng chưa accept rules (không có GPU).
- 21:00 **P02T** (P02 chế độ H + T01 + phán xử 5400 s + suy nghĩ; vẫn thiếu hàm do lỗi lrl_gen): xem bảng.
- 21:40 **E3FT** (MoE + MS1 / ML1 / MJ1 + industry + suy nghĩ + sửa lỗi: 15 hàm, bind OK) = **0.871** < E1L (MoE + MS1, suy nghĩ, chưa sửa) 0.891 < C3T (zero-shot + T01 + suy nghĩ) 0.902 → LoRA W (ML1 / MJ1 học trên E88 gốc) chưa giúp; chờ C3FT để tách tác dụng của hàm / thẻ.
- 15/11 22:52 E7 (MoE E88 + MS2 + ML1 + MJ2, thinking): **0.916** (953/37/10), z +12.6 vs B00-S — best MoE system (> C3T 0.902), below C1L 0.943.
- 15/11 23:32 C3FT: 0.905 (946/41/13), +0.003 vs C3T 0.902 (within noise); below E7 0.916.
- 16/11 00:04 E8 (E7 in mode H, MoE 13.96B + MS2/ML1/MJ2): **0.843** (893/50/57) — best mode-H system, level with P02T 0.841 (no hand-written industry functions).
- 16/11 00:24 P02xFT: 0.817 (891/74/35), < P02T 0.841. E2L / E4FT were refused at launch by the quota check (never ran); E8 covers the mode-H MoE role. All kernels done; no account holding the MoE models has quota for another run before the 03:00 stop.
- 16/11 02:25 (user 02:15: còn 12 h, chạy nhanh nhất, không tiết kiệm acc): phán xử trên ứng viên **gộp** nhiều lượt (oracle 946 → 959 khi gộp H0a + H0a2 + E8; bỏ phiếu gộp không tăng: 0.844–0.858): jph12 (lelynhan), jph1e (huyenco), jph12e (hai1108); thêm 2 lượt mốc nộp seed_base 1 / 2: h0as1 (thnliu), h0as2 (phatle1578).
- 16/11 03:06 (user: chạy mọi biến thể P5 chưa chạy còn trong ngưỡng thời gian): **T02** (kế hoạch: ví dụ tổng hợp nhất quán vòng, 2–5 h) — `src/t02_gen.py` + `predict_flow.prepare_t02`: Qwen (bộ giải) viết 12 cặp câu hỏi / chương trình cho MỌI cửa hàng từ bảng + rulebook (chỉ input), giữ cặp chạy được với toolkit hiển thị (chế độ H) và được bộ giải trả lời lại trùng đáp án (vòng), hybrid chèn 2 (T02a) / 3 (T02b) ví dụ cùng cửa hàng. Khác I3 (v2): I3 chỉ có ví dụ cho hàm cửa hàng (2–4 / 15 DN). Smoke CPU: 28 / 42 cặp chạy, 14 / 14 DN. Đẩy: t02a (linhmai24), t02as1 (duylnng, seed_base 1), t02b (lelynhan). Không chạy (vượt ngưỡng / không khả thi): D01 / D02 / R01 / G01 / G02 (huấn luyện 6–12 h tường + đánh giá > thời gian còn lại trước hạn nộp và Docker), S02 / L01 (đã có tương đương: WJR1 / MJ*, MS1 / MS2), Q02 (không có AWQ chính thức cho Qwen3.5-9B; tự lượng tử không kịp).
- 16/11 03:11 **jph12** (phán xử trên ứng viên gộp H0a + H0a2, chính sách container): ensemble **0.860** (907/47/46; ghép cặp vs H0a z 0.0), list 0.860, point 0.845, point+think 0.865 (z +0.7) → gộp 2 lượt cùng mô hình **không tăng** (mốc 0.863).
- 16/11 03:13 **jph1e** (phán xử trên gộp H0a + E8 MoE): ensemble **0.854** (z −0.8), list 0.853, point 0.850, point+think 0.863 → gộp với MoE không tăng.
- 16/11 03:15 **jph12e** (phán xử trên gộp H0a + H0a2 + E8): ensemble **0.863** (= mốc; z +0.4), list 0.862, point 0.844, point+think 0.858. **Kết luận: phán xử trên ứng viên gộp (2–3 lượt, cùng / khác mô hình) không vượt mốc 0.863** — oracle tăng nhưng bộ chọn không khai thác được; không ghép vào bản nộp (thêm 1 lượt S trong container mà không tăng điểm).
- 16/11 03:40 **LoRA cho MoE E88 — ghép cặp:** MS1 (4M tok) + industry: C3T → E1L −0.011 (z −0.8); MS1 + ML1 + MJ1: C3FT → E3FT −0.034 (z −2.1); **MS2 (16.7M tok) + ML1 + MJ2:** C3FT → E7 +0.011 (z +0.8), chế độ H P02xFT → E8 +0.026 (z +1.5) nhưng P02T → E8 +0.002. → LoRA MoE chỉ giúp khi đủ dữ liệu (MS2), mức tăng chưa có ý nghĩa; MoE + LoRA vẫn kém 9B tinh chỉnh (E7 vs C1L −0.027, z −2.2; E8 0.843 vs 0.863).
- 16/11 03:59 (user: biến thể chỉ cần thắng zero-shot 0.75 là ghép vào framework): ghép vào framework nộp chế độ H `flow_p5_final_h` (0.863) các biến thể vượt B00-S chưa từng ghép ở chế độ H: **ht01 / ht01s1** (T01, huyenco / phuongnehe), **hq01 / hq01s1** (Q01 BF16 DP2×TP2, hai1108 / nhatlong1103), **hn4a** (N4-A, mucnokieu). Đã ghép trước đó: P02 (→ E-series), S01 (trần phán xử), B00T (thinking; framework đã suy nghĩ), V02 4B (C5 0.930 < C0L), T02 (đang chạy).
- 16/11 04:23 h0as2 (mốc nộp, seed_base 2): **0.858** (907/49/46) → nhiễu giữa các lượt của mốc: 0.858–0.863.
- 16/11 04:45 h0as1 (mốc nộp, seed_base 1):  0.857 | 905/48/47 
- 16/11 05:40 **T02 lượt 1 hỏng** (t02a 0.861, t02as1 0.853, t02b 0.863 = mốc): pha T02 sinh 0 ví dụ — mô hình merged (đường dẫn không chứa 'qwen3') bị template bật suy nghĩ, 3000 token suy nghĩ không ra JSON (lỗi PDF 2 §11 #1 lặp lại ở mã mới). Sửa: `t02_gen` đặt `ZC_THINK_KW=explicit` (tắt suy nghĩ thật), thêm 6 ví dụ chương trình từ bank train (lọc hàm ẩn), log số đầu ra parse được + lỗi thực thi. Chạy lại: t02bf (lelynhan).
- 16/11 05:45 **Rà soát nguyên tắc "chỉ thay đúng thành phần" (user 16/11 05:30)** — so config từng lượt với baseline đúng loại:
  - **Sai, chạy lại:** mọi bản MoE fine-tune (E1, E1L, E3, E3FT, E6, E7, E8) dựng trên framework zero-shot nghiêm (lrl chỉ `compile_train` + `lrl_regen`, trần phán xử 5400 s) thay vì `flow_p5_final` (0.952) / `flow_p5_final_h` (0.863) → kết quả lẫn bộ thẻ luật với mô hình. Bản chạy lại (config sinh trực tiếp từ baseline, chỉ khác: mô hình + 3 LoRA MoE gộp FP8 nạp theo tên, `think_kw default` để giữ hành vi suy nghĩ như lượt 0.952): **e7rw7** (= 0.952, solver MS2 × 0.70 như V3A × 0.70), **e8rw7** (= 0.863, chế độ H), **e8r** (ablation trọng số MS2 × 1.0, chế độ H). Gộp LoRA trên bichthuy93 / caohuyen74 (CPU), tự đẩy khi gộp xong (`Process/tools/p5/after_merges.sh`).
  - **Gộp nhiều thay đổi (chỉ dùng làm thăm dò, không dùng để quy công):** P02T, P02x, P02xk, P02xFT, C3, C3T, C3FT (MoE zero-shot + T01 + trần phán xử + thinking), C2L (k_max + T01). Phép so zero-shot sạch: **P02 vs B00-S chỉ khác mô hình: 0.781 vs 0.753 (z 2.26)**; P03 0.773, P01 0.429, M01 0.709, M02 0.717 (M01: DP2×TP2 + KV fp8 là điều chỉnh tương thích để 14B vừa L4).
  - **Đúng:** C0L, C1L (T01 trên 0.952), L02, B00T, T01, Q01, N4-A, H01, A01/A02, các lượt ghép chế độ H (ht01, hq01, hn4a), T02 (trên 0.863), jph* (chỉ đổi bộ ứng viên).
- 16/11 05:44 ht01s1 (T01 ghép vào flow_p5_final_h, seed 1): **0.860** (909/49/42; vs H0a z 0.0) — trong dải mốc 0.857–0.863.
- 16/11 05:48 ht01 (T01 ghép vào flow_p5_final_h, seed 0): **0.857** → T01 ở chế độ H: 0.857 / 0.860 (TB 0.8585) vs mốc 0.857–0.863 (TB 0.860) → **không cải thiện, đóng**.
- 16/11 06:00 hn4a (N4-A ghép vào flow_p5_final_h): **0.848** (902/54/44; z −1.7) → kém mốc, **đóng**.
- 16/11 06:00 **MS3** (user: fine-tune MoE theo cách khác): công thức MS2 (v3a_mix, r 32, α 64, lr 2e-4, mb 4, eff 64, max-len 4608) + **LoRA trên 88 expert định tuyến** (PEFT target_parameters trên gate_up_proj / down_proj hợp nhất; router vẫn đóng băng), 8M token (vừa quota 1 phiên; MS2 16.7M). Mã: sft.py `ZC_LORA_EXPERTS`, moe_ft_job `--lora-experts`, merge_fp8 gộp ΔW từng expert (gate_up → gate_proj / up_proj); test: `Process/tools/p5/test_expert_lora_layout.py` (công thức = PEFT, sai 1e-8), `tests/test_merge_fp8_experts.py` (cos ΔW ≥ 0.996 sau FP8). Huấn luyện: fts3 (nguynduylm). Đánh giá dự kiến: e8r với solver MS3, chia 2 nửa cửa hàng trên 2 acc (`run_kaggle public_test@i/2`, mới).
- 16/11 06:20 fts3 hỏng sau 18 phút: PEFT ParamWrapper (LoRA trên expert hợp nhất) cấm `lora_dropout` ≠ 0 (MS2 dùng 0.05); test CPU dùng dropout mặc định 0 nên không bắt được. Sửa: sft.py đặt dropout 0 khi bật LoRA expert (điều chỉnh tương thích, ghi log), test cấu hình LoRA đúng như sft.py trên mô hình nhỏ (OK). Đẩy lại **fts3b** (6M token, vừa quota còn lại của nguynduylm), chuỗi `after_ms3.sh` khởi động lại.
- 16/11 07:16 **e7rw7** (= flow_p5_final 0.952, chỉ thay 9B + V3A×0.7 / WL1 / WJR1 bằng MoE E88 + MS2×0.7 / ML1 / MJ2): **0.927** (958/31/11; 17 hàm, link 1000, phán xử xong trong trần). Ghép cặp: vs D1 gốc 0.951 Δ −0.024 (z −2.1); vs C0L (0.952 chạy lại cùng mã) 0.942 Δ −0.015 (z −1.4); vs E7 cũ (framework zero-shot) +0.011 (z 1.1). → **MoE + LoRA kém 9B + LoRA khi mọi thành phần khác giữ nguyên**; phần thua của E7 cũ (0.916) chỉ ~1 / 3 do framework lệch.
- 16/11 07:17 **t02bf** (T02 sau sửa: 147 ví dụ cho 14/14 DN, round trip giữ 147/164): **0.777** (852/75/73; vs H0a z −8.1) → **T02 hại mạnh, đóng**. Ví dụ tự sinh (câu đơn giản: doanh thu hôm qua, số đơn tuần này…) thay 3 ví dụ train được truy hồi theo độ giống câu hỏi → mất ví dụ đúng dạng khó; kiểm vòng chỉ chứng minh nhất quán, không chứng minh ví dụ hữu ích (kế hoạch đã cảnh báo).
- 16/11 08:15 **Hoán đổi vai trò phán xử (trên framework 0.952, chỉ chạy pha phán xử trên ứng viên có sẵn, chính sách container):**
  - **jsj** = S / L / units của D1 (9B + V3A×0.7 / WL1) + **J = E88 + MJ2**: ensemble **0.945** (970/25/5; vs D1 0.951 −0.006 z −0.7; vs C0L 0.942 +0.003 z 0.35), point+think 0.947. → phán xử MoE **ngang** phán xử 9B (WJR1).
  - **jsq** = S / L / units của e7rw7 (E88) + **J = 9B + V3A×0.7 + WJR1**: ensemble **0.929** (vs e7rw7 0.927 +0.002 z 0.3; vs D1 −0.022 z −1.95). → đổi phán xử về 9B không cứu được; **phần thua của e7rw7 nằm ở bộ giải S (và/hoặc link) MoE**, không phải phán xử.
  - MS3 huấn luyện xong (fts3b: 6M token, LoRA cả 88 expert, grouped_mm mb 4, 4336 s ≈ 1.38k tok/s); chuỗi gộp + đánh giá chạy. rss (chỉ đổi S) đẩy 08:17 trên thnliu.
- 16/11 08:25 **Huỷ rss / rsl** (đổi riêng S hoặc riêng L sang MoE, phần còn lại 9B): cấu hình lai cần **9B + MoE 13.96B ≈ 23B tham số** → vượt giới hạn 15B, không thể nộp; chỉ là chẩn đoán, và jsq đã cho thấy điểm yếu nằm ở S / L của MoE. Tương tự jsj (9B S/L + MoE J, 0.945) chỉ là chẩn đoán. Tổ hợp **nộp được** chỉ có: toàn 9B (F, 9.4B) hoặc toàn MoE E88 (13.96B, mọi vai trò dùng chung một backbone + LoRA).
- 16/11 08:24 hq01s1 (Q01 BF16 ghép vào flow_p5_final_h, seed 1): **0.854** (902/48/50; z −0.9), thời gian 223 phút (×2 so với FP8). Chờ hq01 seed 0 để kết luận.
