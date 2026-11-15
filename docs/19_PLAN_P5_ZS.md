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
