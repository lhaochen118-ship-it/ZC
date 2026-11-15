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
