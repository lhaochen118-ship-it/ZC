# Execution Plan 0 — Đánh giá zero-shot các hướng thay / bổ sung mô hình cho framework DataMate

*Zalo AI Challenge 2026, track DataMate — lập 15/11/2026 (UTC). Tài liệu để một AI coding agent triển khai. Chưa có thí nghiệm nào
trong tài liệu này được chạy; mọi số là **[Đo]** (đã đo trong repo), **[TL]** (lấy từ tài liệu / model card) hoặc **[ƯL]** (ước lượng,
phải hiệu chỉnh bằng bước đo ngắn).*

**Nguồn hướng thí nghiệm.** Prompt nhắc `proposed_method_merged.pdf / .md`, nhưng file này **không có** trong repo hay container.
Tài liệu đề xuất có trong repo là `docs/proposed_method_1_b.md`; các hướng prompt liệt kê (MoE pruning, generator + verifier, recursive /
depth up-scaling, multi-LoRA, hidden-state probing, GRPO, test-time adaptation, IR / DSL, compression khác) trùng với tài liệu này.
Nếu bản hợp nhất có thêm hướng, bổ sung theo đúng khuôn §3.

---

## Phần 1 — Executive Summary

| Mục | Nội dung |
|---|---|
| Mục tiêu | Tìm nhanh nhất những thay đổi mô hình (thay backbone, thêm verifier, đổi kiến trúc, cơ chế suy luận) vượt `ZERO_SHOT_BASELINE` trong **cùng framework**, rồi fine-tune các hướng thắng để vượt 0.952 |
| Framework chuẩn | `configs/flow_p5_final.json` (kế hoạch 5): S = Qwen3.5-9B + LoRA **V3A × 0.70** (gộp); W1 viết hàm cửa hàng + gắn rulebook (`units_model: solver`); W2 linker = LoRA **WL1**; W3 phán xử = LoRA **WJR1** (chỉ câu có mẫu bất đồng, ensemble list / point / point+think, margin 0.5); toolkit `dm` + `industry.py`; FP8 online, pipeline, k 4 → 16 |
| `BEST_FINETUNED_BASELINE` | **0.952** [Đo]: tb phán xử lại offline 4 log bộ giải (0.949–0.958); lần chạy container D1 (`quynhtran35/zc-d1-1`) 0.952 (973 / 21 / 6) |
| `ZERO_SHOT_BASELINE` | **chưa từng đo** → EXP-001 / 002 (Qwen3.5-9B gốc, cả 3 LoRA tắt, mọi module khác giữ nguyên) |
| Số hướng | **9** (A–I); 2 hướng con bị loại có lý do (Recursive Transformer, nén dense / lượng tử hoá) |
| Biến thể giai đoạn zero-shot | **19 thí nghiệm** (15 cho điểm public; 3 tham chiếu vượt 15B chỉ để đo trần: A0, A4, B2) + tối đa **8 lần chạy cổng** (held-out, seed 2, chế độ H) |
| Biến thể cần huấn luyện (P3) | **8 gói**, chỉ chạy sau cổng quyết định |
| GPU-hours giai đoạn zero-shot | **≈ 139 GPU-h L4** [ƯL] = ≈ 35 giờ-phiên 4×L4 ≈ **76–87 h quota** Kaggle (hệ số đo 2.2, lập kế hoạch 2.5) |
| Thời gian hoàn thành (wall) | **≈ 7.5–8.5 h** từ T0 (có token + quota) cho toàn bộ zero-shot + cổng, với ≤ 10 phiên GPU đồng thời [ƯL] |
| Số acc cần | **11 acc L4** (22 phiên, đỉnh 10 phiên cùng lúc) + 2 acc chỉ cho phiên CPU (tải mô hình); acc T4 không dùng cho đánh giá chính |
| Kết luận định hướng | Chạy ngay: ZS baseline, MoE đã tỉa (có sẵn + tự tỉa theo dữ liệu bài toán), verifier 4B, phân rã LoRA. Song song nếu còn quota: probe ẩn, mở rộng chiều sâu (thăm dò), TTA in-context. Hoãn đến sau cổng: mọi gói huấn luyện (GRPO, LoRA phục hồi, block expansion có huấn luyện, IR / DSL) |

**Câu trả lời nhanh cho các câu hỏi của user (chi tiết ở các phần sau):**

| Câu hỏi | Trả lời |
|---|---|
| Có bao nhiêu hướng? | 9 hướng (A MoE pruning / scaling, B generator–verifier, C chiều sâu / recursive, D multi-LoRA, E probe trạng thái ẩn, F GRPO, G test-time adaptation, H IR / DSL, I mô hình thay thế / nén khác) |
| Mỗi hướng bao nhiêu biến thể? | A 3 + bước chuẩn bị tỉa (+2 tham chiếu A0, A4), B 3 (gồm 1 tham chiếu), C 2 (+1 huấn luyện), D 1 (+1 huấn luyện), E 2, F 0 zero-shot (+2 huấn luyện), G 1 (+1), H 1 phân tích CPU (+1), I 1 (+2 bị loại) |
| Chạy zero-shot ngay được? | EXP-001/002/003, A1 (REAP-64), A0 / A4 (tham chiếu), B1 / B2 / B3, D1, I1 (Gemma, có điều kiện) |
| Cần chuẩn bị thêm? | A2 / A3 (tính độ quan trọng expert + cắt), C2 (ghép lớp), E1 / E2 (trích trạng thái ẩn + probe), G1 (code tự sinh ví dụ), H0 (bộ biên dịch IR) |
| Cần huấn luyện? | C3, D2, F1 / F2, G2, H1, LoRA phục hồi cho backbone thắng (FT-101 / 102) |
| Bao nhiêu thí nghiệm cùng lúc? | đỉnh 10 phiên GPU (5 acc × 2) trong đợt 2–3; tối đa 4 bộ phán xử độc lập / phiên ở EXP-020 |
| Tận dụng 4 L4 / phiên? | mô hình vừa 1 GPU → DP4 (mỗi GPU một bản, claim DN động); mô hình 28–36 GB FP8 → TP2 × DP2; việc nhỏ trên log cố định → 4 worker độc lập |
| Tận dụng 2 phiên / acc? | luôn ghép 2 thí nghiệm độc lập cùng thời lượng lên một acc (bảng §5.3) |
| Thiếu GPU / quota? | ưu tiên P0 → P1; cắt P2; chạy chuỗi checkpoint; chuyển việc CPU sang phiên CPU; không khởi chạy việc không đủ quota (§4.6) |
| Bao lâu cho toàn bộ? | ≈ 7.5–8.5 h wall cho zero-shot + cổng; giai đoạn fine-tune P3 ≈ 12–24 h wall tùy hướng thắng (§5.5) |
| Tiêu chuẩn chọn? | Δ public ≥ +1.0 điểm so với ZS (2 seed, ghép cặp z ≥ 2) và held-out không giảm, hoặc ngang điểm nhưng nhanh hơn ≥ 1.3 lần; hợp lệ ≤ 15B kể cả adapter; ước tính chạy container ≤ 6 h trên 2×RTX 5090 (§6) |

---

## Phần 2 — Baseline Evaluation

### 2.1 Tra cứu lịch sử (đã làm khi lập kế hoạch)

| Lần chạy | Cấu hình | Có phải ZS của framework 0.952? |
|---|---|---|
| `pbase` (`anderson1202/zc-pbase-1`, v2) | Qwen gốc làm bộ giải **kênh 2** trong framework v2 **chế độ H** (ẩn `industry.py`); container 0.753 | **Không**: khác framework (chế độ H, W khác), vai trò kênh phụ |
| `p3pbase` (`hieuho669`) | tên dễ nhầm: thực ra là **P + WG1 × 0.35** (đã fine-tune) trên held-out | **Không** |
| Z-judge / zlk / zlb (kế hoạch 5, `docs/18_PLAN5_TWO_LORA.md`) | chỉ W3 hoặc W2 zero-shot, **S vẫn fine-tune** (judge zero-shot 0.947; S + link zero-shot 0.921) | **Không** (một phần) — dùng để đối chiếu phân rã |
| M0 (kế hoạch 1–2) | súp WF1–4 × 0.35 trên framework cũ: 0.934 | **Không** |

→ **Phải chạy mới** EXP-001 / 002. Kết quả cũ chỉ dùng để đặt kỳ vọng (Qwen gốc yếu hơn nhiều khi làm bộ giải: 0.753 trong framework v2).

### 2.2 Cấu hình ZS (thay đổi tối thiểu)

- Tạo `configs/flow_p5_zs.json` = bản sao `flow_p5_final.json`, chỉ đổi: `"links_lora": ""`, `"judge_lora": ""`, `_doc` mô tả.
  Mọi tham số còn lại giữ nguyên (k, nhiệt độ 0.7, số mẫu, ngưỡng, trần token, FP8, pipeline, `industry.py`).
- Chạy qua đường container (giống D1): `training/soup_job.py --model Qwen/Qwen3.5-9B --adapters NONE --runner predict
  --eval-configs configs/flow_p5_zs.json` (`soup_job.py` đã hỗ trợ `NONE` = mô hình gốc; không gộp S → bớt ≈ 8.5 phút).
- Hệ quả: W1 (`units_model: solver`) chạy trên trọng số gốc; W2 linker và W3 phán xử zero-shot trên trọng số gốc (đường mã đã kiểm ở
  Z-link / Z-judge kế hoạch 5).
- Seed: lần chạy 1 dùng seed mặc định (seed lấy mẫu cố định theo vòng / chỉ số mẫu, `--seed-base 0`); lần 2 đổi `seed-base` (thêm khoá
  `seed_base` vào config nếu `predict_flow.py` chưa đọc — thay đổi 1 dòng, có test).
- **Neo (anchor) EXP-003:** chạy lại đúng `flow_p5_final.json` (V3A × 0.70 + WL1 + WJR1) **trên code hiện tại** (đã có sửa `merge_sub
  first_decides` sau D1) để loại sai lệch do code thay đổi; mọi Δ so với 0.952 báo theo cả hai số (0.952 lịch sử và neo mới).

### 2.3 Ghi nhận bắt buộc cho mỗi lần chạy

`result.json` + `gpu_summary.json` (util, VRAM đỉnh, wall), thời gian từng pha (gộp, W1, W2, S, W3), tok/s từ dòng `[pipe]`, số câu
đúng / sai / bỏ, điểm theo nhóm (§6.1), lỗi nạp / thực thi / định dạng / timeout, commit hash code, config, seed, acc (mã), kernel.
Ghi vào `Process/scores_ep0.csv` + `Process/runs/ep0/<exp>/` (công cụ có sẵn: `automation/archive_v2.py`, `Process/tools/score_sets.py`).

### 2.4 So sánh

- `Δ_ZS(x) = PA(x) − ZERO_SHOT_BASELINE` (tb 2 seed khi có) — **thước đo chọn** ở giai đoạn này.
- `Gap_FT(x) = PA(x) − 0.952` — báo kèm để biết còn cách hệ tốt nhất bao xa.
- Không bao giờ dùng 0.952 làm mốc zero-shot.

---

## Phần 3 — Experiment Design

Quy ước chung cho mọi biến thể: giữ nguyên toolkit, prompt, pipeline, voting / selection / verification, tham số suy luận; chỉ đổi
thành phần ghi trong cột "Thay đổi". Đánh giá chính: public gốc 1,000 câu (đáp án chỉ để chấm). Cổng: held-out `pub_ho400c` +
`tr_ho400c` (cách đo held-out như h0b / h0c: bỏ phiếu, chưa phán xử).

### 3.A Hướng A — MoE lớn đã tỉa expert (scaling rồi thu nhỏ)

**Ý tưởng / giả thuyết.** Mô hình lớn hiểu câu hỏi nghiệp vụ tốt hơn (bằng chứng [Đo] WP5: tỉ lệ chương trình đúng khi sinh dữ liệu D3
trên câu train: Qwen3.6-27B 73.6 %, Qwen3.5-35B-A3B 64.4 %, bộ sinh 9B 51.8 %). Tỉa expert giữ năng lực tốt hơn tỉa FFN / cắt lớp
ở cùng kích thước ([TL] arXiv 2611.01133: Qwen3.6-35B-A3B tỉa 50 % expert giữ ≈ 92.8 % accuracy, ≈ 100 % pass@k, không healing). Active
≈ 3B → có thể nhanh hơn 9B dense. Chọn expert bằng prompt của chính framework (tiếng Việt + pandas + toolkit) giữ nhiều hơn bản tỉa bằng
code tiếng Anh. **Rủi ro:** pass@1 giảm làm phiếu yếu; đầu ra lặp / bị cắt khi tỉa nặng; MoE FP8 trên L4 chưa từng chạy trong repo;
kiến trúc có thể khác 9B ở template / think.

**Kích thước** (từ model card REAP-64 [TL]: 35.95B → 10.93B khi bỏ 192 / 256 expert × 40 lớp): ≈ 3.26M tham số / expert / lớp
→ ≈ 0.130B cho mỗi expert giữ thêm ở cả 40 lớp; phần không phải expert ≈ 2.59B. Giữ N expert: tổng ≈ 2.59 + 0.130·N (B).
N ≈ 73 → ≈ 12.1B; N ≈ 92 → ≈ 14.6B. **Bắt buộc đếm lại** bằng `tools/param_count.py` (đọc header safetensors) và bỏ vision encoder
nếu có (không dùng ảnh; giải phóng thêm ngân sách). Trần cho cả hệ: **≤ 14.8B kể cả adapter** (dư cho LoRA phục hồi).

| ID | Biến thể | Checkpoint | Tham số | Lượng tử | Tích hợp (4×L4) | Huấn luyện? | Zero-shot thật? |
|---|---|---|---|---|---|---|---|
| EXP-010 | **A0 tham chiếu** Qwen3.6-35B-A3B đầy đủ | HF `Qwen/Qwen3.6-35B-A3B` | 35.95B ✗ (không hợp lệ) | FP8 online | TP2 × DP2 (tiền lệ thầy D3 TP2) | không | có — chỉ đo trần năng lực |
| EXP-011 | **A1** REAP-64 có sẵn | `IAmTheRealGumbo/Qwen3.6-35B-A3B-REAP-64-experts` | 10.93B ✓ | FP8 online (≈ 11 GB) | DP4 | không | có |
| EXP-012 | ACAL: tính độ quan trọng expert (chuẩn bị A2 / A3) | từ A0 | — | bf16 (device_map 4 GPU, 72 GB / 90 GB) | 1 phiên | không (chỉ forward) | — |
| EXP-013 | **A2** tự tỉa ≈ 12B (N ≈ 73) | A0 + keep-list | ≈ 12.1B ✓ | FP8 online | DP4 | không | có |
| EXP-014 | **A3** tự tỉa ≈ 14.6B (N ≈ 92) | A0 + keep-list | ≈ 14.6B ✓ | FP8 online (≈ 15 GB, KV hẹp) | DP4, `--gpu-util 0.92` | không | có |

- **Thay đổi:** đường dẫn mô hình S và W (W dùng chung trọng số S như hiện nay), tắt mọi LoRA (cùng điều kiện với ZS baseline).
  Giữ nguyên toàn bộ phần còn lại. Nếu template chat / `<think>` khác 9B: chỉ sửa lớp tương thích trong `agent.py` (không đổi prompt).
- **ACAL (EXP-012):** độ quan trọng kiểu REAP S_j = trung bình trên token được route tới j của g_j(x)·‖f_j(x)‖ (hook trên khối MoE,
  tính f_j cho token được chọn); dự phòng: tần suất × trọng số cổng (chỉ cần router logits). Dữ liệu hiệu chỉnh: ≈ 1,024 prompt S +
  ≈ 512 prompt W1 / W2 / W3 **dựng bằng chính code framework trên DN train** (chạy CPU với `--fake-responder` để ghi prompt), ≤ 2k token
  — không dùng public / private. Đầu ra: `keep_N73.json`, `keep_N92.json` (vài KB) → mọi kernel sau cắt checkpoint tại chỗ
  (`training/moe_prune.py`, ≈ 10 phút) thay vì lưu checkpoint 25–30 GB (vượt giới hạn output kernel).
- **Kết quả kỳ vọng (giả thuyết, chưa đo):** A0 > ZS rõ; A3 ≥ A2 > A1. Nếu A0 ≤ ZS + 1.0 thì hướng A bị hạ ưu tiên (mô hình tỉa hiếm
  khi vượt bản đầy đủ).
- **Dừng sớm:** tốc độ đo trong 10 phút đầu cho thấy wall > 2.5 h → dừng, ghi lý do; lỗi nạp / kernel MoE FP8 → 1 lần sửa, không được
  → thử bf16 TP2 cho A1 / A2; vẫn lỗi → loại biến thể.

### 3.B Hướng B — Generator + Verifier

**Giả thuyết.** Câu sai còn lại tập trung ở **phiếu bị tách** (đáp án đúng có trong mẫu nhưng không thắng). Một verifier khác mô hình
sinh (khác lỗi) chọn tốt hơn phán xử cùng backbone. **Thiết kế ghép cặp:** mọi biến thể B chạy **chỉ pha phán xử** trên **cùng log bộ
giải** của EXP-001 (`soup_job.py --runner judge --judge-logs ...`, config khoá `judge_model`) → so sánh từng câu, rẻ.

| ID | Biến thể | Tham số hệ | Tích hợp | Zero-shot? |
|---|---|---|---|---|
| EXP-020a | **B1** Qwen3.5-4B gốc làm phán xử (list / point / point+think) | 9B + 4B = 13B ✓ | `judge_model` = 4B; 1 GPU / chế độ | có |
| EXP-020b | **B2 tham chiếu** Qwen3.6-27B-FP8 làm phán xử | 36B ✗ | TP2 trên 2 GPU cùng phiên | có — đo trần verifier |
| EXP-021 | **B3** bỏ phán xử (chỉ bỏ phiếu) | 9B | phát lại offline từ log EXP-001 (`tests/replay_judge.py`) | có, CPU |

- **Phiên EXP-020:** GPU0–1 = B2 (TP2), GPU2 = B1 list + point, GPU3 = B1 point+think → 4 worker độc lập trong một phiên.
- **Kỳ vọng:** biết phần lợi ích của verifier trong ZS (B3 vs EXP-001) và trần nếu verifier mạnh hơn (B2). B1 đáng giữ nếu ≥ +0.5 so
  với phán xử 9B gốc trên cùng log (tín hiệu sạch vì ghép cặp).
- **Dừng:** B1 ≤ phán xử 9B gốc → loại verifier mô hình riêng zero-shot (chuyển sang D2 / E: verifier dùng chung backbone, có huấn luyện).

### 3.C Hướng C — Recursive Transformers / Depth Up-Scaling / Block Expansion

**Phân loại bắt buộc.** Thay đổi cấu trúc mà chưa huấn luyện phục hồi **không** là bằng chứng kết luận.

| ID | Biến thể | Tham số | Zero-shot có nghĩa? | Vai trò |
|---|---|---|---|---|
| EXP-040 | **C1** block expansion khởi tạo đồng nhất (chèn 2 nhóm [3 DeltaNet + 1 attention] chép từ nhóm kề, `out_proj` / `down_proj` = 0) | ≈ 11B [ƯL] | **không** — đầu ra phải **trùng** 9B | kiểm tra đúng cài đặt (logits khớp, test CPU mô hình tí hon + 1 GPU); điều kiện cho FT-103 |
| EXP-041 | **C2** nhân đôi 2 nhóm giữa (không khởi tạo 0) — thăm dò | ≈ 11B [ƯL] | **thăm dò**, không kết luận | xem mô hình có chịu được thêm độ sâu không (nếu sập < ZS − 10 → cần huấn luyện dài hơn ước tính) |
| FT-103 | **C3** C1 + huấn luyện chỉ lớp mới (dữ liệu kiểu V3A) | ≈ 11B | — (P3) | thí nghiệm thật của hướng C |
| — | Recursive Transformer (Bae et al., ICLR 2025) | — | — | **Không chạy:** cần huấn luyện tiếp 15–60 tỷ token ([TL]) để phục hồi; vượt xa tài nguyên Kaggle |

- **Tích hợp:** `training/expand_depth.py` sinh checkpoint mới (config `num_hidden_layers`, `layer_types` giữ đúng mẫu 3:1) → vLLM nạp như
  9B. Qwen3.5-9B: 32 lớp, hidden 4096 [TL model card].

### 3.D Hướng D — Một backbone, nhiều LoRA chuyên biệt

Framework hiện tại **đã là** 1 backbone + 3 LoRA (V3A giải, WL1 linker, WJR1 phán xử). Zero-shot không áp dụng cho LoRA mới; thí nghiệm
zero-shot hữu ích là **phân rã** để biết LoRA nào mang lợi ích (định hướng fine-tune cho backbone mới).

| ID | Biến thể | Ghi chú |
|---|---|---|
| EXP-030 | **D1** S gốc + WL1 + WJR1 (chỉ tắt LoRA giải V3A) | `--adapters NONE` + `flow_p5_final.json`; lưu ý: WL1 / WJR1 học trên trọng số S đã gộp V3A → áp lên trọng số gốc có lệch phân phối; kết quả là cận dưới phần đóng góp của W |
| FT-106 | **D2** adapter mới: verifier LoRA (Adapter B) / giải thích business rules (Adapter C) | P3, cần dữ liệu huấn luyện từ log train |

Phân rã đầy đủ = ZS (không LoRA) · D1 (chỉ W) · neo EXP-003 (cả ba) · số zero-shot W đã có (Z-judge 0.947, Z-link 0.921 với S fine-tune).

### 3.E Hướng E — Bộ kiểm tự soi từ trạng thái ẩn (probe)

**Giả thuyết** ([TL] Orgad et al., ICLR 2025): trạng thái ẩn ở token đáp án chứa tín hiệu đúng / sai mạnh hơn đặc trưng bề mặt; probe
phải học trên chính bài toán. Không đổi trọng số mô hình → **tương thích giai đoạn zero-shot**, nhưng probe cần **huấn luyện tối thiểu**
(logistic / MLP nhỏ trên CPU) → phân loại riêng "zero-shot + bộ chọn học được".

| ID | Biến thể | Cách làm |
|---|---|---|
| EXP-050 | **E1** dữ liệu probe | chạy cấu hình ZS trên 1,000 câu train dev (DN train, không held-out) + ghi hidden state lớp {16, 24, 32} tại token cuối chương trình / đáp án (một lượt prefill, prefix cache) → nhãn từ thực thi so với đáp án train |
| EXP-051 | **E2** áp probe | trích hidden state cho ứng viên trong log EXP-001 (public) → thêm điểm probe làm đặc trưng cho bộ chọn / ngưỡng bỏ trả lời, phát lại offline; so với bỏ phiếu + phán xử ZS trên cùng log |

- **Dừng:** AUC probe trên tập train tách riêng < 0.70, hoặc phát lại public không tăng ≥ +0.5 → dừng.

### 3.F Hướng F — GRPO / học tăng cường thưởng thực thi

Không có dạng zero-shot. **Chỉ P3**, sau khi chọn backbone: F1 = chạy thử 150 bước GRPO (LoRA, TRL + vLLM) để đo thời gian / bước và đường
thưởng (không kết luận điểm); F2 = huấn luyện đủ (thưởng 1 nếu đúng, 0.1 nếu chạy được — [TL] Arctic-Text2SQL-R1; dữ liệu câu train
+ paraphrase train có đáp án, bỏ `tr_ho`). DPO offline đã bị loại trước đây (M4); GRPO là on-policy, khác cơ chế.

### 3.G Hướng G — Thích nghi theo từng DN lúc suy luận (TTA)

| ID | Biến thể | Zero-shot? | Ghi chú |
|---|---|---|---|
| EXP-060 | **G1** trước khi trả lời một DN: mô hình tự sinh cặp câu hỏi → chương trình **trên bảng của DN đó** từ khung câu hỏi train (`retrieval.skeleton`), chạy thử, giữ cặp qua kiểm tra vòng (diễn đạt lại → sinh lại → cùng kết quả), dùng làm ví dụ few-shot thay ví dụ DN khác | có (không cập nhật trọng số) | cần ≈ 1 ngày code + test; tốn thêm ≈ 15–25 % thời gian suy luận [ƯL] |
| FT-107 | **G2** LoRA riêng mỗi DN trên các cặp tự sinh ([TL] Akyürek et al., ICML 2025) | không | **cần user quyết:** cập nhật trọng số từ dữ liệu test không nhãn có thể trái nguyên tắc "không train trên private test" |

### 3.H Hướng H — Kế hoạch truy vấn có kiểu (IR / DSL)

| ID | Biến thể | Ghi chú |
|---|---|---|
| EXP-070 | **H0** phân tích phủ (CPU): bao nhiêu % chương trình đã kiểm trên câu train diễn đạt được bằng IR đề xuất (đại lượng, bộ lọc, kỳ, nhóm, hàm gộp, top-k, so sánh) | quyết định có đáng xây compiler không (ngưỡng ≥ 80 %) |
| FT-108 | **H1** compiler + SFT sinh IR (+ sinh có ràng buộc) | P3; thay đổi kỹ thuật lớn nhất |

Zero-shot few-shot IR có thể thử sau khi có compiler, nhưng mô hình chưa học DSL nên không kết luận được → xếp P3.

### 3.I Hướng I — Mô hình thay thế ≤ 15B và nén khác

| ID | Biến thể | Ghi chú |
|---|---|---|
| EXP-016 | **I1** Gemma 4 12B QAT (đã có dataset `nhatlong1103/zc-m-gemma4-12b-qat`) làm backbone, cùng framework | có điều kiện: chỉ chạy nếu kiểm tra tương thích template / think qua (CPU + smoke 5 phút); tiền lệ M9 / QĐ4 trên framework cũ yếu hơn Qwen |
| EXP-015 | **A4 tham chiếu** Qwen3.6-27B-FP8 dense | không hợp lệ (27B); chỉ chạy nếu A0 cho thấy trần cao để biết dense hay MoE gần trần hơn |
| — | Lượng tử hoá 27B / 35B để "vừa" 15B | **loại:** BTC đếm tham số, không đếm byte |
| — | Tỉa chiều rộng / cắt lớp Qwen3.6-27B dense xuống 15B | **loại:** cắt ≈ 45 % cần healing hàng chục tỷ token |

---

## Phần 4 — Kaggle Resource Strategy

### 4.1 Tài khoản (mã định danh an toàn)

- 19 acc có token (15/11): **18 acc 4×L4** + 1 acc chỉ T4. Mã `ACC-01 … ACC-18` gán tại T0 theo **quota còn lại giảm dần**
  (`python3 automation/quota.py`); bảng ánh xạ mã ↔ username lưu ở `~/.kaggle/acc_map.json` (chmod 600, ngoài repo). `ACC-T4` = acc T4.
- Ảnh chụp quota gần nhất **[Đo] 15/11 03:15 UTC, đã cũ** (sau đó ≈ 27 kernel đã chạy): tổng còn ≈ 252 h trên 18 acc L4; 3 acc ≈ 30 h,
  1 acc ≈ 23 h, 6 acc 12–19 h, 5 acc 6–11 h, 3 acc < 6 h. **Phải đo lại tại T0.** Làm mới quota: 21/11 00:00 UTC.
- Token: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<username>)` (cách repo đang dùng) hoặc `~/.kaggle/access_token`; chmod 600; xác minh bằng
  `kaggle competitions list` cho từng acc; **không** ghi token vào code, notebook, log, báo cáo, Git; lọc token khỏi thông báo lỗi.
- Acc cần accept rules `gemma-4-developer-agent` để có L4 (lỗi push "accept this competition's rules" → báo user).

### 4.2 Giới hạn và chi phí đo được

| Mục | Giá trị |
|---|---|
| Phiên / acc | 2 GPU commit (mỗi phiên 4×L4, 22.5 GB / GPU dùng được, không NVLink, 48 CPU) + 5 CPU commit |
| Phiên commit | ≤ 12 h (runner đặt hạn chót 12 h − 40 phút, checkpoint, `_skipped_jobs.json`) |
| Quota | 30 h / tuần / acc; **4×L4 tốn 2.0–2.2 h quota mỗi giờ tường [Đo]** (lập kế hoạch × 2.5) |
| Internet | phiên L4 **không** có internet → mô hình phải nằm trong Kaggle Models / dataset / output kernel; phiên CPU có internet |
| Cài đặt | ≈ 7 phút (wheels vLLM 0.32 từ `nhatlong1103/zc-assets`) |
| D1 (cấu hình 0.952, public) | wall 5,270 s: gộp S 506 s, W1 + W2 ≈ 15 phút, S 1,685 s (DP4, ≈ 430 tok/s / GPU), gộp WJR1 456 s, W3 944 s; util GPU 28–42 %, VRAM đỉnh 19.9 GB |

### 4.3 Chiến lược song song

| Loại việc | Cách dùng 4 L4 | Lý do |
|---|---|---|
| Framework đầy đủ, mô hình ≤ 15 GB FP8 (9B, A1, A2, A3, C2, I1) | **DP4**, claim DN động (`--dp 4`, `--pipeline`) | đường đã đo; mỗi GPU một bản |
| Mô hình 28–36 GB FP8 (A0, A4, B2) | **TP2 × DP2** | không vừa 1 GPU; tiền lệ TP2 (QĐ2). Cần kiểm `predict_flow --dp 2` + `--tp 2` (bổ sung nếu chưa có) |
| Việc chạy trên log cố định (EXP-020, EXP-051) | **4 worker độc lập**, mỗi GPU một biến thể | tiền lệ Z-judge 4 biến thể / phiên |
| Tính độ quan trọng expert (EXP-012) | 1 tiến trình, `device_map` qua 4 GPU (bf16 72 GB) | chỉ forward, không sinh |
| Huấn luyện (P3) | FSDP 4 GPU (`training/full_job.py`), checkpoint mỗi 30 phút | đo 13M token ≈ 2.3 h trên 9B |

Không gom 2 thí nghiệm khác mô hình vào một phiên DP2 + DP2 ở giai đoạn này: lợi ích chưa đo, rủi ro OOM / tranh CPU; util thấp chủ
yếu do các pha W tuần tự (đã có trong framework, không phải mục tiêu thí nghiệm). Ghi thành ý tưởng tối ưu cho sau.

### 4.4 Chuẩn bị mô hình (phiên CPU có internet)

| ID | Việc | Kích thước | Cách |
|---|---|---|---|
| STG-01 | Kiểm Kaggle Models có sẵn Qwen3.6-35B-A3B / Qwen3.5-4B / Qwen3.6-27B-FP8 không (`kaggle models list -s ...`) | — | nếu có → dùng `--ms`, bỏ STG tương ứng |
| STG-02..05 | Qwen3.6-35B-A3B bf16 | ≈ 72 GB | **4 kernel CPU**, mỗi kernel tải một nhóm shard safetensors (≤ 19 GB, dưới giới hạn output kernel ≈ 20 GB — kiểm tại STG-01) + manifest sha256; kernel GPU gắn cả 4 output (`--ks`) và nối thư mục bằng symlink |
| STG-06..07 | REAP-64 | 21.9 GB | 2 kernel CPU |
| STG-08 | Qwen3.5-4B | ≈ 9 GB [ƯL] | 1 kernel CPU |
| STG-09..10 | Qwen3.6-27B-FP8 | ≈ 28 GB [ƯL] | 2 kernel CPU (chỉ khi chạy B2 / A4) |

Công cụ mới: `automation/stage_model.py` (sinh kernel CPU tải một phần repo HF, ghi manifest). Thời gian [ƯL] 20–45 phút / kernel (tải +
lưu output); chạy song song trên các slot CPU của 2 acc (10 slot).

### 4.5 Giám sát, treo, retry

- Watcher nền (`Process/tools/v2/watch_loop.sh` → `watch_done.py`, `run_in_background`, timeout 7,200,000 ms), không poll tay.
- Chưa có output sau **5 phút (L4 / CPU)** hoặc **10 phút (T4)**: đọc trạng thái (`QUEUED` / `RUNNING`), log cài đặt / tải mô hình / nạp
  checkpoint / biên dịch; **không** coi thiếu log là treo (API chỉ lộ log gần lúc xong). QUEUED > 8 phút (L4) → xoá, đẩy lại.
- Treo thật (log không đổi > 20 phút trong pha sinh, tok/s = 0): `kaggle kernels delete -y <owner>/<slug>` (từng lệnh một), sửa nguyên
  nhân, đẩy lại; tối đa **4 lần**; không lặp lại cấu hình lỗi y nguyên. Không ngắt được → báo user ngắt tay.
- Đọc trạng thái chỉ từ dòng `has status "KernelWorkerStatus.X"` (lỗi 503 từng bị đọc sai thành `error`); 429 → `ZC_POLL_S` 150–180 s.

### 4.6 Khi thiếu GPU / quota

1. Trước mỗi lần khởi chạy: `launch_many.py` kiểm `est_h × 2.5 ≤ quota còn lại`; acc < 6 h chỉ nhận việc ngắn (EXP-020, EXP-051, cổng held-out).
2. Không đủ: (a) chuyển sang acc khác; (b) cắt P2 trước P1; (c) giảm đánh giá xuống tập con sàng lọc đã dùng trước đây (`pub_ho400c`
   + một nửa public theo DN) chỉ cho biến thể P2; (d) chia việc có checkpoint qua nhiều phiên.
3. Vẫn không đủ → **không khởi chạy**, ghi lý do vào `handoff/STATUS.md` và đề xuất phương án rẻ hơn.
4. GPU hết hẳn: việc CPU (H0, B3, huấn luyện probe, phân tích log) vẫn chạy; việc GPU vào hàng đợi chờ 21/11.
5. Fallback T4 (2×16 GB, fp16): chỉ cho B1 (4B) hoặc trích hidden state 9B; không dùng T4 cho framework đầy đủ (FP8 không có trên T4,
   bf16 giả lập chậm ≈ 7×).

---

## Phần 5 — Detailed Execution Schedule

### 5.1 Bảng tổng hợp thí nghiệm (bắt buộc)

GPU-hours = số GPU × giờ tường. Quota ≈ giờ tường × 2.2 (đo) — cột cuối dùng × 2.5 để lập kế hoạch. Wall bao gồm cài đặt, nạp, gộp,
lưu output. Nguồn wall: D1 [Đo] cho mô hình 9B; mô hình mới [ƯL] → hiệu chỉnh bằng tok/s 10 phút đầu.

| Experiment ID | Hướng | Biến thể | GPUs | Accounts | GPU-hours | Wall-clock | Priority | Quota (× 2.5) |
|---|---|---|---|---|---|---|---|---|
| EXP-000 | Chuẩn bị | token, quota, test, preflight, config, đếm tham số | CPU | local | 0 | 0.75 h | P0 | 0 |
| STG-01..10 | Chuẩn bị | tải mô hình (CPU, internet) | CPU | ACC-12, ACC-13 (slot CPU) | 0 | 0.5–1 h | P0 | 0 |
| EXP-001 | Zero-shot baseline | Qwen3.5-9B gốc, seed 0 | 4 L4 | ACC-01 | 6.0 | 1.5 h | P0 | 3.75 |
| EXP-002 | Zero-shot baseline | Qwen3.5-9B gốc, seed 1 | 4 L4 | ACC-01 | 6.0 | 1.5 h | P0 | 3.75 |
| EXP-003 | Neo fine-tuned | `flow_p5_final` trên code hiện tại | 4 L4 | ACC-02 | 6.4 | 1.6 h | P0 | 4.0 |
| EXP-030 | Multi-LoRA (D1) | S gốc + WL1 + WJR1 | 4 L4 | ACC-02 | 6.4 | 1.6 h | P1 | 4.0 |
| EXP-010 | MoE pruning (A0, tham chiếu) | Qwen3.6 35B-A3B đầy đủ, TP2 × DP2 | 4 L4 | ACC-03 | 8.0 | 2.0 h [ƯL] | P1 | 5.0 |
| EXP-011 | MoE pruning (A1) | REAP-64 10.93B, DP4 | 4 L4 | ACC-03 | 6.0 | 1.5 h [ƯL] | P1 | 3.75 |
| EXP-012 | MoE pruning (chuẩn bị) | độ quan trọng expert, keep-list N73 / N92 | 4 L4 | ACC-04 | 6.0 | 1.5 h [ƯL] | P1 | 3.75 |
| EXP-013 | MoE pruning (A2) | tự tỉa ≈ 12B | 4 L4 | ACC-06 | 6.4 | 1.6 h [ƯL] | P1 | 4.0 |
| EXP-014 | MoE pruning (A3) | tự tỉa ≈ 14.6B | 4 L4 | ACC-06 | 6.4 | 1.6 h [ƯL] | P1 | 4.0 |
| EXP-020 | Generator / Verifier (B1, B2) | 9B + 4B phán xử; 27B phán xử tham chiếu (trên log EXP-001) | 4 L4 (4 worker) | ACC-05 | 4.0 | 1.0 h [ƯL] | P1 | 2.5 |
| EXP-021 | Generator / Verifier (B3) | bỏ phán xử, phát lại offline | CPU | local | 0 | 0.2 h | P1 | 0 |
| EXP-015 | Scaling (A4, tham chiếu) | Qwen3.6 27B FP8 dense | 4 L4 | ACC-07 | 10.0 | 2.5 h [ƯL] | P2 (điều kiện) | 6.25 |
| EXP-016 | Mô hình thay thế (I1) | Gemma 4 12B QAT | 4 L4 | ACC-05 | 6.0 | 1.5 h [ƯL] | P2 (điều kiện) | 3.75 |
| EXP-040 | Depth (C1) | khối đồng nhất: kiểm logits | CPU + chung phiên EXP-041 | ACC-04 | 0 | 0.3 h | P2 | 0 |
| EXP-041 | Depth (C2) | nhân đôi 2 nhóm, thăm dò | 4 L4 | ACC-04 | 6.4 | 1.6 h [ƯL] | P2 | 4.0 |
| EXP-050 | Probe (E1) | ZS trên 1,000 câu train + hidden state | 4 L4 | ACC-07 | 7.0 | 1.75 h [ƯL] | P2 | 4.4 |
| EXP-051 | Probe (E2) | trích hidden state log EXP-001 + phát lại | 4 L4 (4 worker) | ACC-08 | 3.0 | 0.75 h [ƯL] | P2 | 1.9 |
| EXP-060 | TTA (G1) | ví dụ tự sinh theo DN (không cập nhật trọng số) | 4 L4 | ACC-09 | 7.0 | 1.75 h [ƯL] | P2 (sau ≈ 1 ngày code) | 4.4 |
| EXP-070 | IR / DSL (H0) | phân tích phủ chương trình train | CPU | local | 0 | 2 h (code + chạy) | P2 | 0 |
| EXP-090..093 | Cổng held-out | tối đa 4 biến thể sống sót: `pub_ho400c` + `tr_ho400c` | 4 L4 mỗi cái | ACC-08..09 | 4 × 3.0 = 12.0 | 0.75 h | Cổng | 4 × 1.9 |
| EXP-094..096 | Cổng seed 2 | tối đa 3 biến thể sống sót, seed 1 | 4 L4 mỗi cái | ACC-10..11 | 3 × 6.0 = 18.0 | 1.5 h | Cổng | 3 × 3.75 |
| EXP-097 | Cổng chế độ H | biến thể thắng với `flow_p5_final_h` (ẩn `industry.py`) | 4 L4 | ACC-11 | 8.0 | 2.0 h [Đo: H k16 1.8–2.2 h] | Cổng | 5.0 |
| **Tổng zero-shot + cổng** | | | | **11 acc L4** | **≈ 139** | **≈ 7.5–8.5 h (makespan)** | | **≈ 87** |

Các gói huấn luyện (P3, không thuộc giai đoạn zero-shot):

| ID | Hướng | Nội dung | GPUs | GPU-hours [ƯL] | Wall [ƯL] | Điều kiện bắt đầu |
|---|---|---|---|---|---|---|
| FT-101 | Backbone thắng | LoRA giải phục hồi (công thức V3A) trên backbone thắng | 4 L4 FSDP | 12–28 | 3–7 h (MoE trên HF chậm hơn dense 2–3 lần, cần đo) | backbone qua cổng §6.3 |
| FT-102 | Backbone thắng | huấn luyện lại WL1, WJR1 trên backbone thắng | 4 L4 × 2 phiên song song | 2 × 6 | 1.5 h | sau FT-101 |
| FT-103 | Depth (C3) | +8 lớp, chỉ huấn luyện lớp mới | 4 L4 FSDP | 12 | 3 h + đánh giá 1.6 h | EXP-040 qua (logits khớp) |
| FT-104 | GRPO (F1) | chạy thử 150 bước | 4 L4 | 8 | 2 h | backbone đã chọn |
| FT-105 | GRPO (F2) | huấn luyện đủ | 4 L4, chuỗi checkpoint | 32–80 | 8–20 h | F1 cho đường thưởng tăng |
| FT-106 | Multi-LoRA (D2) | verifier LoRA / adapter giải thích | 4 L4 | 6–8 / adapter | 1.5–2 h | sau E2 / B |
| FT-107 | TTA (G2) | LoRA theo DN | 4 L4 | trong lúc suy luận | +? | **user quyết** về nguyên tắc dữ liệu test |
| FT-108 | IR / DSL (H1) | compiler + SFT IR | 4 L4 | 10–12 | 3 h + 2–3 ngày code | H0 phủ ≥ 80 % |

### 5.2 Phụ thuộc

```
EXP-000 ──> EXP-001, 002, 003, 030 (không cần tải mô hình)
STG-02..05 ──> EXP-010, EXP-012 ──> EXP-013, EXP-014
STG-06..07 ──> EXP-011
STG-08 (+09..10) + EXP-001 (log bộ giải) ──> EXP-020
EXP-001 (log) ──> EXP-021 (CPU), EXP-051
EXP-050 ──> huấn luyện probe (CPU) ──> EXP-051 phát lại
EXP-040 (logits khớp) ──> EXP-041 (cùng phiên), FT-103
Mọi biến thể P1 / P2 ──> cổng EXP-090..097 ──> quyết định ──> FT-101.. (P3)
```

### 5.3 Phân bổ acc / phiên theo thời gian (T0 = có token + quota đã đo)

| Thời điểm (từ T0) | Phiên 1 | Phiên 2 | Ghi chú |
|---|---|---|---|
| 0:00–0:45 | — | — | EXP-000 local (test 30–40 phút, preflight); STG-01..10 trên slot CPU ACC-12 / ACC-13 |
| **ACC-01** 0:45–2:15 | EXP-001 (ZS seed 0) | EXP-002 (ZS seed 1) | P0 |
| **ACC-02** 0:45–2:25 | EXP-003 (neo) | EXP-030 (D1) | P0 / P1 |
| **ACC-03** ≈ 1:45–3:45 | EXP-010 (A0, TP2 × DP2) | EXP-011 (A1) | chờ STG; A1 có thể bắt đầu sớm hơn (STG-06..07 nhỏ) |
| **ACC-04** ≈ 1:45–3:15 | EXP-012 (độ quan trọng expert) | EXP-040 + EXP-041 (C1, C2) | C là P2: chỉ chạy nếu quota T0 ≥ tổng P1 + 20 % |
| **ACC-05** ≈ 2:15–3:15 | EXP-020 (B1 + B2, 4 worker) | EXP-016 (I1, nếu kiểm tương thích qua) | sau khi EXP-001 có log |
| **ACC-06** ≈ 3:15–4:55 | EXP-013 (A2) | EXP-014 (A3) | sau EXP-012 |
| **ACC-07** ≈ 2:30–5:00 | EXP-050 (E1) | EXP-015 (A4, chỉ nếu A0 ≥ ZS + 3) | P2 |
| **ACC-08** ≈ 4:30–5:30 | EXP-051 (E2) | EXP-090 (cổng held-out #1) | |
| **ACC-09** ≈ 5:00–6:45 | EXP-091 (held-out #2) | EXP-060 (G1, nếu code xong) | |
| **ACC-10** ≈ 5:00–6:30 | EXP-094 (seed 2 #1) | EXP-095 (seed 2 #2) | |
| **ACC-11** ≈ 5:30–7:30 | EXP-096 (seed 2 #3) | EXP-097 (chế độ H của biến thể thắng) | |
| 7:30–8:30 | — | — | tổng hợp, kiểm định ghép cặp, báo cáo quyết định; dự phòng retry / hàng đợi |

**Đỉnh đồng thời:** ≈ 10 phiên GPU (2:15–3:45). **Makespan:** đường găng STG-02..05 (≈ 1 h) → EXP-012 (1.5 h) → EXP-014 (1.6 h) → cổng
(≈ 1.5 h) + tổng hợp ≈ 7.5 h; + 1 h dự phòng hàng đợi / retry → **≈ 8.5 h**.

### 5.4 Timeline tuyệt đối (ví dụ T0 = 15/11 09:00 UTC)

| Mốc | UTC |
|---|---|
| Chuẩn bị xong, P0 chạy | 09:45 |
| ZS baseline + neo có kết quả | 11:15–11:30 |
| Tải mô hình xong; A0, A1, ACAL chạy | ≈ 10:45 |
| Verifier (B) xong | ≈ 12:15 |
| A2 / A3 xong | ≈ 13:55 |
| Cổng held-out / seed 2 / chế độ H xong | ≈ 16:30 |
| Báo cáo quyết định | ≈ 17:30 |

**Ràng buộc lịch:** container đóng băng **16/11 04:00 UTC**; hạn nộp **16/11 23:59 GMT+7**. Giai đoạn zero-shot kịp trước đóng băng; giai đoạn
fine-tune (≥ 12 h) vượt mốc đó → **user quyết** có tiếp tục sau 16/11 (quota mới 21/11) hay không. Bản nộp hiện tại không đổi trong
giai đoạn này.

### 5.5 Tổng tài nguyên

| Giai đoạn | GPU-hours | Giờ-phiên 4×L4 | Quota (× 2.2 / × 2.5) | Wall |
|---|---|---|---|---|
| P0 | 18.4 | 4.6 | 10 / 12 | 1.6 h |
| P1 | 43.2 | 10.8 | 24 / 27 | +3–4 h |
| P2 (nếu chạy hết) | 39.4 | 9.9 | 22 / 25 | song song |
| Cổng | 38.0 | 9.5 | 21 / 24 | +2–2.5 h |
| **Tổng zero-shot** | **≈ 139** | **≈ 35** | **≈ 76 / 87** | **≈ 7.5–8.5 h** |
| P3 (một hướng thắng + C3 + GRPO thử) | 50–130 | 12–33 | 27–82 | 12–24 h |

---

## Phần 6 — Evaluation & Decision Gates

### 6.1 Cách đánh giá

- Cùng tập (public gốc 1,000 câu; held-out `pub_ho400c`, `tr_ho400c`), cùng chấm (`score.py`, `Process/tools/score_sets.py`), cùng code
  (commit đóng băng cho cả đợt, ghi hash), cùng bộ seed {0, 1}.
- Điểm theo nhóm: `pub_ind` (614 câu, DN có ngành) / `pub_noind`; loại đáp án (số, danh sách, văn bản, không trả lời được); câu thuật ngữ mới.
- Theo pha: số hàm W1 qua kiểm, độ chính xác link (so với bộ khớp P), tỉ lệ câu vào phán xử, `judge_pick` / `vote`.
- Hiệu năng: wall từng pha, tok/s, util GPU, VRAM đỉnh; ước tính thời gian trên 2×RTX 5090 (tỉ lệ theo tok/s; phải ≤ 6 h cho private test).
- Lỗi: nạp mô hình, thực thi tool, định dạng, timeout, mẫu bị cắt ở trần token, đầu ra lặp.
- **Ghép cặp:** so từng câu với ZS (cùng câu); báo số câu tốt hơn / kém hơn, z ghép cặp (như các vòng trước: `SE = sqrt(b + c) / n`).

### 6.2 Nhiễu và ngưỡng

[Đo] nhiễu một lần chạy: public ± 0.4 điểm, held-out gộp ± 0.3. Ngưỡng:

| Mức | Điều kiện |
|---|---|
| **Có triển vọng** (sau seed 0) | Δ_ZS ≥ +1.0 điểm public; **hoặc** Δ_ZS ≥ +0.5 và held-out tốt hơn ≥ +1.0; **hoặc** \|Δ_ZS\| ≤ 0.5 và nhanh hơn ≥ 1.3 lần |
| **Xác nhận** | tb 2 seed Δ_ZS ≥ +1.0, z ghép cặp ≥ 2, held-out không giảm > 0.3, chế độ H không giảm so với ZS chế độ H |
| **Dừng sớm** | Δ_ZS ≤ −2.0 ở seed 0 và held-out không tốt hơn; lỗi nạp / định dạng > 2 % sau 1 lần sửa; tốc độ dự kiến > 2.5 h wall cho public |
| **Loại** | không đạt "xác nhận" sau 2 seed; vượt 15B kể cả adapter; container ước tính > 6 h trên 2×5090 |

Biến thể **tham chiếu** (A0, A4, B2) không bao giờ được chọn; chỉ dùng để đo trần: nếu A0 ≤ ZS + 1.0 → hạ ưu tiên hướng A; nếu B2 ≤
phán xử 9B gốc + 0.5 → verifier mô hình riêng không đáng đầu tư.

### 6.3 Chọn hướng để fine-tune (P3)

Xếp hạng các biến thể "xác nhận" theo: (1) Δ_ZS tb 2 seed; (2) held-out; (3) tốc độ; (4) độ dư ngân sách tham số cho LoRA; (5) chi phí
fine-tune ước tính. Chọn **tối đa 2 hướng** (một backbone + một cơ chế bổ trợ, ví dụ A3 + B1 hoặc A3 + E). Cơ sở kỳ vọng fine-tune vượt
0.952: neo EXP-003 − ZS = phần lợi ích của 3 LoRA trên 9B; nếu backbone mới có Δ_ZS > 0 và phần lợi ích LoRA tương tự thì dự kiến vượt
neo — phải đo bằng FT-101 / 102, không giả định.

---

## Phần 7 — Risk Management

| Rủi ro | Dấu hiệu | Phòng / xử lý |
|---|---|---|
| OOM | vLLM lỗi khi cấp KV; A3 FP8 ≈ 15 GB trên 22.5 GB | `--gpu-util 0.92`, giảm `max_num_seqs` 128 → 64, `max_len` 8192 giữ nguyên; vẫn OOM → TP2 × DP2 |
| Hàng đợi phiên | QUEUED > 8 phút (L4) | xoá, đẩy lại (tối đa 5 lần), chuyển acc |
| Hết quota | `quota.py` < est_h × 2.5 | `launch_many.py` từ chối; cắt P2; chờ 21/11 |
| Tải / nạp mô hình lỗi | sha256 sai, thiếu shard, output kernel > 20 GB | manifest sha256; chia shard ≤ 19 GB; nạp từ ổ cục bộ (copy khỏi `/kaggle/input` trước khi nạp) |
| Multi-GPU không tương thích | TP2 lỗi với kiến trúc hybrid DeltaNet / MoE | thử nhỏ (`--subset 20`) trước; dự phòng FP8 DP2 với max_len nhỏ hơn |
| Kiến trúc không khớp | vLLM 0.32 thiếu lớp `qwen3_5_moe`, template / `<think>` khác | kiểm `config.json → architectures` với registry vLLM trong smoke; lớp tương thích trong `agent.py`, không đổi prompt |
| Timeout runtime | gần 12 h / phiên | runner hạn chót 12 h − 40 phút; `--budget`; các thí nghiệm ở đây ≤ 2.5 h |
| Rò rỉ dữ liệu | dùng public / private để hiệu chỉnh / huấn luyện | hiệu chỉnh tỉa, probe, TTA chỉ dùng DN train; đáp án public chỉ để chấm; không xem từng câu held-out |
| Vi phạm 15B | đếm tham số sai (vision encoder, embedding không chia sẻ, adapter) | `tools/param_count.py` trên checkpoint thật; trần 14.8B kể cả adapter; ghi số vào `result.json` |
| Chọn nhầm do nhiễu | Δ nhỏ hơn nhiễu | 2 seed + ghép cặp + held-out trước khi kết luận |
| Kết luận sai về kiến trúc mới | C2 thấp vì chưa huấn luyện | C2 chỉ là thăm dò; kết luận hướng C chỉ sau FT-103 |
| Lệch với chính sách nộp chế độ H | thắng ở `flow_p5_final` (còn `industry.py`) nhưng thua ở chế độ H | EXP-097 bắt buộc trước khi chọn |
| Lộ token | token trong log / lỗi | không in biến môi trường; lọc `KAGGLE_API_TOKEN` khỏi log; file token 600 ngoài repo |

---

## Phần 8 — Final Recommendation

1. **Làm trước (P0, ngay khi có token):** EXP-000 → EXP-001 / 002 (ZS baseline 2 seed) + EXP-003 (neo) trên 2 acc; tải mô hình song song
   trên phiên CPU.
2. **Song song ngay sau (P1):** hướng A (A0 tham chiếu, A1 REAP-64, ACAL → A2 / A3), hướng B (B1 4B + B2 tham chiếu trên cùng log, B3 offline),
   hướng D1 (phân rã). Đây là các biến thể có checkpoint / đường mã gần sẵn và giả thuyết rõ nhất.
3. **Đáng thử song song nếu quota cho phép (P2):** E1 / E2 (probe), C1 / C2 (kiểm cài đặt + thăm dò độ sâu), G1 (TTA in-context, sau code),
   H0 (phân tích CPU), I1 (Gemma, có điều kiện), A4 (chỉ khi A0 cho trần cao).
4. **Hoãn đến sau cổng (P3):** FT-101 / 102 (LoRA phục hồi + W cho backbone thắng), FT-103 (block expansion có huấn luyện), FT-104 / 105
   (GRPO), FT-106 (adapter mới), FT-107 (cần user quyết), FT-108 (IR / DSL).
5. **Không làm:** Recursive Transformer (chi phí phục hồi), lượng tử hoá để lách 15B (không hợp lệ), tỉa dense 27B.
6. **Phân bổ GPU:** 11 acc L4, 22 phiên, đỉnh 10 phiên; DP4 cho mô hình vừa 1 GPU, TP2 × DP2 cho tham chiếu lớn, 4 worker độc lập cho
   thí nghiệm trên log cố định; acc quota thấp nhận việc ngắn (EXP-020, EXP-051, held-out).
7. **Lộ trình:** T0 → T0 + 8.5 h: toàn bộ zero-shot + cổng + báo cáo quyết định → (user duyệt thời gian sau 16/11) P3 cho tối đa 2 hướng
   thắng, mỗi hướng ≈ 12–24 h wall, mục tiêu vượt neo EXP-003 / 0.952 với 2 seed và held-out không giảm.

### Việc code cần làm trước khi chạy (cho agent triển khai)

| File | Việc | Test |
|---|---|---|
| `configs/flow_p5_zs.json` | bản sao `flow_p5_final` với `links_lora` = `judge_lora` = "" | preflight với `--fake-responder` |
| `src/predict_flow.py` | đọc `seed_base` từ config (nếu chưa); hỗ trợ `--dp 2` × `--tp 2` | `tests/test_pipeline.py` mở rộng |
| `tools/param_count.py` | đếm tham số từ header safetensors, tách vision / embedding | so với 9B (≈ 9B) và REAP-64 (10.93B) |
| `automation/stage_model.py` | kernel CPU tải shard HF + manifest sha256 | chạy thử 1 shard nhỏ |
| `training/moe_saliency.py` | độ quan trọng expert kiểu REAP (dự phòng tần suất × cổng) | mô hình MoE tí hon CPU |
| `training/moe_prune.py` | cắt expert theo keep-list, sửa config, ghi checkpoint cục bộ | mô hình MoE tí hon: đầu ra bằng mô hình gốc khi giữ đủ expert |
| `training/expand_depth.py` | chèn nhóm lớp (khởi tạo 0 / nhân đôi) giữ mẫu 3:1 | logits khớp 9B khi khởi tạo 0 (tí hon CPU + 1 GPU) |
| `src/probe.py` | trích hidden state + huấn luyện / áp probe | dữ liệu giả |
| `automation/plans_ep0/*.json` | kế hoạch `launch_many.py` theo bảng §5.3 với `est_h` | `launch_many.py --dry-run` |

Mọi lần đẩy kernel phải qua cổng kiểm tra (`automation/preflight.py` in `PREFLIGHT OK`; tự review diff; thay đổi chỉ chạy trên Kaggle →
chạy `--subset` nhỏ trước).

## Tài liệu tham khảo

- `docs/proposed_method_1_b.md` (các hướng đề xuất); `configs/flow_p5_final.json`; `docs/18_PLAN5_TWO_LORA.md`; `handoff/01_SCIENTIFIC_FINDINGS.md`;
  `handoff/ENVIRONMENT_SUMMARY.md`; `Process/runs/v2/d1/` (số đo D1); `Process/kaggle/quota_2026-11-15.txt`.
- Song & Zheng, Rethinking LLM Pruning via Test-Time Scaling, arXiv 2611.01133 (2026).
- IAmTheRealGumbo, Qwen3.6-35B-A3B-REAP-64-experts (model card, Hugging Face).
- Bae et al., Relaxed Recursive Transformers, ICLR 2025 (arXiv 2410.20672).
- Wu et al., LLaMA Pro: Progressive LLaMA with Block Expansion, ACL 2024; Kim et al., SOLAR 10.7B (Depth Up-Scaling), NAACL 2024.
- Orgad et al., LLMs Know More Than They Show, ICLR 2025 (arXiv 2410.02707).
- Akyürek et al., The Surprising Effectiveness of Test-Time Training for Few-Shot Learning, ICML 2025 (arXiv 2411.07279).
- Arctic-Text2SQL-R1 (arXiv 2505.20315); Reasoning-SQL (arXiv 2503.23157); SQL-R1 (arXiv 2504.08600).
