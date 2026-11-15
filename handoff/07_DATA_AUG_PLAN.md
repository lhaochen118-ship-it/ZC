# Data augmentation: mô hình tự viết mẫu mới để huấn luyện lại Qwen (S″, WT v3)

*Lập 15/11/2026 02:50 UTC theo yêu cầu user. Đóng băng container 16/11 04:00 UTC (còn ~25 giờ). Chế độ H, không luật viết tay, không
dùng câu public / private để huấn luyện (chỉ cửa hàng train).*

## 1. Vì sao, và nhắm vào đâu

| Bằng chứng (public gốc, chế độ H) | Hệ quả cho data augmentation |
|---|---|
| 76 / 93 lỗi nằm ở cửa hàng ngành (không còn `industry.py`) | dữ liệu mới phải dạy cách xử lý **định nghĩa mới, bảng mới** |
| Có hàm do WT viết trong prompt, nhưng **chỉ 10 % mẫu S gọi tới** (205 / 2,108 mẫu ở cửa hàng ngành, log EWT0) | S chưa từng được huấn luyện với khối `# SHOP FUNCTIONS` → **dạy S dùng hàm cửa hàng** |
| WT-b học từ dữ liệu do mô hình sinh, kiểm bằng thực thi: đúng 0.596 vs 0.517 (z 2.8) | chưng cất + lọc thực thi có tác dụng thật |
| S′ (chỉ dòng `# RULES` + cặp tương phản, không có hàm) không tăng public (0.861 / 0.856 vs 0.863) | dạy đọc định nghĩa văn xuôi chưa đủ; phải gắn với **lời gọi thực thi được** |

## 2. Vai trò của Claude và Qwen

- **Qwen (Qwen3.5-9B gốc, không suy nghĩ, JSON có ràng buộc)** sinh dữ liệu ở quy mô lớn trên GPU: câu hỏi mới, chương trình, biến thể.
- **Claude** chỉ viết **6–8 ví dụ mẫu** (few-shot) cho prompt sinh, trên dữ liệu cửa hàng train; kiểm ngẫu nhiên 30 bản ghi mỗi nguồn trước khi huấn
  luyện; không viết nhãn hàng loạt và không xem câu public.
- **Nhãn luôn qua thực thi:** chương trình phải chạy được trên dữ liệu cửa hàng, ra giá trị không suy biến, và khớp với lời gọi hàm gốc hoặc
  với đa số mẫu độc lập.

## 3. Các nguồn dữ liệu mới

| Mã | Nguồn | Cách sinh | Nhãn / kiểm | Quy mô |
|---|---|---|---|---|
| **U1** câu dùng hàm cửa hàng | 2,235 hàm của thầy + từ chương trình đã kiểm (TG1b, TG1c, `wt_from_programs`), cửa hàng train trừ 6 DN held-out | Qwen viết 2 câu / hàm theo cách chủ cửa hàng nhắn (có kỳ, viết tắt, không dấu…) + chương trình gọi hàm trên kỳ của câu | chương trình chạy được, kết quả = gọi trực tiếp hàm với kỳ đó; kỳ trong chương trình phải là kỳ câu nói (kiểm ngược bằng S: ≥ 2 / 4 mẫu ra cùng kỳ) | ~3–4k |
| **U2** câu không liên quan khi có khối hàm | bản ghi v3a + khối `# SHOP FUNCTIONS` gồm hàm không liên quan | không sinh mới | chương trình gốc (đã kiểm theo đáp án train) | ~2.5k (50 % replay) |
| **U3** khối `# RULES` + hàm | câu U1 kèm dòng rulebook của thuật ngữ (+ 1 dòng nhiễu) | ghép | như U1 | ~1.5k |
| **P1** pandas khi không có hàm | dòng tổng hợp chưa có hàm đồng thuận | Qwen viết câu + chương trình pandas trên C / T, 6 mẫu | giữ khi ≥ 3 / 6 chương trình độc lập cùng kết quả | ~0.5–1k |
| **W3** dữ liệu WT v3 | lượt thầy thứ ba (seed 999, 30 dòng / DN) + chính WT-b v2 làm thầy (expert iteration) trên dòng thầy cũ chưa có hàm | như TG1 | đồng thuận ≥ 3 + kiểm 13 kỳ | +200–400 hàm |

## 4. Biến thể huấn luyện (song song, có lý do)

| Biến thể | Dữ liệu | Khởi điểm | Lý do | Thời gian |
|---|---|---|---|---|
| **S″-A** | U1 + U2 + U3 + 1,000 replay v3a gốc | LoRA chồng V3A × 0.70, r 32, lr 1e-4, 1 epoch | dạy riêng việc gọi hàm cửa hàng; ít token → nhanh, ít quên | ~2.5 h FSDP 4×L4 (~14M token) |
| **S″-B** | S″-A + P1 + cặp tương phản (1,005) | như trên | thêm khả năng tự viết pandas khi không có hàm | ~3.5 h (~20M token) |
| **WT v3** | dữ liệu WT v2 + W3 | Qwen gốc, như WT-b | thêm độ phủ hàm cho dòng ngành | ~1.6 h |

Không làm: huấn luyện trên câu public (cấm); nhãn "Unanswerable" khi xoá dòng (đã chứng minh sai nghĩa); bộ phán xử học tương phản (WJR2D
đã giảm điểm).

## 5. Đo và chọn

- Mỗi biến thể S″ chạy đường container đầy đủ ở chế độ H với **WT-b tốt nhất** (WT-b hoặc WT-b v2 tuỳ kết quả `pwtb` / `pwtbv2`).
- So với cùng cấu hình dùng S = V3A × 0.70 trên cùng public (so ghép cặp từng câu); 2 lần chạy cho biến thể tốt nhất.
- Chọn cấu hình nộp: `pub_full_H` cao nhất.

## 6. Lịch và acc (5 acc mới + acc còn quota)

| Bước | Giờ UTC | Việc | Acc / phiên | Giờ tường |
|---|---|---|---|---|
| 1 | 02:50–04:20 | code: chế độ sinh câu từ hàm (`lrl_gen --mode unitq`), dựng bản ghi S″ (`s3_build.py`), ví dụ mẫu; smoke CPU | CPU | 1.5 h |
| 2 | 04:20–05:30 | sinh U1 / P1 (Qwen gốc, DP 4) | acc mới 1 × 2 phiên (U1 ∥ P1) | 1.1 h |
| 2′ | 04:20–06:00 | W3: lượt thầy 3 + WT-b v2 làm thầy | acc mới 2 × 2 phiên | 1.5 h |
| 3 | 05:30–06:00 | dựng dữ liệu S″-A / S″-B, kiểm ngẫu nhiên | CPU | 0.5 h |
| 4 | 06:00–09:30 | huấn luyện S″-A ∥ S″-B (FSDP 4 GPU, mb 2–4 theo độ dài đo bằng tokenizer, expandable_segments) | acc mới 3 × 2 phiên | 2.5–3.5 h |
| 4′ | 06:15–08:00 | huấn luyện WT v3 | acc mới 2 (phiên 1) | 1.6 h |
| 5 | 09:30–11:45 | public H: S″-A + WT, S″-B + WT, (WT v3 nếu xong) | acc mới 4 × 2 phiên + acc mới 5 × 1 | 2.2 h |
| 6 | 11:45–14:00 | lần chạy thứ 2 cho cấu hình tốt nhất + đường container | acc mới 5 + acc cũ | 2.2 h |
| 7 | 14:00–20:00 | Docker (adapter, `flow_v2_final.json`), thử bố cục trên Kaggle, tài liệu | 1 acc | — |

Quota: mỗi phiên huấn luyện / đo tốn ~4–8 h quota; 5 acc mới (~150 h) đủ cho các bước 2–6. Acc cũ còn ~120 h dùng cho lưới public đang chạy và
dự phòng.

## 7. Phương án chạy đã tối ưu lại (03:30 UTC, sau khi user yêu cầu rà lại)

**Huấn luyện (S″-A, S″-B).**
- Một lần huấn luyện không thể trải qua nhiều phiên (không có NCCL giữa phiên Kaggle) → mỗi biến thể **một phiên, FSDP FULL_SHARD trên cả 4 L4**,
  hai biến thể **song song** trên 2 phiên của cùng một acc (duylnng).
- Vì sao FSDP chứ không DDP: 9B bf16 = 18 GB > phần trống của một L4 (22.5 GB) khi cộng kích hoạt → DDP chỉ chạy được với QLoRA 4-bit (chậm hơn
  ~1.3–1.5× mỗi token và lệch số học với mô hình fp8 lúc suy luận). FSDP chia tham số 4.5 GB / GPU.
- Độ dài đo bằng tokenizer Qwen (không ước theo ký tự): bản ghi S′ p50 3,453 / p99 4,414 / max 4,550 token; S″ thêm khối hàm (~+300) →
  `--max-len 5120`; mb 4 (S′ ở 4,608 đỉnh 16.5 GB / GPU) + `expandable_segments` (tránh phân mảnh như lần OOM của WT); batch hiệu dụng 64.
- **Giảm token** (đòn bẩy lớn nhất vì prompt dài, loss chỉ trên phần trả lời): S″-A ≈ 13M token (~2.1 h), S″-B = S″-A + 1,005 cặp tương phản /
  cách viết ≈ 16.5M (~2.6 h); bỏ nhóm "thuật ngữ mới trong # RULES" (S′ cho thấy không giúp).
- Checkpoint mỗi 20 bước + tại hạn chót; không cần nối acc vì < 3 h.

**Đo (public H).**
- Mỗi biến thể một phiên, DP 4 (mỗi GPU một vLLM fp8, `gpu_util` 0.92 ≈ 20 GB / GPU).
- **Dùng lại hàm cửa hàng và link** của lần chạy WT-b (`units_file`, `links_file`; chỉ cho thí nghiệm, bản nộp vẫn sinh lại): bớt ~15–20 phút
  mỗi lần, và so S″ với S **trên cùng một bộ hàm** (so ghép cặp sạch hơn).

**Chia acc.**

| Acc | Phiên 1 | Phiên 2 |
|---|---|---|
| lelynhan | sinh U1 (4 GPU = 4 shard) | — |
| duylnng | huấn luyện S″-A | huấn luyện S″-B |
| caotienminh1 | đo S″-A + WT-b | đo S″-B + WT-b |
| phatle1578 | đo lại cấu hình tốt nhất (seed 2) | đường container |
| mucnokieu | lượt thầy 3 → WT v3 (nếu WT v2 > v1) | dự phòng |

## 8. Điều chỉnh 03:30 UTC sau kết quả lưới public và kiểm dữ liệu UQ

- **WT-b bị loại** (hàm WT-b sai nghĩa trên rulebook public: mẫu gọi hàm đúng 0.149, hàm của S 0.514; `handoff/06` §5). Vì vậy S″ được đo với
  **hàm do S viết** (dùng lại `cards_unit` + `links` của H0a: `configs/flow_h_reuse.json`), so ghép cặp với H0a / H0a2 trên cùng bộ hàm.
- **UQ** (lelynhan, 4 shard, 19 phút): 1,336 cặp câu – chương trình chạy được (từ 907 hàm thầy). Claude kiểm ngẫu nhiên 30 cặp: ~15 % sai kỳ
  ("Chủ nhật tuần trước" → `last_week`, "1/7–10/7" ở tương lai, "cuối tuần này" → `last_week`). Nhãn nhiễu về kỳ có thể làm hỏng kỹ năng
  kỳ của S (câu không ngành đúng 0.96).
- **Lọc bằng mô hình (không luật viết tay):** S (V3A × 0.70, chế độ H) giải lại 1,336 câu với hàm thầy trong `# SHOP FUNCTIONS`
  (`configs/flow_uqchk.json`, tập `uqchk_a/b`, 2 phiên lelynhan); giữ cặp khi đáp án của S trùng đáp án chương trình sinh
  (`training/data/uq_filter.py`). Hai đường mô hình đồng ý = kiểm nhãn (kỳ, nghĩa của hàm).
- **Biến thể (thay S″-B):** S″-A = dữ liệu chưa lọc (đang huấn luyện, duylnng phiên 1, 3,436 bản ghi ≈ 13M token, mb 2 × 5120,
  `expandable_segments`); **S″-A′** = cùng công thức với U1 / U3 đã lọc (duylnng phiên 2 khi uqchk xong). So A vs A′ cho biết tác dụng của
  việc lọc nhãn. S″-B (thêm bản ghi S′) bỏ vì S′ đã không tăng điểm.
- Đo: mỗi biến thể 1 lần `flow_h_reuse` (cùng acc với output huấn luyện, vì output kernel chỉ gắn được trong cùng acc) + 1 lần đường đầy đủ
  (hàm do chính S″ viết) cho biến thể tốt hơn.
