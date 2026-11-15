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
