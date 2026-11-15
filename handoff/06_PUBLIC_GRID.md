# Public grid — thử có hệ thống các giải pháp trên public (chế độ H)

*Lập 15/11/2026 01:30 UTC theo yêu cầu user 15/11 01:15: dừng held-out, thử nhiều giải pháp trên public, chọn điểm cao nhất; bản nộp vẫn
chế độ H (không hàm ngành viết tay). Đóng băng container 16/11 04:00 UTC.*

## 1. Căn cứ: lỗi của mốc chế độ H

Mốc: `flow_p5_final_h`, `pub_full_H` **0.863** (H0a, H0a2). 93 câu chưa đúng ở H0a:

| Nhóm | Câu | Giải pháp nhắm vào |
|---|---|---|
| Cửa hàng ngành, không mẫu nào đúng | 35 | **G1** hàm do mô hình viết (WT) |
| Cửa hàng ngành, có mẫu đúng nhưng bỏ phiếu trắng / chọn sai | 23 + 6 | **G3** chọn (số mẫu, phán xử) |
| Cửa hàng ngành, đồng thuận sai | 12 | **G1**, **G2** S đọc luật |
| Không ngành, có mẫu đúng nhưng mất | 10 | **G3** |
| Không ngành, khác | 7 | **G2** |

## 2. Yếu tố và biến thể

Mỗi biến thể là một cấu hình đường container đầy đủ (`soup_job --runner predict`). Chính sách phán xử chọn offline trên log của
chính lần chạy, không tốn GPU.

| Yếu tố | Biến thể | Giả thuyết | Căn cứ |
|---|---|---|---|
| **G1** bộ viết hàm (pha units) | W = S (mốc) / **WT-b** / WT-s / WT-b v2 (+204 hàm tổng hợp) / WT-b 8 mẫu + 2 vòng sửa | hàm đúng cho dòng rulebook ngành giảm nhóm "không mẫu nào đúng" | WT-b đúng 0.596 vs 0.517 trên dòng held-out (z 2.8) |
| **G2** bộ giải S | V3A × 0.70 (mốc) / **S′ seed 1 / 2** | S′ học đọc `# RULES` và cặp tương phản | S′ huấn luyện xong, chưa đo |
| **G3a** số mẫu S | k_max 16 (mốc) / 24 | thêm ứng viên đúng cho câu khó | 39 / 93 lỗi có mẫu đúng |
| **G3b** phán xử | tổ hợp list + point + point_think ngưỡng 0.5 (mốc) / ngưỡng 0.6 / chỉ point_think / thêm list+c, point+c (đọc 100 %) | chọn đúng hơn ở câu nhiều ứng viên | offline trên log H: 0.855–0.869 |
| **G4** thẻ W1 | thẻ public đã có (bị lọc gợi ý ngành) / **W1 sinh lại ở chế độ H** | giống điều kiện private (rulebook mới → W1 viết lại) | 14 / 125 thẻ mất gợi ý |

## 3. Giai đoạn

| Giai đoạn | Nội dung | Số lần chạy | Quy tắc chọn |
|---|---|---|---|
| **A** (đang chạy) | đơn yếu tố so mốc: WT-b, WT-s, WT-b8, k24, S′1, S′2; phán xử lại log H (jh) | 6 + 1 | so ghép cặp từng câu với H0a / H0a2 |
| **B** | ghép yếu tố thắng: S tốt nhất × W tốt nhất × k ∈ {16, 24}; + W1 sinh lại; WT-b v2 | 4–6 | điểm cao nhất, xét cả ghép cặp |
| **C** | xác nhận: 2 cấu hình đầu × 2 lần chạy (đường container) | 4 | TB 2 lần; chọn cấu hình nộp |
| **D** | Docker: thử bố cục image trên Kaggle với cấu hình nộp | 1 | khớp ± 0.01 |

## 4. Chia việc (16 acc; mỗi lần chạy ~2.0–2.4 h tường ≈ 4.4–5.3 h quota)

- Mỗi biến thể chạy **một phiên riêng, DP 4**: mỗi GPU một bản vLLM fp8, `gpu_util` 0.92 (~20 GB / GPU, ~80 / 96 GB), hàng đợi động
  theo cửa hàng. Đây là cách nhanh nhất về giờ tường khi đủ phiên.
- Phán xử lại log: **mỗi GPU một biến thể** (4 biến thể / phiên).
- Huấn luyện (WT v2): FSDP 4 GPU, mb 2 × tích luỹ 8, `expandable_segments`.
- Acc đủ 2 phiên trước khi sang acc khác. Kiểm quota trước mỗi lần đẩy (`launch_many.py`, hệ số 2.5): acc dưới 6 h chỉ nhận job vừa đủ.
- Adapter dùng chung: dataset `<acc>/zc-ad3` (V3A, WL1, WJR1, WTB, WTS, S2S1, S2S2) trên 9 acc, để acc nào cũng chạy được mọi tổ hợp.

| Acc | Quota (01:30) | Giai đoạn A (đang chạy) | Giai đoạn B / C |
|---|---|---|---|
| synchan5704 | 9.8 h | pWTB, pWTS | — (hết) |
| nhatlong1103 | 19.4 h | pWTB8 | B ×1, C ×2 |
| loinguyentran | 27.1 h | pHK24 | B ×2, C ×2 |
| nguynduylm | 9.5 h | S′1, S′2 | — |
| anderson1202 | 11.2 h | jh (phán xử) | B ×1 |
| phuongnehe | 20.5 h | huấn luyện WT-b v2 | B ×1 (WT-b v2), C ×1 |
| thnliu | 25.9 h | — | B ×2, C ×1 |
| linhmai24, lelynhan | (mới) | — | B ×2 mỗi acc, D |
| huyenco | 18.4 h | — | B ×2 |
| phatle1578 | 12.1 h | — | B ×1 |
| mucnokieu | 10.9 h | — | B ×1 |
| yuan1103, nguyenhoa99, quynhtran35 | 6.6–7.9 h | — | dự phòng (1 lần chạy mỗi acc) |
| quynhle07 | 0 h | — | hết quota |

## 5. Nhật ký

*(ghi kết quả từng lần chạy: điểm `pub_full_H`, ghép cặp với mốc, chính sách phán xử tốt nhất)*
| Lần chạy | Cấu hình | pub_full_H (ensemble) | point_think riêng | Ghi chú |
|---|---|---|---|---|
| H0a / H0a2 | mốc `flow_p5_final_h` | **0.863 / 0.863** | 0.868 / 0.858 | |
| F1 | + API extra | 0.847 | — | loại |
| es2s2p | S′ seed 2 (W = S′) | 0.856 | 0.848 | −0.007 so mốc |
| (offline) | ngưỡng bỏ phiếu θ 0.3 / 0.4 / 0.5 × ngưỡng phán xử 0.5 / 0.6 trên log H0a, H0a2, es2s2p | θ 0.5 tốt nhất ở cả 3 (0.863 / 0.863 / 0.856); m 0.6: +0.001 / +0.006 / −0.006 | — | chọn đáp án không phải đòn bẩy |
| jh (phán xử lại log H0a / H0a2) | list+c, point+c, point+think, list+think (WJR1) | các tổ hợp 0.849–0.866 (tốt nhất list_c+point_think m0.6: 0.863 / 0.866) | point+c 0.845 / 0.854 | phán xử không phải đòn bẩy ở chế độ H |
| es2s1p | S′ seed 1 | 0.861 | 0.860 | hai seed S′ đều < mốc (0.856 / 0.861) → S′ riêng loại; chờ S′ + WT-b |
| pwtb | WT-b viết hàm (20 hàm, W = S: 14) | 0.855 | 0.850 | pub_ind 0.561 (mốc 0.606): hàm đúng hơn trên train held-out chưa thành điểm public — S gọi hàm ở 10 % mẫu |
| pwtb8 | WT-b, 8 mẫu + 2 vòng sửa (21 hàm) | 0.846 | 0.837 | kém hơn |
| ps2b | S′ seed 2 + WT-b | 0.830 | — | 69 sai: S′ và WT-b cùng kéo xuống |
| (offline, log H0a / pwtb) | mẫu gọi hàm cửa hàng: hàm do S viết đúng 0.514 khi được gọi; hàm WT-b đúng **0.149** | — | — | WT-b viết hàm "lười" gọi hàm toolkit không liên quan (vd. `so_sua` = `top_customers`) vẫn qua kiểm thực thi 13 kỳ; câu mà đa số mẫu gọi hàm: S 13 đúng / 4 sai, WT-b 7 đúng / 10 sai → **loại G1 WT-b / WT-s**; hướng mới B′: bộ viết hàm = S (16 mẫu, có suy nghĩ + budget forcing) |
| (offline) | gộp mẫu 2 lần chạy mốc (≈ k 32), bỏ phiếu thuần θ 0.5 | 0.857 (từng lần 0.851 / 0.855) | — | thêm mẫu gần như không tăng: đòn bẩy là chất lượng mẫu ở câu ngành (mẫu đúng ~0.2 ở câu khó) |

## 6. Giai đoạn B′ (03:10 UTC): bộ viết hàm = S

Căn cứ: hàm do S viết có ích (câu đa số mẫu gọi hàm: 13 đúng / 4 sai), hàm WT-b có hại; 31 dòng rulebook ngành trên public nhưng H0a chỉ có 14
hàm (dev_011 không hàm nào, dev_014 chỉ 2). Đòn bẩy: thêm hàm đúng do S viết.

| Lần chạy | Cấu hình | Acc | Giả thuyết |
|---|---|---|---|
| phu16 | S viết hàm, 16 mẫu, giữ khi ≥ 3 mẫu cùng kết quả | caotienminh1 | thêm mẫu → nhiều dòng đạt đồng thuận hơn, hàm chọn theo đa số đáng tin hơn |
| phut | S viết hàm có suy nghĩ (7,168 token) + **budget forcing** (đóng `</think>` ở trần rồi giải mã JSON có ràng buộc) | caotienminh1 | đọc kỹ định nghĩa nhiều điều kiện (vd. tỉ lệ bảo hành = lượt / số máy bán, bỏ đơn huỷ / trả) |
| phnou | không hàm (đối chứng) | mucnokieu | đo phần đóng góp của hàm do S viết |
| phk24 | k_max 24 (W = S, 12 hàm) | 0.852 | — | 911 đúng / 59 sai / 30 bỏ: thêm mẫu → thêm trả lời sai; pha giải +35 phút (4,927 s vs 2,825 s) → **k24 loại** |
| pwts | WT-s viết hàm (20 hàm) | 0.846 | — | 62 sai; cùng hướng WT-b → loại |
| pwtbk24 | WT-b + k24 | 0.849 | — | loại (WT-b, k24 đều không giúp) |
| pwtbma1 | WT-b, giữ hàm không cần đồng thuận (29 hàm) | 0.848 | — | thêm hàm → thêm sai (64) → loại |
| ps1b | S′ seed 1 + WT-b | 0.842 | — | loại |
| pwtbu24 | WT-b, 24 mẫu viết hàm | 0.846 | — | loại |
| pbase | Qwen gốc làm bộ giải (kênh thứ 2 để gộp) | — | 0.753 (container) | quá yếu → loại |
