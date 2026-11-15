# STATUS — trạng thái sống (đọc đầu tiên khi tiếp tục)

*Cập nhật 2026-11-15 07:00 UTC. User 06:00: báo cáo score, **ngắt các hướng không tiềm năng**, user sẽ đổi hướng nghiên cứu (chờ chỉ thị;
không đẩy kernel mới). Đóng băng 16/11 04:00 UTC. Findings: PDF 1 §9. Lỗi + cách tránh: PDF 2 §10. Nhật ký kết quả: `handoff/06` §7.*

**Tốt nhất chế độ H:** phux2n (I3, 2 ví dụ tự sinh / cửa hàng) **0.869** nhưng ghép cặp +0.006 (z +0.7) = nhiễu (chỉ 14 ví dụ cho 3 cửa hàng sống
qua kiểm thực thi); mốc `configs/flow_v2_final.json` 0.859–0.863 (3 lần: H0a, H0a2, pfix; pfix có sửa gộp → container = ensemble). Hàm cửa hàng do
S viết đóng góp +0.03 (không hàm: 0.834).

**Đã đóng:** WT (mọi biến thể), thêm mẫu (k24, u16), S′, F, phán xử / ngưỡng, Qwen gốc làm kênh 2 (06:05); bộ viết hàm có suy nghĩ (phut 0.839,
z −2.3; ngắt phmq2t 06:53).

**Kết quả giai đoạn C:** pfix 0.859, phux 0.857 (16 ví dụ / 2 cửa hàng), phux2n 0.869, phut 0.839, phmq 0.859 (28 hàm), phmqx 0.840, phmqc 0.862, phmquv 0.840 (3 lần này: shard 2 viết hàm hỏng → 15–17 hàm, không kết luận được; đã sửa log để thấy traceback).

**Còn chạy (watcher `watch.out` + Monitor):** phcodeb, phux2b, phqcb, phuvb, phquvx, phall, phmq2, phmq2uv, phmq2x;
S″: es3ar (S″-A), es3fr (S″-A′, đẩy 06:32). Kết quả tới ~08:45; chấm `python3 Process/tools/v2/grid_report.py <runs dir>`.

## Tiến độ v2 (lịch sử, 14/11 17:05 → 15/11 03:40)
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
| Đo WT đầu-cuối (614 câu H: ewtb, ewts_wt) | nhatlong1103 ×2 | huỷ (user 15/11: dừng held-out) | |
| Lưới public (15/11 01:30 →) | 16 acc | xem `handoff/06_PUBLIC_GRID.md` §5 | mọi biến thể ≤ mốc 0.863 |
| UQ sinh câu dùng hàm (S″) | lelynhan/zc-uq-1 | xong | 1,336 cặp chạy được; ~15 % sai kỳ (kiểm 30) → kiểm ngược uqchk |


*(Trạng thái kế hoạch 4–5 trước 14/11 17:00: PDF 1 §5, §8; lịch sử git.)*
