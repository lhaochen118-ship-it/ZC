# New7 (7 phương pháp / 14 biến thể) — kế hoạch cập nhật theo thời gian ĐO được (15/11 16:00 UTC)

Nguồn: `New7_P5_Package.zip` (user 15/11). Cập nhật: (1) thời gian lấy từ các lần chạy thật hôm nay, không còn ước lượng; (2) đổi cách
đánh giá để tối thiểu hoá thời gian — mọi biến thể **ở giai đoạn chọn / kiểm** được đo bằng **phát lại trên log ứng viên đã có** thay vì
sinh lại toàn bộ trên GPU; (3) chỉ đo trên **public D1** (user 15/11 13:20: không chia tập khác); (4) mốc ghép cặp: B00-S s0 / s1
(0.753 / 0.755) và backbone tốt nhất P02 (0.781; + phán xử không trần 0.788).

## 1. Thời gian đo được (4×L4, Kaggle)

| Việc | Đo | Nguồn |
|---|---|---|
| Lượt public đầy đủ, 9B zero-shot (W1 + units + link + S + phán xử) | **110–113 phút** tường; quota ≈ 2.2 × tường | b00s0 / b00s1 |
| Pha phán xử trên log có sẵn (chỉ câu bất đồng, 261 câu) | 9B: ~45 phút; MoE E88: 77 phút; 4B: 25–30 phút | s01, p02j, v02 |
| Phát lại CPU một chương trình ứng viên (sandbox + toolkit + units) | **0.17 s**; độ trung thực 100 % (bảng chuẩn dựng lại offline khớp lúc chạy) | n1a_replay |
| Cài vLLM + nạp mô hình mỗi kernel | 7–10 phút | mọi kernel |
| Fine-tune LoRA MoE E88 (4M token, FSDP 4 GPU, grouped_mm, mb 4) | 40–55 phút | fts / ftl / ftj |

## 2. Thiết kế tối thiểu thời gian

| Nhóm | Biến thể | Cách đo | Tài nguyên | Tường dự kiến |
|---|---|---|---|---|
| Sinh lại (prompt đổi) | **N4-A** | 2 seed song song, 2 phiên 4×L4 cùng acc | 8 L4 | **~1.9 h** (đang chạy từ 15:55) |
| Kiểm trên chương trình đã có, không gọi LLM | **N1-A**, N1-B, N7-B (sau N7-A) | phát lại CPU trên log 3 lượt (B00-S s0, s1, P02) | 4 lõi CPU cục bộ / kernel CPU | **5–10 phút / log** |
| Kiểm cần Qwen viết thêm (chỉ câu cần) | **N7-A** (chứng chỉ đơn vị), **N3-A** (phản ví dụ + Qwen chọn), **N6-A** (Qwen viết DuckDB SQL, đối chiếu) | một lượt vLLM trên log (`--runner script`, 4 shard = 4 GPU), rồi chấm CPU | 1 phiên 4×L4 / biến thể / log | **20–40 phút** + cài 10 phút |
| Cấu trúc sâu (có cổng) | N1-B, N2-A, N4-B, N7-B | chỉ khi biến thể mẹ có Δ > 0 | CPU / 1 phiên | 10–60 phút |
| P3 / chặn | N2-B, N3-B, N6-B, **N5-A / N5-B** | N5: vLLM không cho can thiệp activation; chạy toàn luồng bằng HF ≈ 10× chậm → vượt 6 h của container ⇒ **không khả thi trong hạn**; N2-B / N3-B / N6-B chỉ sau khi mẹ qua cổng | — | — |

Đường găng: N4-A (1.9 h) ‖ code N7-A / N3-A / N6-A (~2 h) → lượt GPU (~0.7 h) → chấm + ghép vào hệ tốt nhất (~0.5 h) ⇒ **~3.5 h** cho
toàn bộ G1 + G2 thay vì 9–16 GPU-h của kế hoạch gốc cho G1 (mỗi biến thể chọn / kiểm không cần lượt đầy đủ 2 h × 2 seed).

## 3. Cổng và ghép

- Δ ghép cặp so với cùng log (cùng ứng viên) — phát lại loại bỏ nhiễu sinh mẫu, nên z đáng tin hơn so sánh 2 lượt khác nhau.
- Biến thể qua cổng (Δ > 0 trên cả 2 seed B00 và trên P02) được ghép vào hệ tốt nhất (P02 + phán xử; framework 0.952 / E3 khi có).
- Không luật nghiệp vụ viết tay: mọi kiểm tra là toán / dữ liệu trung lập; các quyết định ngữ nghĩa do Qwen.
