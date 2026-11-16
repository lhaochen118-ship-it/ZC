# Thử nghiệm kết hợp model / component trên framework 0.952 (user 16/11 07:20)

Baseline cố định **F** = `configs/flow_p5_final.json` (Qwen3.5-9B + V3A × 0.70 / WL1 / WJR1, có `industry.py`): D1 **0.951** (lần gốc,
`Process/runs/v2/d1`), C0L **0.942** (chạy lại cùng mã hiện tại, `think_kw` legacy). Nhiễu giữa 2 lần ≈ 0.01 → mọi kết luận bằng **ghép
cặp từng câu** (z) với cả hai. Mọi cấu hình thử nghiệm được sinh bằng code từ F, in danh sách khoá khác F; điều chỉnh tương thích ghi `_doc`.
Chế độ H (bản nộp, CLAUDE.md) đo song song trên `flow_p5_final_h` (0.863) khi có.

## 1. Component đã đo (bước 1)

| Component | Chức năng | Zero-shot (vs B00-S 0.754) | Trên F / F_H (tinh chỉnh) | Chi phí |
|---|---|---|---|---|
| Suy nghĩ mọi mẫu | S | +0.051 (z 4.7) | đã có trong F (template) | — |
| MoE E88 | S / L / J | +0.025 (z 2.3, 2 seed) | e7rw7 (S+L+J = E88 + LoRA): **0.927** (vs D1 z −2.1, vs C0L z −1.4) | ≈ 9B |
| MoE E92 | S / L / J | +0.019 (z 1.6, 1 seed) | **chưa** (cần 3 LoRA mới) | ≈ 9B |
| Phán xử không trần | J | +0.023 (z 2.8) | không áp dụng: J của F xong trong trần | — |
| Q01 BF16 | S | +0.010 (z 1.3) | hq01 / hq01s1 (F_H) đang chạy | S ×2 thời gian |
| T01 | S (ví dụ) | +0.010 (z 1.0) | C1L 0.943 (+0.001 vs C0L); F_H 0.857 / 0.860 | ≈ 0 |
| k_max 24 | S | — | C2L (+T01) 0.935 | S ×1.5 |
| 4B làm J | J | âm | C5 0.930 | — |
| Gộp ứng viên nhiều lượt | J | — | F_H: 0.854–0.863 | S ×2–3 |
| T02, N4-A | S | ≤ 0 | F_H: 0.777, 0.848 | — |

## 2. Không gian tổ hợp và thứ tự ưu tiên (bước 2)

Ba vai trò dùng model: bộ giải **S**, bộ link **L**, bộ phán xử **J**; mỗi vai trò ∈ {9B (+ LoRA của F), E88 (+ LoRA tương ứng), E92 (+ LoRA)}.
F = (9B, 9B, 9B); e7rw7 = (E88, E88, E88). Thiết kế thừa số cho biết vai trò nào MoE có ích, rồi mới ghép component (T01, Q01).

| ID | S | L | J | Component | Cách đo | Acc | Trạng thái |
|---|---|---|---|---|---|---|---|
| F | 9B | 9B | 9B | — | D1 0.951, C0L 0.942 | — | xong |
| e7rw7 | E88+MS2×0.7 | E88+ML1 | E88+MJ2 | — | đầy đủ | bichthuy93 | **0.927** |
| jsj | 9B | 9B | **E88+MJ2** | — | chỉ phán xử, trên ứng viên + link + hàm của D1 | duckzuykobip | đang chạy |
| jsq | E88 | E88 | **9B+WJR1** | — | chỉ phán xử, trên ứng viên của e7rw7 | phuongnehe | đang chạy |
| rss | **E88+MS2×0.7** | 9B | 9B | — | đầy đủ (W LoRA gộp vào 9B+V3A×0.7: `w_base`) | thnliu | chờ gộp MS2×0.7 |
| rsl | 9B | **E88+ML1** | 9B | — | đầy đủ | huyenco | chờ gộp ML1 |
| e8rw7 / e8r | E88 (×0.7 / ×1.0) | E88 | E88 | — | đầy đủ, chế độ H (so 0.863) | bichthuy93 / caohuyen74 | đang chạy |
| e8ms3 | E88+**MS3** (LoRA cả 88 expert) | E88 | E88 | — | chế độ H, 2 nửa cửa hàng | loinguyentran / namhunhh | chờ huấn luyện fts3b |
| hq01 ×2 | 9B **BF16** | 9B | 9B | Q01 | chế độ H | hai1108 / nhatlong1103 | đang chạy |
| (sau) | vai trò thắng | | | + T01 / Q01 | ghép khi một vai trò E88 thắng F | — | có điều kiện |

**Không chạy (lý do):** E92 + 3 LoRA — cần huấn luyện 3 LoRA mới (MS ~16.7M token ≈ 2.7 h tường ≈ 6 h quota; không acc nào còn ≥ 6 h,
nối checkpoint qua acc + gộp + đánh giá vượt hạn nộp); E92 zero-shot (0.773) < E88 (0.781) và E88 + LoRA đã thua F → xác suất vượt
0.952 thấp. Tổ hợp E88 + T01 / k24: T01 / k24 không cải thiện trên F nên không ghép. Phán xử không trần, suy nghĩ: không đổi F.

## 3. Kết quả (cập nhật khi có)

Xem `docs/19_PLAN_P5_ZS.md` (nhật ký) và bảng ở đây khi xong.
