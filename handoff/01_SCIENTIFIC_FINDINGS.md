# PDF 1 — Scientific Findings (DataMate, đến 13/11/2026)

*Phạm vi: đóng góp, ý nghĩa khoa học, kết quả tích cực và tiêu cực, bằng chứng, điều kiện kiểm chứng. Kinh nghiệm kỹ thuật /
vận hành ở PDF 2; tiến độ, quyết định, vị trí file và việc tiếp theo ở PDF 3. Số liệu: `Process/scores.csv` (mọi lần chạy),
`docs/12_PLAN2.md` (kế hoạch 2), `docs/14_PURE_MODEL_PLAN.md` (mô hình thuần), `docs/15_AUDIT.md` (kiểm tra 13/11),
`docs/16_PLAN3_LEARNED_RULES.md` §9 (kế hoạch 3), `EXPERIMENT_LOG.md` (kế hoạch 1).*

## 1. Bài toán và thước đo

- Hỏi đáp tiếng Việt trên dữ liệu bảng bẩn (CSV / Excel / SQLite), mỗi DN một cấu trúc, rulebook nghiệp vụ. **PA = (đúng − sai) / N**,
  bỏ trả lời = 0. Train 30 DN / 3000 câu; public 14 DN / 1000 câu (không trùng DN với train); private: có **loại DN mới**.
- Huấn luyện trên 24 DN train; 6 DN giữ ngoài (`tr_ho`, 631 câu + 1195 paraphrase); `pub_ho` = 1373 paraphrase câu public.
- **Tập đo sạch (13/11):** Qwen tự chấm tương đương câu gốc ↔ paraphrase (2 prompt, không ai xem từng câu) → `pub_ho_clean` 1171 / 1373,
  `tr_ho_clean` 1711; tập con `pub_ho400c` (348), `tr_ho400c` (375). **~10 điểm của `pub_ho` cũ là nhãn sai** (paraphrase đổi nghĩa).
- Nhiễu một lần chạy: public ±0.4, held-out ±0.3 → quyết định cần ≥ 2 seed. Giữ / bỏ: `docs/12_PLAN2.md` §4.

## 2. Hệ đang nộp: hybrid P (luật tổng quát hỗ trợ, mô hình viết mọi chương trình)

| Mốc | Public | Held-out |
|---|---|---|
| Kế hoạch 1, M0 (súp WF1–4 × 0.35) | 0.934 | gộp 0.752 |
| O + WG1 × 0.35 (11/11) | 0.9855 | gộp 0.778 |
| **P + WG1 × 0.35 (12/11)** = O + 3 kiểm tra tín hiệu sạch (`count_check`, `size_use_check`, `ticket_avg_check`) | **0.993 / 0.994 / 0.993** (3 seed; container 0.994) | gộp 0.7748; **sạch: `pub_ho400c` 0.698, `tr_ho400c` 0.923** |

## 3. Kết quả tích cực

1. **Luật tổng quát làm ngữ cảnh + kiểm tra** (hướng R, 17 vòng): 0.934 → 0.993 public với cùng loại mô hình. Đòn bẩy lớn nhất là
   gợi ý / làm sạch câu và kiểm tra lỗi được chọn bằng **"tín hiệu sạch"** (chỉ nhận kiểm tra không chạm mẫu đúng nào trên mọi log).
2. **Dữ liệu paraphrase > kỹ thuật huấn luyện phức tạp:** WG1 (D1 + 3818 paraphrase) hơn súp, RFT, DPO. D2b (paraphrase kiểu câu mới,
   WG11) nâng held-out (0.784, cao nhất) nhưng public 0.987 < WG1.
3. **Mô hình thuần (không luật lúc suy luận, chỉ đọc dữ liệu + toolkit; `docs/14`)**: Q0 0.841 → tốt nhất **QS7 0.905** public
   (tự gợi ý + câu nhiễu); câu nhiễu (D2N) giúp cả public lẫn held-out; "tự gợi ý" (nhãn bắt đầu bằng gợi ý luật đọc từ câu) +1.8–2.2.
4. **Chuỗi checkpoint qua acc hoạt động** (QS10 26M token: 59 bước acc A → 60 bước acc B; v3a: 22 → 51 → 77) — huấn luyện dài không
   bị giới hạn bởi quota một acc.
5. **Học luật tự động (kế hoạch 3, vòng 0):** Qwen tự đề xuất 178 luật từ 598 ca sai trên câu train; quét tín hiệu sạch nhận **4 luật
   kiểm tra** (bắt 39 / 21 / 9 / 4 mẫu sai, 0 mẫu đúng) — cùng tiêu chí đã dùng tay ở vòng 17, nay tái lập được không cần Claude.
   Biên dịch rulebook thành thẻ luật (111 thẻ public, 79 train) khớp đúng thuật ngữ mới (vd "Khách sộp nhất" → `by="aov"`, ≥ 3 đơn).

## 4. Kết quả tiêu cực và hạn chế

1. **Số đo của P lạc quan** (`docs/15` §4.2): luật viết tay được chỉnh theo lỗi public (dùng đáp án public) và viết trước khi tách 6 DN
   held-out → public 0.994 / held-out của P không ước lượng được private (có ngành mới).
2. **Mô hình thuần bão hoà ~0.89–0.905 public**: vòng 2 (RFT QS9, 26M token QS10: 0.889 / 0.896) không hơn vòng 1. Khoảng cách tới P
   nằm ở câu kiểu public (sạch: QS10 0.606 vs P 0.687 trên `pub_ho400c`; câu train ≈ ngang: 0.909 vs 0.92).
3. **Lệch phong cách train ↔ test**: kỳ viết tắt "T4/2025", "Th12/25" ở 25.9 % câu public, **0 %** câu train; QS7 PA 0.845 trên nhóm
   này vs 0.934 phần còn lại (P 0.997 vì `qparse` viết cho đúng kiểu này). Lỗi thuần còn lại theo phần chương trình khác P (QS10, 48 sai):
   kỳ 24, thực thể / tham số 32, hàm 17.
4. Prompt của mô hình thuần **không chứa định nghĩa rulebook public** (chỉ dòng mới so với train + public) → thuật ngữ như "khách sộp"
   chỉ đến được qua gợi ý luật (P) hoặc thẻ biên dịch (kế hoạch 3).
5. Mô hình nộp chỉ học 70.6 % câu train (bỏ 6 DN held-out và 250 câu không có chương trình đã kiểm); nhãn đến từ 55 mẫu luật → học
   sinh bắt chước độ phủ của luật (loss 0.03, phải trộn × 0.35).
6. Học luật tự động: 171 / 175 luật quy nạp bị loại (độ chính xác 0.09–0.17: điều kiện quá rộng); thuật ngữ mới do mô hình sinh nhiều
   khi không mới hoặc định nghĩa lệch chương trình → `tr_newterm` chỉ là chỉ báo thô.
7. WG5–8 (seed, r 64, lr 1e-4), WG9–12 (dữ liệu khác) không hơn WG1 ở public.

## 5. Ý nghĩa khoa học

- Với mô hình ≤ 10B, **ngữ nghĩa trong toolkit đã kiểm + LLM viết chương trình ngắn** đạt 0.99 public; phần còn thiếu của mô hình thuần
  là **hiểu cách nói của người dùng thật** (kỳ viết tắt, lóng, thuật ngữ rulebook), không phải tính toán.
- **Thước đo held-out phải được làm sạch** trước khi so hệ (10 điểm nhiễu nhãn có thể đảo kết luận); một bộ chấm tương đương do chính
  mô hình làm là đủ và không cần người.
- **Luật có thể do mô hình học**: cùng quy trình "đề xuất → thống kê trên log → nhận khi chỉ chạm mẫu sai" thay được người viết luật;
  biên dịch rulebook lúc suy luận mang định nghĩa của ngành mới vào prompt mà không cần đáp án.

## 6. Điều kiện kiểm chứng

| Kết luận | Cách kiểm |
|---|---|
| Điểm mọi lần chạy | `Process/scores.csv`; tính lại: `python3 Process/tools/score_sets.py <run dir>` |
| P trên tập sạch | `Process/runs/p3pbase/` (submission nén) + `tests/splits/{pub,tr}_ho400c.txt` |
| Tập sạch | `Process/runs/pj/PJ/judge.jsonl.gz`, `tests/splits/*_clean.txt` |
| Lỗi thuần so với P | `python3 Process/tools/cmp_pure_p.py <logs thuần> <logs P>`; `period_share.py` |
| Luật quy nạp / quét | `Process/runs/lrl0d/LRL0/induce*/`, `python3 tests/lrl_accept.py scan …` (`training/data/out/lrl0/scan_report.json` dựng lại được) |
| Rò rỉ / phủ dữ liệu | `docs/15_AUDIT.md` §7 |
| Test CPU | `bash tests/run_all.sh` → `ALL PASS` |
