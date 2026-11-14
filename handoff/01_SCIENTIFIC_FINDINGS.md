# PDF 1 — Scientific Findings (DataMate, đến 14/11/2026)

*Phạm vi: đóng góp, ý nghĩa khoa học, kết quả tích cực và tiêu cực, bằng chứng, điều kiện kiểm chứng. Kinh nghiệm kỹ thuật /
vận hành ở PDF 2; trạng thái / vị trí / việc tiếp theo ở PDF 3 và `handoff/STATUS.md`. Số liệu: `Process/scores.csv` (kế hoạch 1–2),
`Process/scores_p3.csv` (3–4), `docs/17_PLAN4_SWV.md` §7. Tài liệu kế hoạch 1–3 (docs/08, 12–16, `EXPERIMENT_LOG.md`) đã xoá 14/11,
còn trong lịch sử git.*

## 1. Bài toán và thước đo

- Hỏi đáp tiếng Việt trên dữ liệu bảng bẩn (CSV / Excel / SQLite), mỗi DN một cấu trúc, rulebook nghiệp vụ. **PA = (đúng − sai) / N**,
  bỏ trả lời = 0. Train 30 DN / 3000 câu; public 14 DN / 1000 câu (không trùng DN với train); private: có **loại DN mới**.
- Huấn luyện trên 24 DN train; 6 DN giữ ngoài (`tr_ho`, 631 câu + 1195 paraphrase); `pub_ho` = 1373 paraphrase câu public.
- **Tập đo sạch (13/11):** Qwen tự chấm tương đương câu gốc ↔ paraphrase (2 prompt, không ai xem từng câu) → `pub_ho_clean` 1171 / 1373,
  `tr_ho_clean` 1711; tập con `pub_ho400c` (348), `tr_ho400c` (375). **~10 điểm của `pub_ho` cũ là nhãn sai** (paraphrase đổi nghĩa).
- Nhiễu một lần chạy: public ±0.4, held-out ±0.3 → quyết định cần ≥ 2 seed. Giữ / bỏ: `docs/12_PLAN2.md` §4.

## 2. Các hệ và hệ đang nộp

**Hệ nộp từ 14/11 (user 13/11: nộp hệ không luật viết tay):** S–W, `configs/flow_final.json` — Qwen3.5-9B + v3a × 0.70, W (hàm luật do mô hình
viết từ rulebook), k_max 16, bỏ phiếu thường: **public 0.938** (§5). Mốc cũ P (luật viết tay hỗ trợ, `configs/flow_p_final.json`) dưới đây.

### 2.1 P: hybrid (luật tổng quát hỗ trợ, mô hình viết mọi chương trình) — hệ nộp đến 13/11

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

## 5. Kế hoạch 4 — S–W–V, không luật viết tay (13–14/11, `docs/17_PLAN4_SWV.md`)

User 13/11 tối: ngừng phát triển luật viết tay (hints / qparse / qnorm / cờ / precheck của P); luật nếu có phải do mô hình tự viết; bản nộp
cuối là hệ không luật viết tay. Kiến trúc: **S** mô hình giải (Qwen3.5-9B + LoRA, viết chương trình trên toolkit), **W** mô hình viết luật
thực thi được lúc suy luận (dòng rulebook ngành / thuật ngữ mới → một hàm Python trên bảng của DN, chỉ giữ hàm chạy được và tự nhất quán),
**V** bộ chọn học được (hồi quy logistic trên đặc trưng bất biến của các nhóm đáp án, học từ log của S trên DN train) chọn đáp án hoặc bỏ.

### 5.1 Kết quả tích cực

| Kết luận | Bằng chứng |
|---|---|
| **Toàn bộ điểm "ngành mới" trước đây là của `industry.py` viết tay.** Ẩn mọi hàm ngành (`--hide-api industry`): trên 614 câu của 5 DN có ngành, S còn 0.476 (`pub_ind`) / 0.415 (`pub_hoind`) so với 0.900 khi có hàm viết tay | e0hvp, g1v3apf2; `Process/scores_p3.csv` |
| **W (hàm do mô hình viết từ rulebook) có tác dụng có ý nghĩa thống kê** ở ngành mới: `pub_ind` 0.476 → 0.595 (+0.119, SE ghép cặp 0.031, z 3.8; 28 câu tốt hơn / 7 kém hơn), `pub_hoind` 0.415 → 0.493 (+0.078, z 3.3) | e0hvp vs g0hfp (cùng 614 câu, cùng S, chỉ khác W) |
| W dùng chính trọng số của S tốt ngang W dùng mô hình gốc (G0f ≈ G0g) → một mô hình 9.65B đủ cho cả hai vai, đúng giới hạn 15B | G0f / G0g |
| Đo W trực tiếp (offline, dùng hàm viết tay chỉ làm thước): 3 / 9 hàm W đúng gần như mọi tháng; lỗi chủ yếu là dạng giá trị (dict nhiều đại lượng, tỉ lệ thay vì %) và lỗi API → kiểm tra trên 13 kỳ + vòng tự sửa (lỗi chạy thật đưa lại cho W) đưa số hàm dùng được 12 → 41 | `docs/17` §7 13/11 22:20, G0f |
| **V có ích** (phát lại trên log thật, cùng mẫu): held-out `pub_ho400c` 0.603 → 0.652, `tr_ho400c` 0.917 → 0.928, ngành mới không W −0.107 → +0.018 | g1v3aho, e0hv |
| **Nhiều mẫu hơn có ích:** k_max 16 (k_fast 4) thay 8: v3a 0.936 → **0.938**, V4S2 0.920 → 0.932 (bỏ phiếu thường), thời gian đánh giá +20–30 % | pva3k16, pvs2k16 |
| **Hệ nộp cuối (S–W, không luật viết tay): public 0.938** = v3a × 0.70 + W + k_max 16 + bỏ phiếu thường | pva3k16; `configs/flow_final.json` |

### 5.2 Kết quả tiêu cực và hạn chế

| Kết luận | Bằng chứng |
|---|---|
| Hệ không luật viết tay vẫn kém P ~5.6 điểm public (0.938 vs 0.994) | pva3k16 vs P |
| **v4** (học tiếp trên v3a × 0.70 với dữ liệu dùng hàm + RFT + DR) **không giúp public**: seed 2 S + W + V 0.916 (vote 0.920) < v3a 0.936; kém ở DN ngành (`pub_noind` 0.741 vs 0.830) | G1 v4 |
| V không có lợi trên public ở mọi lần chạy (v3a k 16: 0.935 vs bỏ phiếu 0.938; ngưỡng θ 0…0.6 chỉ đổi ±0.005) — lợi chỉ ở held-out → hệ nộp dùng bỏ phiếu thường | phát lại trên 7 lần chạy public |
| v3b (thêm thẻ luật trong prompt, DR, phiên âm; 19.2M token) và soup với v4 đều không hơn v3a trên public (0.925–0.932) | g1v3bpf, pv3bk16, pvsoup8/16 |
| Lỗi còn lại của v3a (39 câu không đúng ở k 8): 17 có mẫu đúng nhưng bỏ / chọn sai (bộ chọn hoàn hảo → ~0.957), 22 không mẫu nào đúng (11 ở DN ngành) → trần của hướng này là năng lực S | phân tích g1v3apf2 |
| W lấp mới ~0.12 / 0.42 khoảng cách tới hàm viết tay; W mạnh hơn (8 mẫu, 2 vòng tự sửa) không hơn W mặc định (−0.02 / −0.05 ghép cặp) | g0hfq |
| Nhiễu giữa hai lần chạy giống hệt trên 112 câu tới 0.15 (−0.036 / −0.107 / −0.188) → tập 112 câu không phân biệt được biến thể W; cần ≥ 600 câu và so ghép cặp | E0, e0hv, e0hvp |
| Huấn luyện chỉ có rulebook train **không có mục ngành nào** → không đo được W trên DN train; mọi số ngành mới đến từ 5 DN public | `shopctx.industry_of` |

### 5.3 Ý nghĩa khoa học

- Tách công: **luật do mô hình viết thành mã chạy được và kiểm bằng dữ liệu** (W) có tác dụng đo được ở ngành chưa gặp, khác với luật dạng
  văn xuôi trong prompt (kế hoạch 3: không có ích). Điểm mấu chốt là kiểm tra bằng thực thi (chạy được, không suy biến, tự nhất quán giữa
  các mẫu) và phản hồi lỗi thật.
- Đo đúng cần **so ghép cặp trên cùng câu** và tập đủ lớn; các so sánh độc lập trên 112 câu đã cho kết luận sai dấu nhiều lần.
- Một lỗi nạp lặng lẽ (đường dẫn bộ chọn trên Kaggle) làm V không chạy ở mọi lần đo trong ~8 giờ — mọi thành phần tùy chọn phải **dừng hẳn**
  khi cấu hình nêu mà không nạp được, và log phải ghi dấu của từng thành phần (`conf_sel`).

## 6. Ý nghĩa khoa học (kế hoạch 1–3)

- Với mô hình ≤ 10B, **ngữ nghĩa trong toolkit đã kiểm + LLM viết chương trình ngắn** đạt 0.99 public; phần còn thiếu của mô hình thuần
  là **hiểu cách nói của người dùng thật** (kỳ viết tắt, lóng, thuật ngữ rulebook), không phải tính toán.
- **Thước đo held-out phải được làm sạch** trước khi so hệ (10 điểm nhiễu nhãn có thể đảo kết luận); một bộ chấm tương đương do chính
  mô hình làm là đủ và không cần người.
- **Luật có thể do mô hình học**: cùng quy trình "đề xuất → thống kê trên log → nhận khi chỉ chạm mẫu sai" thay được người viết luật;
  biên dịch rulebook lúc suy luận mang định nghĩa của ngành mới vào prompt mà không cần đáp án.

## 7. Điều kiện kiểm chứng

| Kết luận | Cách kiểm |
|---|---|
| Điểm mọi lần chạy | `Process/scores.csv`; tính lại: `python3 Process/tools/score_sets.py <run dir>` |
| P trên tập sạch | `Process/runs/p3pbase/` (submission nén) + `tests/splits/{pub,tr}_ho400c.txt` |
| Tập sạch | `Process/runs/pj/PJ/judge.jsonl.gz`, `tests/splits/*_clean.txt` |
| Lỗi thuần so với P | `python3 Process/tools/cmp_pure_p.py <logs thuần> <logs P>`; `period_share.py` |
| Luật quy nạp / quét | `Process/runs/lrl0d/LRL0/induce*/`, `python3 tests/lrl_accept.py scan …` (`training/data/out/lrl0/scan_report.json` dựng lại được) |
| Rò rỉ / phủ dữ liệu | `docs/15_AUDIT.md` §7 (lịch sử git, trước 14/11) |
| Test CPU | `bash tests/run_all.sh` → `ALL PASS` |
| Kế hoạch 4 (S–W–V) | `Process/scores_p3.csv` (mọi lần chạy 13–14/11), `docs/17_PLAN4_SWV.md` §7; log đã lưu `Process/runs/` (g1v3apf2, pva3k16, e0hvp, g0hfp …) |
| W có tác dụng (ghép cặp) | `tests/splits/pub_ind.txt`, `pub_hoind.txt`; so từng câu e0hvp vs g0hfp (cùng S, hàm ngành ẩn: `configs/flow_sv_h.json` vs `flow_swv_h.json`) |
| Hệ nộp | `configs/flow_final.json`, adapter `nhatlong1103/zc-final-adapter` (sha256 `src/adapter_hashes.json`); chạy lại: `training/soup_job.py --runner predict` |
