# PDF 1 — Scientific Findings (DataMate, đến 15/11/2026)

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
| **Hệ nộp cuối (S–W, không luật viết tay): public 0.938** (đường eval) / **0.922** (đường container, lần chạy khác: dao động giữa lần chạy ~±0.01) = v3a × 0.70 + W + k_max 16 + bỏ phiếu thường | pva3k16, f2a3; `configs/flow_final.json` |

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

## 8. Kế hoạch 5 — hai LoRA huấn luyện riêng (S giải / W viết luật – nối luật – phán xử), 14/11 (`docs/18_PLAN5_TWO_LORA.md`)

| Kết luận | Bằng chứng (public, đáp án chuẩn) |
|---|---|
| **Nút thắt là chọn ứng viên + kiến thức rulebook, không phải năng lực sinh**: gộp 11 lần chạy có mẫu đúng ở 991 / 1000 câu, bỏ phiếu chỉ 0.931–0.947 | phát lại log kế hoạch 3–4 |
| **W3 phán xử huấn luyện** (LoRA trên S, dữ liệu = chính các mẫu của S trên DN train) chỉ phán câu có mẫu bất đồng: +0.010–0.015 ổn định trên nhiều log S | WJB1 0.950 / 0.946 vs bỏ phiếu 0.938 / 0.936 |
| Phán xử zero-shot đã giúp khi **chỉ phán câu bất đồng** (0.947); phán mọi câu thì hỏng câu đúng (0.941) | zjs / zjb |
| **Đưa dòng rulebook vào prompt phán xử** (huấn luyện kèm dòng — nhãn bộ khớp P trên câu train — và lúc chạy dòng do linker W2 nối) là cải tiến lớn thứ hai | WJR1 tổ hợp 3 biến thể: 0.949–0.958 (TB 0.952) vs WJB1 ~0.948–0.950 trên cùng 4 log |
| Linker W2 huấn luyện (WL1) chính xác hơn hẳn zero-shot (0.78 vs 0.24 so với bộ khớp P) | lkw1 |
| Dòng rulebook trong prompt **S** không giúp (WL1: 0.940 ≈ mốc) hoặc hại (zero-shot: 0.921) — dòng chung chung kèm hàm kéo sai | zlk, zlb, zlw |
| W1 bind (mô hình viết lời gọi toolkit cho từng dòng rulebook, chỉ giữ khi chạy được trên dữ liệu ngành): 56 / 125 dòng | w1b2 |
| Huấn luyện riêng từng vai cho phép quy công: phán xử (W3) đóng góp phần lớn; tăng cường từ chối (WJA1) và point (WJP1) không hơn | E1 |
| Đường container: 0.943 / 0.941 (2 lần, 984 / 1000 đáp án trùng) — đủ tất định cho luật ±0.03 | F5 |

**Hạn chế:** chọn ngưỡng / tổ hợp trên chính public (4 log) → lạc quan ~0.003–0.005; ~10 câu mọi mẫu S cùng sai (câu chủ quan, thuật ngữ rulebook mà
S không gọi đúng hàm) nằm ngoài tầm phán xử; phán xử chọn sai ~10 câu bất đồng.


## 9. Kế hoạch v2 — chế độ H, không hàm ngành viết tay (14/11 17:00 → 15/11 04:00; `handoff/04`–`07`)

**Bối cảnh.** User 14/11 17:00: bản nộp chạy **chế độ H** (`hide_api: industry` — mọi hàm `industry.py` viết tay bị ẩn khỏi S, thẻ luật, link,
phán xử, W1); mọi cải tiến phải là năng lực mô hình tự học. User 15/11 01:15: dừng held-out, thử nhiều giải pháp trên public, chọn điểm
cao nhất. Số liệu: `Process/scores_v2.csv` (mọi lần chạy), log `Process/runs/v2/`, nhật ký `handoff/06_PUBLIC_GRID.md` §5.

**Hai con số mỗi lần chạy:** *container* = `<config>/submission.jsonl` (file Docker sẽ ghi) và *ensemble* = `judge/ensemble/submission.jsonl`
(quyết định của bộ phán xử). Container luôn thấp hơn 0.000–0.013 vì một lỗi gộp (§9.3, PDF 3 §5 bước 1).

### 9.1 Kết quả (public 1000 câu, chế độ H)

| Hệ | ensemble | container | đúng / sai / bỏ (container) |
|---|---|---|---|
| Kế hoạch 5 có `industry.py` (D1, tham chiếu) | 0.952 | 0.951 | 973 / 22 / 5 |
| **Mốc chế độ H** `flow_p5_final_h` (H0a / H0a2) | **0.863 / 0.863** | **0.860 / 0.859** | 915 / 55 / 30 |
| F: khối API tự sinh (F1) | 0.847 | 0.842 | |
| S′ seed 1 / 2 (học đọc `# RULES` + cặp tương phản) | 0.861 / 0.856 | 0.846 / 0.843 | |
| k_max 24 (phk24) | 0.857 | 0.852 | 911 / 59 / 30 |
| WT-b viết hàm (pwtb) / 8 mẫu + 2 vòng sửa (pwtb8) | 0.855 / 0.846 | 0.855 / 0.838 | |
| WT-s (pwts) / WT-b + k24 / WT-b u24 / WT-b min_agree 1 | 0.853 / — / 0.848 / 0.851 | 0.846 / 0.849 / 0.846 / 0.848 | |
| S′ + WT-b (ps1b / ps2b) | 0.855 / 0.841 | 0.842 / 0.830 | |
| Phán xử lại log H (jh: list+c, point+c, list+think …; θ, ngưỡng) | 0.849–0.866 | — | |

**Không giải pháp nào vượt mốc.** Các lần chạy còn đang chạy lúc bàn giao (kết quả ghi vào `handoff/06` §5 khi xong): pwtbw1r, pbase,
pwtbv2 (đều WT-b → dự kiến kém mốc), **phu16** (S viết hàm 16 mẫu), **phnou** (không hàm, đối chứng), S″-A huấn luyện, uqchk_a/b.

### 9.2 Kết luận khoa học

| Kết luận | Bằng chứng |
|---|---|
| **Ẩn `industry.py` mất ~9 điểm** (0.952 → 0.863); 76 / 93 câu chưa đúng nằm ở 5 DN có mục ngành (`pub_ind` 0.606, `pub_noind` 0.125) | H0a vs D1 |
| **Nút thắt là chất lượng mẫu ở câu ngành, không phải số mẫu hay chọn mẫu**: ở 143 câu khó (vòng 2) mỗi mẫu đúng ~0.20 (ngành) / ~0.60 (không ngành); direct ≈ think | `Process/tools/v2/var_acc2.py` trên H0a, H0a2 |
| Thêm mẫu không giúp: gộp mẫu 2 lần chạy (≈ k 32) bỏ phiếu thuần 0.857 vs 0.851 / 0.855; k24 thêm cả đúng lẫn sai (911 / 59 vs 915 / 55) | `pool.py`, phk24 |
| Phán xử / ngưỡng không phải đòn bẩy ở chế độ H: mọi tổ hợp 0.849–0.866 trên cùng log | jh, θ-grid |
| **Hàm do mô hình viết là đòn bẩy, nhưng chỉ khi đúng nghĩa**: mẫu S gọi hàm do S viết đúng 0.514; gọi hàm WT-b đúng **0.149**; câu mà đa số mẫu gọi hàm: S-hàm 13 đúng / 4 sai, WT-b-hàm 7 đúng / 10 sai | `unit_use.py`, `unit_final.py` trên H0a, pwtb |
| **WT-b chính xác hơn trên dòng held-out của DN train (0.596 vs 0.517, z 2.8) nhưng sai nghĩa trên rulebook public** — viết hàm "lười" gọi hàm toolkit không liên quan (`so_sua` = `top_customers`, `thu_may_cu` = số đơn) vẫn qua kiểm thực thi 13 kỳ + tự nhất quán. Đo trên phân phối train không dự báo được phân phối DN mới; kiểm thực thi không kiểm *nghĩa* | ewuwtb vs pwtb |
| Bộ viết hàm dao động giữa lần chạy (14 → 12 hàm cùng cấu hình); public có 31 dòng ngành, chỉ ~12–14 thành hàm; dòng định nghĩa khái niệm (Dòng máy, hãng, Trả góp) cần tham số không phải kỳ → bị loại ở kiểm tra | H0a, phk24; `units.verify` |
| Dữ liệu do Qwen sinh (UQ: 1,336 câu – chương trình từ 907 hàm thầy, chạy được) có ~15 % sai kỳ khi Claude kiểm 30 mẫu → cần kiểm ngược bằng đường mô hình thứ hai trước khi dạy S | `training/data/out/s3/uq`, `handoff/07` §8 |

### 9.3 Kết quả âm / hạn chế

- **S′** (học đọc định nghĩa văn xuôi + cặp tương phản): không tăng public — dạy đọc văn xuôi không đủ, phải gắn với lời gọi thực thi được.
- **WT** (LoRA viết hàm, chưng cất từ Qwen gốc + lọc thực thi): tốt hơn offline, kém hơn trên public → **loại**.
- **F** (khối API tự sinh), **A** (phán xử ràng buộc định dạng), **B-lite** (phán xử xem lại câu đồng thuận), **WJR2D** (phán xử học tương phản): loại.
- **Lỗi gộp container:** `merge_sub.read` bỏ mọi dòng abstain, nên khi bộ phán xử quyết định bỏ trả lời, file cuối lấy đáp án của S → mất
  0.003 (H0a), 0.004 (H0a2), 0.007 (pwts), 0.013 (ps1b). Sửa đúng gốc: file phán xử là nguồn quyết định cho mọi câu nó phủ (kể cả abstain).
- Số liệu chế độ H chỉ trên 1 public (1000 câu, nhiễu một lần chạy ±0.005–0.008); chọn cấu hình trên chính public → lạc quan nhẹ.

### 9.4 Ý nghĩa

- Với ngành chưa gặp, "hàm do mô hình viết" chỉ có ích khi **đúng nghĩa**; kiểm thực thi (chạy được, không suy biến, tự nhất quán) lọc lỗi cú
  pháp / API nhưng không lọc lỗi nghĩa. Cần tín hiệu nghĩa độc lập (ví dụ hai đường mô hình khác nhau cùng ra một giá trị — cách đang thử
  với uqchk) thay vì chỉ thêm mẫu.
- Chưng cất từ thầy cùng cỡ (Qwen gốc) + lọc thực thi tăng độ chính xác trong phân phối nhưng học cả "đường tắt" của thầy; đánh giá phải
  trên phân phối mục tiêu (rulebook DN mới), không chỉ dòng held-out cùng nguồn.

### 9.5 Điều kiện kiểm chứng

| Kết luận | Cách kiểm |
|---|---|
| Điểm mọi lần chạy v2 | `Process/scores_v2.csv`; tính lại `python3 Process/tools/score_sets.py <submission.jsonl>` (cần `data/raw/public_gt.jsonl`) |
| Tỉ lệ đúng khi gọi hàm | `python3 Process/tools/v2/unit_use.py <dir có cards_unit.jsonl + solver_logs.jsonl>` (giải nén từ `Process/runs/v2/<run>/`) |
| Lỗi gộp container | so `Process/runs/v2/h0a/h0a/flow_p5_final_h/submission.jsonl.gz` với `.../judge/ensemble/submission.jsonl.gz` (19 câu khác: phán xử bỏ, file cuối trả lời) |
| Độ chính xác WT offline | `Process/runs/v2/ewuref`, `ewuwtb`, `ewuwts` (`tests/wt_accuracy.py`) |
