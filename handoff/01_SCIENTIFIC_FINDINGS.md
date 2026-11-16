# PDF 1 — Scientific Findings (DataMate, đến 16/11/2026)

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


## 9. Kế hoạch v2 — chế độ H, không hàm ngành viết tay (14/11 17:00 → 15/11 06:00; `handoff/04`–`07`)

**Bối cảnh.** Bản nộp chạy **chế độ H** (`hide_api: industry`: mọi hàm `industry.py` viết tay bị ẩn); mọi cải tiến phải là năng lực mô hình
tự học. Từ 15/11 01:15: chỉ đo trên public. Số liệu: `Process/scores_v2.csv`, log `Process/runs/v2/`, nhật ký `handoff/06_PUBLIC_GRID.md`,
bảng tổng hợp `python3 Process/tools/v2/grid_report.py <runs>`. *ensemble* = file phán xử; *container* = file Docker ghi (trước bản sửa 04:40
thấp hơn ensemble 0–0.013 do lỗi gộp, §9.3).

### 9.1 Kết quả đã đo (public 1000 câu)

| Hệ | ensemble | container |
|---|---|---|
| Kế hoạch 5 còn `industry.py` (D1, tham chiếu) | 0.952 | 0.951 |
| **Mốc chế độ H** `flow_p5_final_h` (H0a / H0a2) | **0.863 / 0.863** | 0.860 / 0.859 |
| WT-b v2 viết hàm (pwtbv2) | **0.864** | 0.859 |
| S′ seed 1 / 2 | 0.861 / 0.856 | 0.846 / 0.843 |
| k_max 24 / S viết hàm 16 mẫu (phu16) | 0.857 / 0.853 | 0.852 / 0.847 |
| WT-b / WT-s / WT-b + k24 / u24 / min_agree 1 / 8 mẫu / W1 sinh lại | 0.855 / 0.853 / — / 0.848 / 0.851 / 0.846 / 0.847 | 0.838–0.855 |
| F (khối API tự sinh) / S′ + WT-b | 0.847 / 0.841–0.855 | 0.842 / 0.830–0.842 |
| **Không hàm cửa hàng** (phnou, đối chứng) | 0.834 | 0.831 |
| Qwen gốc làm bộ giải (pbase) | — | 0.753 |
| Phán xử lại log H (jh), ngưỡng θ / margin | 0.849–0.866 | — |

### 9.2 Kết luận tích cực (có bằng chứng)

| Kết luận | Bằng chứng |
|---|---|
| **Hàm cửa hàng do mô hình viết là đòn bẩy thật ở chế độ H**: bỏ hàm mất 0.029–0.032 (ghép cặp z −3.0) | phnou vs H0a / H0a2 |
| Nút thắt là **chất lượng mẫu ở câu ngành**, không phải số mẫu / cách chọn: ở câu khó mỗi mẫu đúng ~0.20 (ngành) vs ~0.60 (không ngành); gộp 2 lần chạy (≈ k32) 0.857; k24, u16 không giúp; phán xử / ngưỡng 0.849–0.866 | `var_acc2.py`, `pool.py`, phk24, phu16, jh |
| **Lỗi tập trung ở 2 DN**: dev_014 (điện thoại, 35 / 63 câu hỏng) và dev_012 (sửa xe, 22 / 50) ≈ 57 / ~93 câu hỏng. Ba loại dòng rulebook gây lỗi: (a) một dòng định nghĩa **nhiều đại lượng** ("Thu máy cũ" định nghĩa cả "khách phải trả"); (b) dòng định nghĩa **cách nhóm** ("Dòng máy", "hãng"); (c) đại lượng **tại một ngày** ("Trả góp": còn nợ / quá hạn tới ngày X); thêm cột riêng của ngành (hoa hồng thợ, biển số, ô "Làm gì") | phân tích log H0a theo DN |
| Cùng vài hàm sai lặp lại ở **mọi** bộ viết (đếm xe khác nhau 0 / 32, tỉ lệ bảo hành 0 / 18, khách phải trả 0 / 46) → cần viết hàm khác cấu trúc (nhiều hàm / dòng, thấy câu hỏi thật), không phải thêm mẫu | phu16, pwtbv2, H0a |
| **Lỗi gộp container** (phán xử quyết định bỏ → file cuối lấy đáp án S) làm mất 0.003–0.013 ở mọi lần chạy; đã sửa (`merge_sub first_decides`, test CPU) | so file container vs ensemble trên 14 lần chạy |
| Kiểm ngược dữ liệu sinh bằng đường mô hình thứ hai lọc được nhãn nhiễu: S đồng ý với 709 / 1,336 cặp UQ (535 bất đồng, 92 bỏ) | uqchk_a/b |

### 9.3 Kết luận tiêu cực (đóng hướng)

| Hướng | Bằng chứng | Kết luận |
|---|---|---|
| **WT (LoRA viết hàm chưng cất từ Qwen gốc)** | offline tốt hơn (0.596 vs 0.517, z 2.8) nhưng public ≤ mốc ở 9 lần chạy; hàm "lười" gọi hàm toolkit sai nghĩa (`so_sua` = `top_customers`) vẫn qua kiểm thực thi; mẫu gọi hàm WT-b đúng 0.149 (hàm S 0.514); WT-b v2 chỉ bằng mốc | **đóng** (đã ngắt pwv2uvb, phunb) |
| Thêm mẫu (k24, u16, gộp lần chạy) | 0.853–0.857 | đóng |
| S′ (đọc định nghĩa văn xuôi + tương phản) | 0.861 / 0.856 | đóng |
| F, A (phán xử ràng buộc), B-lite, WJR2D, Qwen gốc làm kênh 2 | 0.753–0.855 | đóng |
| Số đo trên phân phối nguồn (dòng held-out DN train) dự báo sai phân phối đích (rulebook DN mới) | WT-b | bài học phương pháp |
| Kiểm thực thi (chạy được, không suy biến, tự nhất quán) **không kiểm nghĩa** | hàm sai qua kiểm ở mọi bộ viết | cần tín hiệu nghĩa độc lập (I6) |

### 9.4 Giai đoạn C (06:05 → 11:15): chất lượng hàm cửa hàng + S″ — kết quả

Tất cả là năng lực mô hình, không luật viết tay. Ghép cặp so với H0a (ensemble; nhiễu giữa 3 lần chạy mốc ≈ ±0.004, giữa 2 biến thể gần giống ≈ ±0.01).

| Ý tưởng | Lần chạy → public (z) | Kết luận |
|---|---|---|
| Mốc sau sửa gộp | pfix 0.859 | container = ensemble; mốc 0.859–0.863 |
| I2 S thấy mã nguồn hàm (`units_code`) | phcodeb 0.867 (+0.4); phmqc 0.862 (−0.1, shard hỏng) | trong nhiễu |
| I3 ví dụ tự sinh của cửa hàng (`unit_examples`) | phux 0.857 (−0.6); phux2n **0.869** (+0.7); phux2b (+I2) 0.855 (−0.9) | **chưa được thử thật**: kiểm thực thi chỉ giữ 14–23 ví dụ cho 2–4 / 15 cửa hàng; phux2n / phux2b trái chiều = nhiễu |
| I4 / I4b 1–3 hàm / dòng (`units_multi`, `unit_multi_v`) | phmq 0.859 (28 hàm, pub_ind 0.617); phmq2 0.856; phmq2x 0.850 | nhiều hàm hơn, pub_ind nhích, tổng = mốc |
| I5 câu thật của cửa hàng cho bộ viết (`units_qctx`) | phqcb 0.849 (−1.6) | không giúp |
| I6 S tự kiểm hàm, bỏ hàm bị phủ định (`unit_verify`) | phuvb 0.855, phmq2uv 0.853, phmquv 0.840*, phall 0.850*, phquvx 0.839 (**−2.7**) | **luôn ≤ mốc** → đóng; bỏ cả hàm đúng (S không có hàm sai cùng kiểu) |
| Bộ viết hàm có suy nghĩ + budget forcing | phut 0.839 (**−2.3**; sai 46 → 65) | đóng (phmq2t ngắt) |
| S″-A / S″-A′ (UQ lọc bằng uqchk, LoRA chồng V3A × 0.70) | es3ar 0.863 (+0.000), es3fr 0.856 (−0.7) | = mốc: thêm 709 cặp câu gọi hàm không đổi hành vi S |

\* shard 2 / 4 của bộ viết hàm thoát rc 1 ở 4 lần chạy I4 + I5 (phmqx 0.840, phmqc, phmquv, phall) → mất hàm dev_011, nửa dev_012; không kết luận
được. Đã sửa ghi log (traceback của shard hỏng vào `eval.log`); nguyên nhân chưa rõ (không tái hiện trên CPU).

**Kết luận giai đoạn C:** 18 lần chạy, không biến thể nào vượt mốc có ý nghĩa thống kê; giới hạn nằm ở **ngữ nghĩa** hàm S viết cho 2 cửa hàng
(dev_014, dev_012), không ở cách trình bày / kiểm / số lượng hàm. Cấu hình nộp giữ `configs/flow_v2_final.json`. Hướng còn mở: tăng sản lượng
ví dụ I3 (hiện ≤ 23) để thử thật; mô hình viết hàm mạnh hơn (`docs/proposed_method_1_b.md`, `docs/execution_plan_0`).

### 9.5 Điều kiện kiểm chứng

| Kết luận | Cách kiểm |
|---|---|
| Điểm mọi lần chạy | `Process/scores_v2.csv`; `python3 Process/tools/v2/grid_report.py <runs dir>` (cần `data/raw/public_gt.jsonl`) |
| Tỉ lệ đúng khi gọi hàm, theo hàm | `Process/tools/v2/unit_use.py`, `unit_use2.py`, `unit_final.py` trên `Process/runs/v2/<run>/` (giải nén) |
| Lỗi gộp | `Process/runs/v2/h0a/h0a/flow_p5_final_h/submission.jsonl.gz` vs `judge/ensemble/submission.jsonl.gz` |
| Đóng góp của hàm | phnou vs H0a (ghép cặp, `grid_report.py`) |

## 10. Kế hoạch P5 — zero-shot, thay mô hình lõi, ghép component (15/11 08:00 → 16/11 09:00; `docs/19_PLAN_P5_ZS.md`, `docs/21_COMBO_PLAN.md`)

Giao thức: public 1000 câu, đường container (`predict_flow`), **ghép cặp từng câu** (z) với baseline đúng loại. Zero-shot: B00-S
(Qwen3.5-9B gốc, chế độ H) 0.753 / 0.755. Tinh chỉnh: F = `flow_p5_final` (D1 0.951, C0L 0.942); F_H = `flow_p5_final_h` (0.863, 0.863,
0.858, 0.857 — nhiễu giữa lượt ±0.003, nên ngưỡng "tốt hơn" ≈ z ≥ 2).

### 10.1 Đóng góp / kết quả tích cực

| Phát hiện | Bằng chứng |
|---|---|
| **Tỉa expert theo saliency trên prompt của chính tác vụ** (REAP tự tính trên DN train) giữ được năng lực khi đưa MoE 35B-A3B xuống ≤ 15B | E88 (13.96B) zero-shot 0.781 / 0.777 (z 2.3 / 2.0 vs B00-S); E92 0.773; REAP-64 cộng đồng (tỉa trên dữ liệu chung) 0.429 — sụp |
| **Suy nghĩ trước khi viết chương trình** là đòn bẩy zero-shot lớn nhất | B00T 0.805 (+0.051, z 4.7); hành vi này nằm ẩn trong template (không truyền `enable_thinking` ⇒ suy nghĩ) và giải thích C0 0.921 vs 0.952 |
| **Phán xử không trần thời gian** giúp mô hình gốc | S01 0.776 (+0.023, z 2.8); với mô hình tinh chỉnh không áp dụng (phán xử xong trong trần) |
| **Gộp LoRA vào checkpoint block-FP8** (khử lượng tử → +ΔW → lượng tử lại 128×128) và **LoRA trên expert hợp nhất của MoE** dùng được | sai số tương đối ≤ 0.011; ΔW gộp khớp PEFT (cos ≥ 0.996); 250 module / adapter |
| **Bộ phán xử MoE ngang 9B** trên cùng ứng viên | jsj (S/L 9B, J = E88+MJ2) 0.945 vs D1 0.951 (z −0.7), vs C0L +0.003 |

### 10.2 Kết quả tiêu cực và hạn chế

| Kết quả | Bằng chứng | Ý nghĩa |
|---|---|---|
| **MoE + LoRA thua 9B + LoRA khi mọi thành phần khác giữ nguyên** | e7rw7 (F, chỉ thay mô hình + 3 LoRA) 0.927 (vs D1 z −2.1); e8rw7 (F_H) 0.836 (vs H0a z −2.8; phán xử MoE bị cắt ở trần 1800 s) | lợi thế zero-shot của E88 (+0.025) không còn sau tinh chỉnh; LoRA chỉ trên attention + shared expert (routed experts đóng băng) |
| Điểm yếu của MoE nằm ở **bộ giải / link**, không ở phán xử | jsq (S/L MoE, J = 9B+WJR1) 0.929 ≈ e7rw7 0.927 | cải thiện MoE phải nhắm bộ giải (MS3: LoRA cả expert — đang đo) |
| Các E-series đầu (E1–E8) **không đo được hiệu quả mô hình** | dựng trên framework zero-shot (bộ thẻ luật khác, `lrl_regen`, trần 5400 s) | ~1/3 khoảng cách E7 cũ do framework lệch; bài học phương pháp (PDF 2 §11.5) |
| Không component nào cải thiện F_H có ý nghĩa | T01 0.857 / 0.860; Q01 BF16 0.867 / 0.854 (×2 thời gian); N4-A 0.848; T02 (ví dụ tự sinh, kiểm vòng) 0.777; gộp ứng viên 2–3 lượt rồi phán xử 0.854–0.863 (oracle 946 → 959 nhưng bộ chọn không khai thác) | nút cổ chai là **chọn ứng viên**, không phải sinh ứng viên |
| Kiểm tra trung lập trên chương trình (N1-A chạy lại, N7-A chứng chỉ đơn vị, N3-A phản ví dụ, N6-A SQL thứ hai) | 0 câu đổi / 0.722–0.751 (âm) | mô hình gốc đặt đơn vị không nhất quán (60 % chứng chỉ bị cờ); SQL độc lập kém pandas |
| 4B làm phán xử, Qwen3-14B / 4B làm lõi, I02 (grammar ràng buộc) | 0.715–0.769; 0.709 / 0.717; 0.675 (337 phút) | đổi lõi đặc không giúp; ràng buộc grammar CPU-bound |
| Tổ hợp lai 9B + MoE (mỗi vai trò một backbone) | ≈ 23B tham số | **vi phạm giới hạn 15B** — chỉ dùng chẩn đoán |

### 10.3 Ý nghĩa khoa học

- Ưu thế zero-shot của một backbone (tỉa MoE theo tác vụ) **không dự báo** ưu thế sau tinh chỉnh LoRA cùng công thức: LoRA trên 9B đặc
  phủ toàn bộ tham số tuyến tính, còn trên MoE chỉ phủ phần dùng chung; so sánh công bằng phải giữ nguyên framework và chỉ đổi lõi.
- Hành vi ẩn của template (suy nghĩ mặc định) là một biến số lớn hơn nhiều thay đổi phương pháp (+0.03–0.05); mọi thí nghiệm phải
  cố định nó tường minh.
- Với hệ đã có oracle ~95 %, thêm ứng viên (nhiều lượt, nhiều mô hình) không tăng điểm nếu bộ chọn không đổi.

### 10.4 Điều kiện kiểm chứng

Cấu hình, job, log và bảng điểm: `configs/flow_*` (`_doc` ghi điều chỉnh tương thích), `automation/jobs/p5/`, `Process/runs/p5_results/`,
`Process/scores_p5zs.csv`; ghép cặp bằng `Process/tools/p5/jscore.py` (vs H0a / H0a2) hoặc đoạn mã trong `docs/19`. Kiểm gộp FP8:
`tests/test_merge_fp8_experts.py`, `Process/tools/p5/test_expert_lora_layout.py`. Đang chờ: MS3 (LoRA cả 88 expert) trên F_H, e8r.
