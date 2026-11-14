# Proposed plan v2: hệ không luật viết tay, đo trung thực, hướng tới 1.0

*DataMate, Zalo AI Challenge 2026 · bản 2, sửa 14/11/2026 17:00 UTC (bản 1: 16:10 UTC) · nguồn: `handoff/04_PROPOSED_PLAN.md` · kế hoạch chạy
từng bước: **`handoff/05_RUN_PLAN.md`** · đóng băng container 16/11 04:00 UTC*

**Bản 2 thay đổi gì so với bản 1 (theo yêu cầu user 14/11 17:00).**
1. **E bỏ danh sách lấy từ câu public sai.** Cặp tương phản do mô hình sinh có hệ thống từ cấu trúc chương trình train. Không còn trục nào chọn từ lỗi
   public ("4/26", "đh", "z", lô / máy / nhập…).
2. **C thêm ca "định nghĩa nằm ngoài toolkit".** Nhãn là pandas trên `C` / `T` do mô hình viết và kiểm bằng thực thi. S′ thấy hàm do mô hình viết
   trong prompt (`# SHOP FUNCTIONS`).
3. **Có lại cổng held-out và chế độ đo trung thực "H"** (ẩn `industry.py`, hàm ngành Claude viết theo rulebook public). Mọi quyết định nhận / bỏ phải
   qua held-out; số chính là `pub_full_H`, `pub_full` (còn `industry.py`) chỉ là tham chiếu.
4. **Giải pháp mới G: LoRA viết tool (WT)**, huấn luyện độc lập, thay `industry.py`. §3.G nêu lý do chọn LoRA thay vì Qwen gốc.

**Tóm tắt.** Điểm 0.952 của cấu hình cuối kế hoạch 5 có phần do **hàm viết tay**: `configs/flow_p5_final.json` không đặt `hide_api`, nên S vẫn gọi
được `industry.py` (đúng 3 ngành của rulebook public). Khi ẩn các hàm này, 269 câu public của 5 DN có ngành rơi từ 0.900 xuống 0.476. Có hàm
do mô hình viết (W) thì lên 0.595. Ước lượng `pub_full_H` ≈ 0.86–0.89 (đo ở bước H0). Trên held-out paraphrase public hệ chỉ đạt 0.60–0.65.
Kế hoạch v2 vì vậy làm ba việc:
- đo trung thực (H + held-out);
- thay `industry.py` bằng hàm do một LoRA viết tool tạo ra từ rulebook lúc suy luận;
- giữ các sửa của bản 1 (A sửa đầu ra phán xử, B định tuyến, C S′ đọc rulebook, F API tự sinh), sau khi bỏ mọi phần xây từ lỗi public.

Mục tiêu thực tế: `pub_full_H` 0.91–0.95 và held-out không giảm. 1.0 không được bảo đảm (§6).

## 0. Bản tóm tắt để chạy

**Mốc.** `configs/flow_p5_final.json`: public TB 0.952 trên 4 log S (`f5a`, `f5b`, `pva3k16`, `g1v3apf2`). Đáp án chuẩn là `score 1.jsonl`, chỉ
dùng để chấm. Đóng băng 16/11 04:00 UTC. Lịch giờ, acc, lệnh: `handoff/05_RUN_PLAN.md`.

| Thứ tự | Việc | Làm gì | Đo ở đâu | Nhận khi |
|---|---|---|---|---|
| 0 | **H0 đo trung thực + Docker dự phòng** | `flow_p5_final_h` (= cấu hình cuối + `hide_api: industry`) trên `pub_full` (đường container), held-out `pub_ho400c`, `tr_ho400c`, `pub_hoind`, `tr_newterm`; bản thường trên cùng held-out làm tham chiếu; D1 container `flow_p5_final` | — | có số mốc H cho mọi tập |
| 1 | **A** đầu ra phán xử | `list` / `point` sinh có ràng buộc (`<choice>` hoặc xác suất token lựa chọn) | phán xử lại 4 log public + log H0 (cả held-out) | tỉ lệ đọc được ≥ 95 %; TB tăng, không log nào giảm, held-out không giảm > 0.005 |
| 1 | **F** khối API tự sinh | `# API` sinh từ `api_spec` thay `dm.API_DOC` viết tay | một lần chạy `pub_full_H` + held-out, so ghép cặp | không âm ở cả hai |
| 2 | **Dữ liệu (CPU + GPU sinh, chỉ DN train, trừ 6 DN held-out)** | (i) rulebook đối chứng (xoá / đổi định nghĩa, cả ngoài toolkit / đổi tên / dòng nhiễu); (ii) cặp tương phản sinh có hệ thống; (iii) dòng rulebook tổng hợp kiểu "ngành mới" + hàm do thầy viết (Qwen gốc có suy nghĩ, tự nhất quán, kiểm 13 kỳ) | — | nhãn qua kiểm thực thi; thống kê tỉ lệ giữ |
| 3 | **G** LoRA viết tool WT | 2 nhánh độc lập: WT-b (trên Qwen gốc), WT-s (trên V3A × 0.70); so với Qwen gốc có suy nghĩ và W = S | độ chính xác hàm trên DN held-out; 614 câu `pub_ind` + `pub_hoind` ở chế độ H | hơn W hiện tại có ý nghĩa (z ≥ 2), held-out không giảm |
| 3 | **B** định tuyến xung đột rulebook | W2′ (dòng áp dụng + trạng thái định nghĩa) → câu đồng thuận nghi ngờ gửi phán xử WJR2 | DN train held-out, rồi log public + H0 | precision "thiếu định nghĩa" ≥ 0.9; câu hỏng ≤ 1/3 câu sửa; held-out không giảm |
| 4 | **C** S′ đọc rulebook | LoRA trên V3A × 0.70, prompt có lớp ngữ nghĩa + hàm WT; dữ liệu v3a + (i) + (ii) | `pub_full_H` + held-out, 2 seed | z ≥ 2, hai seed cùng dấu, held-out không giảm |
| 5 | **Ghép + đóng gói** | cấu hình tốt nhất qua cổng; đường container × 2; Docker; `README_SOLUTION` | — | hai lần chạy lệch ≤ 0.03; ≤ 6 h |

**Quy tắc khi chạy.**
- Không luật viết tay lúc suy luận. Hàm ngành viết tay bị ẩn trong mọi phép đo mới.
- Không dùng câu public (kể cả paraphrase) để huấn luyện. Dữ liệu mới bỏ 6 DN train held-out.
- So ghép cặp trên cùng câu; thay đổi ở S cần ≥ 2 seed.
- Mọi thay đổi phải qua cổng held-out (§5). Tuân thủ PDF 2 §9 và cổng kiểm tra trong `CLAUDE.md`.

## 1. Hiện trạng và cách phân tích

- **Hệ hiện tại** (`configs/flow_p5_final.json`): S = Qwen3.5-9B + LoRA v3a × 0.70 viết chương trình (4–16 mẫu / câu, đóng băng). W1 gắn dòng rulebook
  với lời gọi toolkit đã chạy thử. W2 (WL1) nối câu với dòng rulebook. W3 (WJR1) phán xử câu có mẫu bất đồng. Đơn vị W (`units_model: solver`)
  viết hàm cho dòng rulebook ngành / mới.
- **Phần viết tay còn chạy lúc suy luận:**

  | Phần | Mô tả | Xử lý trong v2 |
  |---|---|---|
  | `src/industry.py` | hàm của 3 ngành rulebook public (điện thoại, sửa chữa, mẹ và bé), có regex như `bim\|ta` | ẩn trong mọi phép đo (chế độ H); WT thay thế |
  | `src/dm.py` | ngữ nghĩa nghiệp vụ mặc định của phần rulebook chung, ví dụ doanh thu không VAT, đơn hợp lệ, hoà theo mã | giữ (logic toolkit, quyết định 08/11); C dạy S′ ưu tiên định nghĩa trong rulebook hơn mặc định |
  | `src/automap.py` | ánh xạ bảng bằng luật | giữ; ngoài phạm vi 34 giờ |

- **Số trung thực đã có** (`Process/scores_p3.csv`, kế hoạch 4):
  - ẩn hàm ngành: `pub_ind` (269 câu) 0.900 → 0.476; có W → 0.595;
  - `pub_hoind` (345 paraphrase) 0.415 → 0.493;
  - held-out `pub_ho400c` 0.603 (bỏ phiếu) / 0.652 (V), `tr_ho400c` 0.917 / 0.928.
- Kế hoạch 5 chưa đo held-out hay chế độ H lần nào. Ước lượng `pub_full_H` ≈ 0.952 − 269 × (0.900 − 0.595) / 1000 ≈ 0.87.
- Lỗi của bản thường (4 log, phân tích ở bản 1) vẫn đúng: 24–28 câu mỗi log, gồm 8–19 đồng thuận sai, 6–15 có mẫu đúng nhưng mất, 2–5 không mẫu
  nào đúng. Tỉ lệ đọc được phán quyết của WJR1: `list` 19–25 %, `point` 35–40 %, `point+think` 89–90 %.
- Phân tích chỉ trên public gốc và train. Held-out chỉ xem số tổng, không xem từng câu.

## 2. Root causes

| # | Nguyên nhân gốc | Biểu hiện | Quy mô | Bằng chứng |
|---|---|---|---|---|
| **RC0** | **Ngữ nghĩa ngành nằm trong hàm viết tay, không trong mô hình.** W hiện tại (zero-shot, trọng số S) viết hàm từ rulebook kém: sai dạng giá trị (dict, tỉ lệ thay vì %), sai API, sai bảng | ẩn `industry.py` thì DN ngành rơi ~0.3–0.4; ngành mới ở private không có hàm viết tay | ~0.08 `pub_full`; ở private có thể lớn hơn | `pub_ind` 0.900 / 0.476 / 0.595; G0f ≈ G0g (W = S ≈ W = Qwen gốc); W 8 mẫu + 2 vòng sửa không hơn W mặc định → **tăng tính toán lúc suy luận không đủ, cần huấn luyện** |
| **RC1** | S chưa từng được dạy dùng rulebook. Khi huấn luyện, rulebook là hằng số, không có trong prompt; nhãn đến từ 55 khuôn chương trình | tiêu chí không có định nghĩa vẫn bị trả lời; thuật ngữ có định nghĩa chỉ đúng khi linker nối trúng | 13 câu sai ở cả 4 log (6 câu tiêu chí chủ quan) + 5–7 tuỳ log | như bản 1; v3a chỉ có 0.7 % nhãn `unanswerable` (public 3.1 %) |
| **RC2** | Sai đồng thuận là vô hình. Bộ phán xử chỉ xét câu có mẫu bất đồng | 8–19 câu / log không bao giờ được xem lại | ~45 % câu chưa đúng | phán mọi câu thì hỏng câu đúng (0.941 vs 0.947) |
| **RC3** | Bộ phán xử mất phán quyết ở 2 / 3 biến thể (văn xuôi tới trần token) | câu bất đồng lùi về bỏ phiếu | `f5a`: 30 / 94 câu bất đồng, 11 sai / bỏ | 352 / 470 đầu ra `list` không có `<choice>` |
| **RC4** | Lệch phân phối cách nói giữa train và public: kỳ viết gọn chiếm 26 % câu public, 0 % câu train; viết tắt, trợ từ | S và bộ phán xử cùng mù vì chung trọng số, chung dữ liệu gốc | ~6–10 câu / log | thống kê kỳ viết gọn trên toàn bộ câu. **Trong v2 đây chỉ là chẩn đoán**: giải pháp E không được lấy trục từ các câu sai này |
| RC5 | Nhầm đối tượng được đếm; hàm không có trong tài liệu API | đếm sản phẩm thay vì lô, lượt thay vì máy; `purchased_qty_base` vắng trong `API_DOC` | 3–4 câu | F sửa phần API (tổng quát); phần "đối tượng đếm" để E sinh có hệ thống, không vá theo câu |

## 3. Proposed solutions

**Nguyên tắc.** Không luật viết tay lúc suy luận: mọi thành phần đọc câu hỏi hay rulebook là mô hình; phần cố định chỉ là bố cục vào / ra, kiểm
tra thực thi và hạ tầng. Mỗi LoRA huấn luyện **độc lập** cho một vai, rồi ghép theo pha lúc suy luận (`w_model`, `w_mode: merge`, ~1.5 phút / adapter).
Dữ liệu mới do mô hình sinh, nhãn kiểm bằng thực thi, chỉ từ DN train và bỏ 6 DN held-out. Họ thuật ngữ dùng để kiểm tách khỏi họ dùng huấn luyện.

### A — Sửa kênh đầu ra của bộ phán xử (RC3; không huấn luyện)

Biến thể không suy nghĩ sinh có ràng buộc (structured output của vLLM: chỉ `<choice>k</choice>` hoặc JSON `{"choice": k}`), hoặc lấy phân phối xác
suất của token lựa chọn. Kiểm chứng bằng phán xử lại log đã lưu: 4 log public, log H0 `pub_full_H`, log held-out H0. Không nhiễu sinh. Bằng chứng
như bản 1 (LEVER [1], PICARD [2]; JSON có ràng buộc đã đưa W1 từ 13 lên 56 dòng).

### B — Định tuyến "xung đột rulebook" tới bộ phán xử (RC2, RC1a; S đóng băng)

Như bản 1. W2′ in dòng áp dụng + trạng thái (có định nghĩa / không cần / thiếu định nghĩa). Câu đồng thuận được gửi phán xử khi (i) "thiếu định
nghĩa" hoặc (ii) có dòng nối mà không ứng viên nào dùng lời gọi đã kiểm của dòng đó; ở (ii) S chạy lại một lượt với dòng đó.
- Hai điều kiện này là luật **điều phối**: đầu vào là đầu ra của mô hình, không đọc câu hỏi.
- Dữ liệu W2′ từ rulebook đối chứng (i). Cùng một câu có nhãn khác nhau tuỳ rulebook, nên mô hình buộc phải đọc rulebook.
- Đo precision trên DN train held-out trước khi chạy public.

### C — S′ đọc rulebook: huấn luyện bằng rulebook đối chứng (RC1)

Lớp ngữ nghĩa của split (mọi dòng: thuật ngữ | định nghĩa | lời gọi đã kiểm của W1 | **hàm WT đã kiểm**) đặt ở phần đầu prompt dùng chung, nhờ
prefix cache gần như không tốn thêm. LoRA chồng trên V3A × 0.70 học các bộ ba (rulebook′, câu, chương trình′) mà chỉ rulebook thay đổi:
- (a) đổi định nghĩa giữa các phương án toolkit biểu diễn được, nhãn tính lại bằng thực thi;
- **(a′) mới: đổi định nghĩa sang cách tính toolkit không có hàm.** Ví dụ ngưỡng, cửa sổ thời gian, đối tượng đếm, điều kiện kết hợp. Nhãn là
  chương trình pandas trên `C` / `T`, hoặc lời gọi hàm WT tương ứng. Nhãn do thầy viết (Qwen gốc có suy nghĩ, ≥ 3 / 6 mẫu cùng kết quả trên 13 kỳ)
  và phải **khác** kết quả của định nghĩa gốc ở ít nhất một kỳ. Chiếm ~25–30 % phần đối chứng. Mục đích: S′ học rằng khi định nghĩa trong rulebook
  khác mặc định của toolkit thì đi theo định nghĩa, và tự viết pandas khi không có hàm. Đây là chỗ S hiện tại gãy khi gặp tình huống mới;
- (b) đổi tên thuật ngữ;
- (c) xoá dòng → `Unanswerable`;
- (d) dòng nhiễu → chương trình không đổi.

Câu `unanswerable` 3–5 %. Thước đo không cần đáp án public: độ nhạy rulebook (tỉ lệ cặp đối chứng mà chương trình đổi đúng) và F1 tiêu chí chưa
định nghĩa, đo trên DN held-out. Chi phí: ~30M token ≈ 4.5–5.5 giờ FSDP 4×L4 / seed, 2 seed song song. Rủi ro hồi quy như v4 → chỉ nhận khi
qua cổng §5. Bằng chứng: symbol tuning [6], KAFT [7], CAD [8], BIRD [9].

### D — Kênh thứ hai (RC2, RC5; chỉ khi còn giờ)

Như bản 1: một mẫu "đọc trước, viết sau" bằng trọng số gốc, bật suy nghĩ; bất đồng với S thì chuyển phán xử. Không bắt buộc; cắt đầu tiên nếu trễ.

### E — Dữ liệu tương phản sinh có hệ thống (RC4, RC5) — sửa theo đề xuất 1

- **Không còn danh sách trục chọn tay.** Trục tương phản được rút **tự động** từ cấu trúc các chương trình đã kiểm trên câu train: mỗi "khe" là một
  thành phần của chương trình (kỳ và cách viết kỳ, hàm đo, đối số `by=`, thực thể được đếm, bộ lọc, hướng xếp hạng, đếm hay liệt kê, đơn vị).
- Với mỗi câu train có chương trình đã kiểm, Qwen gốc sinh một **biến đổi tối thiểu**: đổi đúng một khe → câu′ + chương trình′. Nhãn nhận khi:
  - chương trình′ chạy được và đáp án khác câu gốc;
  - S giải câu′ (4 mẫu) không mâu thuẫn với chương trình′; nếu mâu thuẫn, Qwen gốc có suy nghĩ phân xử, không có đa số thì loại.
- Cách viết (kỳ, viết tắt, không dấu, phương ngữ) cũng do mô hình **tự liệt kê** từ một chỉ dẫn chung và từ câu train. Không có mẫu nào lấy từ câu public.
- Phân bố trục theo tần suất khe trong chương trình train, không theo lỗi public.
- Dùng cho bộ phán xử WJR2 và dữ liệu C.
- Kỳ vọng: giảm nhầm loại "đọc gần đúng" một cách tổng quát. Quy ước nào không suy ra được từ dữ liệu / rulebook (ví dụ "M/YY") có thể vẫn sai, và
  được chấp nhận là sai.
- Bằng chứng: Dr.Spider [13], ViLexNorm [14], CAD [8].

### F — Tài liệu API tự sinh từ toolkit (RC5; không huấn luyện)

Như bản 1: `# API` sinh từ `api_spec` (mọi hàm công khai, chữ ký, một dòng docstring). Ở chế độ H, khối này **không** chứa hàm `industry.py`.

### G — LoRA viết tool WT (RC0) — mới

**Câu hỏi của user:** dùng Qwen gốc làm bộ viết tool, hay huấn luyện thêm một LoRA viết tool, độc lập rồi ghép?

| Tiêu chí | Qwen gốc (không adapter, có suy nghĩ) | LoRA WT huấn luyện riêng |
|---|---|---|
| Bằng chứng trong repo | G0g (Qwen gốc) ≈ G0f (trọng số S): 37 vs 41 hàm, `pub_noindH` 0.009 vs 0.018. Có suy nghĩ ở trần 4096: 317 / 389 mẫu không kịp đóng `</think>`. 8 mẫu + 2 vòng sửa không hơn | Vai được huấn luyện riêng đều vượt zero-shot rõ: WL1 0.78 vs 0.24; WJB1 +0.010–0.012; JSON có ràng buộc 13 → 56 dòng |
| Loại lỗi còn lại của W (`docs/17` 13/11 22:20) | dạng giá trị (dict, tỉ lệ thay %), API (`p[...]`), tên bảng sai: lỗi **quy ước**, zero-shot sửa không hết | đúng loại lỗi mà SFT sửa rẻ nhất |
| Thời gian suy luận (container ≤ 6 h) | suy nghĩ 8–12k token × 6 mẫu × số dòng → chậm, dễ chạm trần `units_max_s` | không suy nghĩ, đầu ra ngắn (~300 token) |
| Rủi ro "luật Claude" | không có | có, nếu nhãn lấy từ hàm viết tay → nhãn chỉ từ: chương trình đã kiểm bằng **đáp án train của BTC**, hoặc hàm của thầy đạt tự nhất quán và kiểm 13 kỳ; **không** dùng `industry.py` làm nhãn |
| Chi phí | 0 | sinh dữ liệu ~3 phiên × 2.5 h + huấn luyện ~2.5 h / nhánh |

**Đề xuất: huấn luyện WT, dùng Qwen gốc có suy nghĩ làm thầy sinh dữ liệu và làm nhánh đối chứng.** Đây là chưng cất kèm lọc bằng thực thi
(STaR [18], LATM [19], CREATOR / CRAFT [20, 21], Self-Debugging [22], CodeT [23], distilling step-by-step [24]).

**Huấn luyện trên Qwen gốc (WT-b) là nhánh chính, WT-s (trên V3A × 0.70) là nhánh kiểm tra.** Lý do:
- viết tool cần pandas tổng quát, trong khi V3A học hẹp theo 55 khuôn;
- WT-b độc lập với S, nên khi C thay S bằng S′ thì WT vẫn dùng nguyên được;
- gộp WT vào Qwen gốc không thêm tham số.

Hai nhánh chạy song song (tài nguyên đủ); cổng G-WT chọn.

**Dữ liệu WT** (prompt đúng như `lrl_gen.unit_items` lúc suy luận; đầu ra JSON `plan, name, signature, doc, triggers, code[]`):

| Nguồn | Nhãn |
|---|---|
| (1) thuật ngữ mới từ câu train có chương trình đã kiểm (`lrl_gen --mode newterm`, mở rộng từ 54 lên ~2k): thuật ngữ + định nghĩa do mô hình viết | chương trình đã kiểm, tổng quát theo kỳ `p` (đúng theo đáp án train) |
| (2) dòng rulebook train × DN train | lời gọi / hàm thầy viết, kiểm 13 kỳ |
| (3) dòng đối chứng (định nghĩa đổi, cả ngoài toolkit) | hàm thầy, tự nhất quán ≥ 3 / 6, khác hàm gốc ≥ 1 kỳ |
| (4) **mục ngành tổng hợp**: Qwen gốc tự đặt 10–25 dòng rulebook "ngành" cho bảng sẵn có của từng DN train (giao hàng, lô, nợ, trả góp, nhân viên…) | hàm thầy, tự nhất quán + kiểm 13 kỳ |

Cả 4 nguồn bỏ 6 DN held-out, dành chúng để đo. Ước ~5–7k bản ghi, ~15M token.

**Ghép lúc suy luận:** pha `prepare_units` gộp WT vào bản sao Qwen gốc (WT-b) hoặc S (WT-s), viết hàm cho mọi dòng ngành / mới, rồi
`units.verify` (13 kỳ) và tự nhất quán như hiện tại. Hàm đã kiểm vào `# SHOP FUNCTIONS` của S / S′. Tổng tham số: 9.65B + 4 adapter
(S, WT, W2′, WJR2) ≈ 10B < 15B.

## 4. Prioritization

Số tăng là dự báo, tính trên **`pub_full_H`** (trung thực). Bản 1 từng dự báo kế hoạch 5 đạt 0.96–0.975 và thực tế 0.952.

| Hạng | Giải pháp | Gốc | Tăng dự kiến `pub_full_H` | Chi phí | Tổng quát sang private |
|---|---|---|---|---|---|
| 1 | **G** LoRA viết tool | RC0 | +0.03 … +0.06 | sinh dữ liệu ~7 h phiên + huấn luyện 2 × 2.5 h | rất cao: ngành mới chỉ có thể đúng nhờ hàm do mô hình viết |
| 2 | **A** đầu ra phán xử | RC3 | +0.002 … +0.006 | vài giờ, không huấn luyện | cao |
| 3 | **B** định tuyến | RC2, RC1a | +0.004 … +0.010 | huấn luyện ~2 h + phán xử lại | cao |
| 4 | **C** S′ | RC1 | +0.005 … +0.015 nếu đạt; có thể âm | 2 × 5 h | rất cao |
| 5 | **E** tương phản hệ thống (qua WJR2 / C) | RC4, RC5 | +0.002 … +0.006 | sinh ~3 h phiên, gộp vào huấn luyện | trung bình |
| 6 | **F** API tự sinh | RC5 | +0.001 … +0.003 | vài giờ | cao |
| 7 | **D** kênh thứ hai | RC2 | +0.002 … +0.005 | +1 mẫu dài / câu | cao; chỉ khi còn giờ |

Thứ tự triển khai:
- **H0 ngay** (không cần code mới);
- **A, F** trong 4 giờ đầu;
- **sinh dữ liệu thầy** (G, C, E dùng chung) từ T+2;
- **G** huấn luyện sớm nhất (T+8), vì là đòn bẩy lớn nhất cho hệ không luật;
- **B, WJR2, C** song song sau đó;
- **D** chỉ khi còn giờ.

## 5. Đo và quy tắc nhận

| Tập | Câu | Vai |
|---|---|---|
| `pub_full_H` | 1000 public gốc, ẩn `industry.py` | **số chính** (chọn cấu hình) |
| `pub_full` | 1000, còn `industry.py` | tham chiếu, so với kế hoạch cũ |
| `pub_ho400c_H` | 348 paraphrase public (đã làm sạch) | held-out |
| `tr_ho400c` | 375 câu của 6 DN train held-out | held-out (không DN nào trong dữ liệu mới) |
| `pub_hoind_H` | 345 paraphrase của 5 DN ngành | held-out ngành mới |
| `tr_newterm` | 54 câu thuật ngữ mới | held-out thuật ngữ |
| `pub_ind_H` | 269 câu gốc DN ngành | đo G (cùng `pub_hoind_H` = 614 câu) |
| độ chính xác hàm WT | dòng của 6 DN held-out (nguồn 1–4) | đo G offline |

**Quy tắc nhận một thay đổi.**
1. Thay đổi đo trên log cố định (A, B, WJR2): TB các log tăng, không log public nào giảm, mỗi tập held-out (phán xử lại trên log held-out H0)
   không giảm quá 0.005.
2. Thay đổi đổi việc sinh (F, G, C): tăng ghép cặp có z ≥ 2 trên tập đích, và **mỗi** tập held-out không giảm quá 1 SE ghép cặp. C cần 2 seed cùng dấu.
3. Tăng `pub_full` / `pub_full_H` mà giảm held-out → **loại**.
4. Báo cả số H, số thường và held-out ở mọi mốc.
5. Cấu hình nộp: `pub_full_H` cao nhất trong các cấu hình qua cổng held-out.

## 6. Đánh giá khách quan về mục tiêu 1.0

- **Đã chứng minh:**
  - phán xử huấn luyện +0.010–0.015;
  - W có tác dụng ở ngành mới (+0.119 trên 269 câu, z 3.8), nhưng chỉ lấp được 0.12 / 0.42 khoảng cách tới hàm viết tay;
  - vai được huấn luyện riêng vượt zero-shot ở mọi lần thử (WL1, WJB1, WJR1).
- **Giả thuyết:**
  - WT lấp được phần lớn khoảng cách mà không cần hàm viết tay;
  - S′ học dùng rulebook mà không hồi quy như v4;
  - dữ liệu tương phản hệ thống chuyển được sang cách nói thật.
- **Dự báo (H):** mốc ~0.87 → +G 0.90–0.93 → +A, B, E, F 0.92–0.95 → +C tốt nhất ~0.96. Số thường (còn `industry.py`) dự kiến 0.955–0.97.
- **1.0** đòi hỏi sửa hết các câu sai ở mọi log (cả quy ước viết không suy ra được) mà không làm hỏng câu ổn định. Xác suất thấp.
- **Rủi ro:**
  - 34 giờ cho ~6 lần huấn luyện + lưới đánh giá;
  - chọn trên chính public lạc quan 0.003–0.005, nên mới cần held-out;
  - hàm WT sai mà vẫn qua kiểm 13 kỳ → S gọi nhầm. Giảm bằng tự nhất quán ≥ 3 / 6 và đo độ chính xác hàm trên DN held-out.

## 7. Nguồn tham khảo

1. Ni et al., *LEVER: Learning to Verify Language-to-Code Generation with Execution*, ICML 2023 — arXiv 2302.08468
2. Scholak et al., *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding*, EMNLP 2021 — arXiv 2109.05093
3. Zhang et al., *R-Tuning: Instructing LLMs to Say "I Don't Know"*, NAACL 2024 — arXiv 2311.09677
4. Lee et al., *TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring* — arXiv 2403.15879
5. Han et al., *R³-SQL: Ranking Reward and Resampling for Text-to-SQL*, ACL Findings 2026 — arXiv 2604.25325
6. Wei et al., *Symbol Tuning Improves In-Context Learning in Language Models*, EMNLP 2023 — arXiv 2305.08298
7. Li et al., *Large Language Models with Controllable Working Memory* (KAFT), ACL Findings 2023 — arXiv 2211.05110
8. Kaushik et al., *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*, ICLR 2020 — arXiv 1909.12434
9. Li et al., *BIRD: Can LLM Already Serve as a Database Interface?*, NeurIPS 2023 — https://bird-bench.github.io
10. Krogh & Vedelsby, *Neural Network Ensembles, Cross Validation, and Active Learning*, NeurIPS 1994
11. Pourreza et al., *CHASE-SQL*, ICLR 2025 — arXiv 2410.01943
12. Gao et al., *XiYan-SQL* — arXiv 2411.08599
13. Chang et al., *Dr.Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL Robustness*, ICLR 2023 — arXiv 2301.08881
14. Nguyen et al., *ViLexNorm*, EACL 2024
15. Fan et al., *CycleSQL*, ICDE 2025 — arXiv 2411.02948
16. Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 — arXiv 2310.01798
17. vLLM, *LoRA Adapters* và *Structured Outputs* — https://docs.vllm.ai/en/stable/features/lora/
18. Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*, NeurIPS 2022 — arXiv 2203.14465
19. Cai et al., *Large Language Models as Tool Makers* (LATM), ICLR 2024 — arXiv 2305.17126 (tách vai "làm tool" / "dùng tool")
20. Qian et al., *CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of LLMs*, EMNLP Findings 2023 — arXiv 2305.14318
21. Yuan et al., *CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets*, ICLR 2024 — arXiv 2309.17428
22. Chen et al., *Teaching Large Language Models to Self-Debug*, ICLR 2024 — arXiv 2304.05128
23. Chen et al., *CodeT: Code Generation with Generated Tests*, ICLR 2023 — arXiv 2207.10397
24. Hsieh et al., *Distilling Step-by-Step!*, ACL Findings 2023 — arXiv 2305.02301
