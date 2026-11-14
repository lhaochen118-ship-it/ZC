# Proposed plan — đưa hệ kế hoạch 5 (public TB 0.952) hướng tới 1.0

*DataMate, Zalo AI Challenge 2026 · lập 14/11/2026 16:10 UTC · nguồn: `handoff/04_PROPOSED_PLAN.md` · đóng băng container 16/11 04:00 UTC*

**Tóm tắt.** Sau kế hoạch 5, điểm mất còn lại chia hai phần gần bằng nhau: (1) câu mà **mọi mẫu của S cùng sai** nên không bao giờ được phán xử (8–19 câu mỗi lần chạy), gốc là S chưa từng được dạy dùng rulebook; (2) câu có mẫu đúng nhưng **bộ phán xử chọn sai hoặc không ra phán quyết** (6–15 câu). Đề xuất ba việc theo thứ tự: sửa kênh đầu ra của bộ phán xử (không huấn luyện); định tuyến câu "xung đột rulebook" tới bộ phán xử bằng một bộ nối luật biết báo *thiếu định nghĩa*; huấn luyện S đọc rulebook bằng **rulebook đối chứng**. Ước lượng 0.952 → 0.965–0.975; **1.0 không được bảo đảm** (mục 6).

## 0. Bản tóm tắt để chạy (cho khung chat thực thi)

**Mốc.** Cấu hình `configs/flow_p5_final.json`, public TB **0.952** trên 4 log S (`f5a`, `f5b`, `pva3k16`, `g1v3apf2`). Đáp án chuẩn: `score 1.jsonl` ở gốc repo (trùng `data/raw/public_gt.jsonl` dựng bằng `automation/rebuild_public_gt.py`). Đóng băng container **16/11 04:00 UTC**.

**Còn sai 24–28 câu mỗi log**, ba nhóm: 8–19 câu mọi mẫu S cùng sai (không được phán xử); 6–15 câu có mẫu đúng nhưng mất; 2–5 câu không mẫu nào đúng.

| Thứ tự | Việc | Làm gì | Đo ở đâu | Nhận khi |
|---|---|---|---|---|
| 0 | **Docker dự phòng** | D1 cho `flow_p5_final` (`handoff/03_HANDOFF.md` §5) | đường container | có điểm container của đúng cấu hình WJR1 |
| 1 | **A** sửa đầu ra phán xử | `src/judge.py`: biến thể `list` / `point` sinh có ràng buộc, chỉ in `<choice>k</choice>` (hoặc lấy xác suất token lựa chọn). Hiện WJR1 chỉ đọc được 19–25 % / 35–40 % phán quyết | phán xử lại 4 log đã lưu (`soup_job --runner judge`, `tests/replay_judge.py`) | tỉ lệ đọc được ≥ 95 %; TB > 0.952, không log nào giảm |
| 1 | **F** khối API tự sinh | `# API` của prompt sinh từ `api_spec` (mọi hàm công khai) thay `dm.API_DOC` viết tay; hiện `purchased_qty_base` không có trong prompt | một lần chạy public, so ghép cặp với `f5a` | không âm; âm thì chỉ thêm dòng thiếu vào cuối khối |
| 2 | **Dữ liệu** (CPU, chỉ cửa hàng train) | (i) rulebook đối chứng: xoá / đổi định nghĩa / đổi tên thuật ngữ / thêm dòng nhiễu, nhãn tính lại bằng thực thi; (ii) cặp tương phản: tháng-năm ↔ ngày-tháng, đơn ↔ ngày ↔ số lượng, lô ↔ sản phẩm, máy ↔ lượt, nhập ↔ bán, đếm ↔ liệt kê | — | mỗi cặp chạy được, hai nhãn khác nhau đúng dự định |
| 3 | **B** định tuyến xung đột rulebook (S đóng băng) | Huấn luyện **W2′** (từ `w_link.py` + dữ liệu i): mỗi câu in dòng áp dụng + trạng thái *có định nghĩa / không cần / tiêu chí chủ quan thiếu định nghĩa*. Câu đồng thuận gửi tới bộ phán xử khi "thiếu định nghĩa", hoặc khi có dòng nối mà không ứng viên nào dùng lời gọi đã kiểm (S chạy lại một lượt với dòng đó). Huấn luyện **WJR2** = WJR1 + dữ liệu (ii) + câu đồng thuận bị định tuyến | cửa hàng train giữ ngoài, rồi 4 log public | precision "thiếu định nghĩa" ≥ 0.9; số câu sai đồng thuận / log giảm; câu bị làm hỏng ≤ 1/3 câu sửa được |
| 3 | **C** S′ đọc rulebook (song song với B) | LoRA chồng trên V3A × 0.70; prompt có lớp ngữ nghĩa đầy đủ của split (thuật ngữ, định nghĩa, lời gọi W1) ở phần đầu dùng chung; dữ liệu = v3a dựng lại theo bố cục mới + (i) + (ii), câu không trả lời được 3–5 %; ~40–50M token, 2 seed | public × 2 seed, cùng bộ phán xử | độ nhạy rulebook ≥ 0.95 trên train giữ ngoài; tăng ghép cặp z ≥ 2, hai seed cùng dấu; không đạt → giữ S |
| 4 | **D** kênh thứ hai (chỉ khi A xong và còn giờ) | một mẫu mỗi câu bằng trọng số gốc (`w_mode: lora`, tắt adapter theo request), bật suy nghĩ; bất đồng với S → bộ phán xử | 4 log public | tăng ghép cặp, không log nào giảm |
| 5 | **Đóng gói** | chọn cấu hình, đường container × 2, thời gian 1,000 câu, Docker, `README_SOLUTION` | — | hai lần chạy lệch ≤ 0.03; trong 6 giờ |

**Quy tắc khi chạy.** Không luật viết tay lúc suy luận; không dùng câu public để huấn luyện; so ghép cặp trên cùng câu, ≥ 2 seed cho thay đổi ở S; thành phần nào không qua cổng thì tắt bằng cấu hình; tuân thủ PDF 2 §9 và cổng kiểm tra trong `CLAUDE.md`. Dự báo: A + F → 0.954–0.960; thêm B hoặc C → 0.962–0.972; tốt nhất ~0.98; 1.0 xác suất thấp.

## 1. Hiện trạng và cách phân tích

- **Hệ hiện tại** (`configs/flow_p5_final.json`): S = Qwen3.5-9B + LoRA v3a × 0.70 viết chương trình (4–16 mẫu / câu, đóng băng); W1 gắn dòng rulebook với lời gọi toolkit đã chạy thử; W2 (linker WL1) nối mỗi câu với dòng rulebook; W3 (bộ phán xử WJR1) chọn giữa các nhóm đáp án ở câu có mẫu bất đồng.
- Đáp án: file `score 1.jsonl` của user (chính xác 100 %); bản dựng lại bằng `rebuild_public_gt.py` trùng nó ở cả 1,000 câu. Tôi phát lại chính sách cuối trên 4 log S đã lưu: **0.949 / 0.953 / 0.958 / 0.949, TB 0.952** — khớp số đã báo cáo. Đường container thật (bộ phán xử cũ WJB1): 0.943 / 0.941.
- Chỉ phân tích câu public gốc; không đề xuất nào được xây từ từng câu sai.

| Phép đo (4 log: `f5a`, `f5b`, `pva3k16`, `g1v3apf2`) | Giá trị |
|---|---|
| Câu chưa đúng mỗi log | 28 / 26 / 24 / 28 |
| — mọi mẫu S đồng thuận sai, không được phán xử | 8 / 9 / 12 / 19 |
| — bất đồng nhưng không mẫu nào đúng | 5 / 4 / 2 / 3 |
| — **có mẫu đúng nhưng mất** (phán xử sai, hoặc lùi về bỏ phiếu) | 15 / 13 / 10 / 6 |
| Trần "có ít nhất một mẫu đúng" | 987 / 987 / 986 / 978 |
| Ổn định: đúng ở cả 4 log / sai ở cả 4 log / sai ở ≥ 1 log | 957 / 13 / 43 câu |
| Tỉ lệ đầu ra của WJR1 đọc được phán quyết: `list` / `point` / `point+think` | **19–25 % / 35–40 % / 89–90 %** (WJB1: 88 % / 92 % / 98 %) |

## 2. Root causes

| # | Nguyên nhân gốc | Biểu hiện bề mặt | Quy mô | Bằng chứng |
|---|---|---|---|---|
| **RC1** | **S chưa từng được dạy dùng rulebook.** Khi huấn luyện, rulebook là hằng số và không có trong prompt; nhãn sinh từ 55 khuôn chương trình | (a) Tiêu chí **không có định nghĩa** vẫn được trả lời bằng hàm xếp hạng gần nhất; (b) thuật ngữ **có** định nghĩa chỉ đúng khi linker nối trúng | 13 câu sai ở cả 4 log, trong đó 6 câu (a); thêm 5–7 câu (b) tuỳ log | "Thợ nhiệt tình / NCC uy tín / khách thân thiện nhất": linker trả về rỗng (đúng), S vẫn trả lời 4 / 4. "Trung thành nhất": có dòng nối → đúng (`f5a`, `f5b`), không có → sai (`pva3k16`). "Ưa chuộng nhất": linker sót → sai cả 4 log. Tập v3a chỉ có 36 / 4,919 nhãn không trả lời được (0.7 %; public 3.1 %). Đưa dòng rulebook vào prompt S đóng băng: 0.921–0.940, không hơn mốc |
| **RC2** | **Sai đồng thuận là vô hình.** Bộ phán xử chỉ xét câu có mẫu bất đồng; các mẫu cùng prompt nên lỗi tương quan | 8–19 câu mỗi log không bao giờ được xem lại | ~45 % số câu chưa đúng | Phán mọi câu thì hỏng câu đúng (zero-shot 0.941 so với 0.947) → cần tín hiệu chọn *câu nào* đáng xét lại |
| **RC3** | **Bộ phán xử mất phần lớn phán quyết ở hai trong ba biến thể.** WJR1 không suy nghĩ vẫn lập luận văn xuôi tới trần token, không in `<choice>` | Câu bất đồng lùi về bỏ phiếu | `f5a`: 30 / 94 câu bất đồng lùi về bỏ phiếu, 11 trong đó sai hoặc bỏ | 352 / 470 đầu ra `list` không có phán quyết; cùng lỗi đã gặp ở W1 (PDF 2 §9 mục 3) |
| **RC4** | **Lệch quy ước cách nói giữa train và public**, S và bộ phán xử cùng mù vì chung trọng số và cùng dữ liệu gốc | "4/26", "5/25" đọc thành ngày thay vì tháng / năm; "đh" → đĩa; "từng bên doanh thu bao nhiêu" → so sánh; trợ từ "z" → cỡ Z | ~6–10 câu mỗi log, phần lớn có mẫu đúng | Kỳ viết gọn: 26 % câu public, 0 % câu train; bộ phán xử chọn cách đọc "ngày" 3 / 3 phiếu ở "4/26 doanh thu" |
| RC5 | **Nhầm đối tượng được đếm** và **hàm không có trong tài liệu API của prompt** | Hỏi số **lô** sắp hết hạn (đáp án 14) → S liệt kê **sản phẩm**; hỏi số **máy** đã bảo hành (5) → đa số mẫu đếm **lượt** (4); hỏi số viên **nhập** (1,500) → S đếm số **bán** (1,409) | 3–4 câu, 2 câu không log nào có mẫu đúng | Câu "nhập bao nhiêu viên": hàm `purchased_qty_base` có trong toolkit nhưng **không có trong `API_DOC`** viết tay → 1 / 32 mẫu gọi tới. Câu đếm lô: `lots_expiring` có sẵn, 0 / 32 mẫu lấy `len(...)`. Đây không phải câu mơ hồ: cả ba đều có đáp án xác định |

Tách bạch: "trả lời bừa câu *thân thiện nhất*" và "sai câu *ưa chuộng nhất*" là biểu hiện; gốc chung là S không coi rulebook là đầu vào quyết định (RC1), và hệ không có cơ chế nghi ngờ một đáp án đồng thuận (RC2). Kế hoạch 5 đã chứng minh phần "chọn ứng viên" (+0.010–0.015) nhưng giữ S đóng băng nên chưa chạm tới gốc này.

## 3. Proposed solutions

**Nguyên tắc.** Không luật viết tay lúc suy luận: mọi thành phần đọc câu hỏi là mô hình; phần cố định chỉ là bố cục vào / ra và hạ tầng. Dữ liệu huấn luyện mới do mô hình sinh, nhãn kiểm bằng thực thi, chỉ từ cửa hàng train; họ thuật ngữ dùng để kiểm tra tách khỏi họ dùng để huấn luyện.

### A — Sửa kênh đầu ra của bộ phán xử (trúng RC3; không huấn luyện)

- **Cơ chế.** Biến thể không suy nghĩ sinh có ràng buộc: chỉ được in `<choice>k</choice>` (structured output của vLLM), hoặc lấy thẳng phân phối xác suất của token lựa chọn trong một lượt chạy. Xác suất này thay cho "tỉ lệ 5 mẫu" trong tổ hợp, mịn hơn và rẻ hơn.
- **Vì sao.** Hai trong ba thành viên tổ hợp đang mất 60–80 % phán quyết; tổ hợp thực chất chỉ còn `point+think`.
- **Bằng chứng.** LEVER dùng chính xác suất của bộ kiểm để xếp hạng (+4.6–10.9 điểm) [1]; giải mã có ràng buộc loại lỗi định dạng [2]. Trong repo: JSON có ràng buộc đã đưa W1 từ 13 lên 56 dòng dùng được.
- **Kiểm chứng.** Phán xử lại 4 log đã lưu (10–15 phút / biến thể, không có nhiễu sinh). **Trạng thái:** lỗi đã đo; mức tăng là giả thuyết.

### B — Định tuyến "xung đột rulebook" tới bộ phán xử (trúng RC2 và RC1a; S vẫn đóng băng)

- **Cơ chế.** Nâng linker W2 thành bộ *kiểm định tiêu chí*: với mỗi câu, ngoài các dòng áp dụng, nó in một trong ba trạng thái — có dòng định nghĩa / không cần định nghĩa (đại lượng đo được trực tiếp) / **tiêu chí chủ quan không có định nghĩa**. Câu đồng thuận được gửi tới bộ phán xử **chỉ khi** (i) trạng thái là "không có định nghĩa", hoặc (ii) có dòng nối mà không ứng viên nào dùng lời gọi đã kiểm của dòng đó; ở (ii) S được chạy lại một lượt với dòng đó để sinh ứng viên mới. Bộ phán xử (đã có lựa chọn "không trả lời được") quyết định.
- **Dữ liệu huấn luyện.** Sinh *đối chứng* trên rulebook train: xoá dòng định nghĩa của một tiêu chí → nhãn "không có định nghĩa"; đổi tên thuật ngữ; thêm dòng nhiễu. Cùng một câu có nhãn khác nhau tuỳ rulebook nên bộ kiểm định buộc phải đọc rulebook thay vì thuộc danh sách.
- **Vì sao.** Tạo bất đồng *có thông tin* đúng ở nơi có xung đột với rulebook, thay vì phán mọi câu. Với thước đo phạt −1, mỗi câu sai đồng thuận được sửa là +0.002.
- **Bằng chứng.** R-Tuning: dạy từ chối bằng dữ liệu có nhãn "không biết" tổng quát sang miền khác [3]; TrustSQL: dưới thước đo có phạt, biết từ chối quan trọng ngang trả lời đúng [4]; R³-SQL: lấy mẫu lại khi nhóm ứng viên thiếu đáp án đúng [5]. Trong repo: WL1 huấn luyện đạt 0.78 so với 0.24 zero-shot; huấn luyện WL1 chỉ 7.3M token (~1 giờ).
- **Rủi ro.** Báo nhầm làm hỏng câu đồng thuận đúng → đo precision trên cửa hàng train trước, ngưỡng do bộ hiệu chỉnh học. **Trạng thái:** giả thuyết, chưa đo.

### C — S′ đọc rulebook: huấn luyện bằng rulebook đối chứng (trúng gốc RC1)

- **Cơ chế.** Đặt "lớp ngữ nghĩa" của split (mọi dòng: thuật ngữ | định nghĩa | lời gọi đã kiểm của W1) vào phần đầu prompt dùng chung — prefix cache làm chi phí thêm gần 0 — rồi huấn luyện một LoRA chồng trên S với các bộ ba (rulebook′, câu hỏi, chương trình′) mà **chỉ rulebook thay đổi**: (a) đổi định nghĩa giữa các phương án toolkit biểu diễn được, nhãn tính lại bằng thực thi; (b) đổi tên thuật ngữ; (c) xoá dòng → `Unanswerable`; (d) dòng nhiễu → chương trình không đổi.
- **Vì sao.** Khi nhãn phụ thuộc rulebook, cách duy nhất giảm loss là đọc nó. S′ không còn phụ thuộc độ phủ của linker ("ưa chuộng nhất") và tự nhận ra tiêu chí thiếu định nghĩa vì thấy toàn bộ danh sách. Ca (d) dạy đúng điều S đóng băng thiếu: dòng không liên quan thì bỏ qua (nguyên nhân 0.921 của Z-link).
- **Bằng chứng.** Symbol tuning: nhãn tuỳ ý buộc mô hình dùng ngữ cảnh, kể cả ghi đè tri thức sẵn có [6]; KAFT: ngữ cảnh phản thực + ngữ cảnh nhiễu khi tinh chỉnh tăng mức nghe theo ngữ cảnh 24 lần ở PaLM [7]; dữ liệu đối chứng tối thiểu chống lối tắt [8]; BIRD: có / không có tri thức nghiệp vụ trong ngữ cảnh chênh ~20 điểm với GPT-4 (theo tóm tắt bài gốc) [9].
- **Thước đo không cần đáp án public.** *Độ nhạy rulebook* (tỉ lệ cặp đối chứng mà chương trình đổi đúng) và F1 tiêu chí chưa định nghĩa trên họ từ không dùng khi huấn luyện, đo ở cửa hàng train giữ ngoài.
- **Chi phí / rủi ro.** ~40–50M token ≈ 6–8 giờ FSDP 4×L4, 2 seed. Học tiếp trên v3a từng làm giảm public (v4: 0.916–0.921 so với 0.936) → chỉ nhận khi so ghép cặp vượt mốc. **Trạng thái:** giả thuyết; khác v3b / v4 ở chỗ rulebook đầy đủ, đối chứng trên thuật ngữ đã biết, có ca xoá và ca nhiễu.

### D — Kênh suy luận thứ hai để lộ sai đồng thuận ngoài rulebook (trúng RC2, RC5)

- **Cơ chế.** Một mẫu "đọc trước, viết sau" mỗi câu bằng cùng trọng số gốc nhưng tắt LoRA theo request và bật suy nghĩ; bất đồng với S thì chuyển cho bộ phán xử. Tổng tham số không đổi (9.65B + adapter).
- **Bằng chứng.** Lợi ích của tổ hợp tỉ lệ với mức khác nhau giữa các thành viên [10]; CHASE-SQL và XiYan-SQL dựa vào nhiều bộ sinh khác kiểu [11, 12]. Trong repo: LoRA động đã chạy đúng sau `merge.py vllm` (0.951 so với 0.950); gộp mẫu hai lần chạy nâng trần 986 → 991 nhưng bộ phán xử hiện tại chưa tận dụng được.
- **Rủi ro.** Mô hình gốc viết chương trình yếu hơn S → nhiều bất đồng sai; chỉ có lợi khi bộ phán xử đã được sửa (A) và học thêm (E). **Trạng thái:** giả thuyết.

### E — Dữ liệu tương phản về cách nói và đối tượng đếm, dùng chung cho bộ phán xử và S′ (trúng RC4, RC5)

- **Cơ chế.** Mô hình sinh *cặp tương phản tối thiểu* quanh các cách đọc gần nhau (tháng / năm ↔ ngày / tháng; số đơn ↔ số ngày ↔ số lượng; liệt kê từng bên ↔ so sánh; **đối tượng đếm**: lô ↔ sản phẩm, máy ↔ lượt, nhập ↔ bán; đếm ↔ liệt kê), nhãn kiểm bằng thực thi, viết theo nhiều kiểu chữ (không dấu, viết tắt, phương ngữ). Bổ sung vào dữ liệu phán xử (WJR2) và dữ liệu của C.
- **Bằng chứng.** Dr.Spider: mô hình bền nhất vẫn mất 14 % (50.7 % ở nhiễu khó nhất) khi câu hỏi bị biến đổi [13]; ViLexNorm [14]. Trong repo: paraphrase và câu nhiễu là hai loại dữ liệu có lợi rõ nhất.
- **Trạng thái.** Bất định: quy ước "M/YY" không suy ra được từ rulebook; chỉ kỳ vọng mô hình học rằng cách đọc cho **kỳ có dữ liệu và khớp kiểu viết của cả bộ câu** đáng tin hơn.

### F — Tài liệu API sinh tự động từ toolkit (trúng RC5; không huấn luyện)

- **Cơ chế.** Khối `# API` của prompt sinh từ chính đặc tả toolkit (`api_spec`: mọi hàm công khai, chữ ký, một dòng mô tả từ docstring) thay cho `API_DOC` viết tay; W1 chỉ gắn dòng rulebook với hàm có trong khối này. Không thêm luật nào: chỉ bảo đảm S nhìn thấy mọi công cụ nó được phép gọi.
- **Bằng chứng.** `purchased_qty_base` vắng trong prompt nên gần như không mẫu nào dùng. Rủi ro: S học trên `API_DOC` cũ nên bố cục mới có thể gây lệch → đo ghép cặp một lần chạy public; nếu âm thì chỉ bổ sung dòng thiếu vào cuối khối, và đưa bố cục mới vào dữ liệu của C.

## 4. Prioritization

Mức tăng tính từ số câu chạm tới và tỉ lệ sửa 40–70 %, đã trừ phần trùng. Tất cả là dự báo. Lưu ý hiệu chỉnh: kế hoạch 5 từng dự báo 0.96–0.975 và đạt 0.952.

| Hạng | Giải pháp | Gốc | Câu chạm tới mỗi log | Tăng PA dự kiến | Chi phí | Tổng quát sang private |
|---|---|---|---|---|---|---|
| 1 | **A** sửa đầu ra phán xử | RC3 | ~11 câu lùi về bỏ phiếu rồi sai / bỏ | +0.002 … +0.006 | vài giờ, không huấn luyện | Cao |
| 2 | **B** định tuyến xung đột rulebook | RC2, RC1a | 6–10 câu sai đồng thuận | +0.006 … +0.012 | huấn luyện ~1–2 h + phán xử lại log | Cao: private có rulebook mới |
| 3 | **C** S′ + rulebook đối chứng | RC1 | 8–13 (trùng với B) | +0.008 … +0.016 nếu đạt; có thể âm | huấn luyện 6–8 h × 2 seed | Rất cao |
| 4 | **E** dữ liệu tương phản | RC4 | 6–9 | +0.003 … +0.006 | gộp vào huấn luyện của B / C + WJR2 (~2.5 h) | Trung bình |
| 5 | **D** kênh thứ hai | RC2, RC5 | 3–6 | +0.002 … +0.006 | +1 mẫu dài / câu, không huấn luyện | Cao |

| 6 | **F** tài liệu API tự sinh | RC5 | 1–2 | +0.001 … +0.003 | vài giờ, một lần chạy public | Cao: ngành mới cần thấy đủ hàm |

Ưu tiên triển khai: **A và F ngay; B và C chạy song song** (cùng nguồn dữ liệu đối chứng; B là đường an toàn với S đóng băng, C là lời giải gốc); E đi kèm dưới dạng dữ liệu; D chỉ thử khi A + E xong.

## 5. Recommended plan

Mốc T0 = 14/11 17:00 UTC; đóng băng T+35 (16/11 04:00 UTC).

| Bước | Việc | Giờ | Tài nguyên | Cổng kiểm chứng |
|---|---|---|---|---|
| 0a | **Docker dự phòng** cho `flow_p5_final` (việc D1 còn nợ) để luôn có bản nộp 0.95 | T+0 → 8 | 1 acc | điểm container của đúng cấu hình WJR1 |
| 0b | **A:** phán xử có ràng buộc trên 4 log đã lưu, 4 biến thể / phiên; **F:** một lần chạy public với khối API tự sinh | T+0 → 4 | 1–2 acc | tỉ lệ đọc được ≥ 95 %; TB 4 log tăng so với 0.952, không log nào giảm |
| 0c | CPU: sinh dữ liệu đối chứng (xoá / đổi / nhiễu) và cặp tương phản từ cửa hàng train; kiểm nhãn bằng thực thi | T+0 → 4 | CPU | mỗi cặp chạy được, hai nhãn khác nhau đúng như dự định |
| 1 | Huấn luyện song song: **W2′** (kiểm định tiêu chí), **WJR2** (phán xử + tương phản + câu đồng thuận bị định tuyến), **S′** × 2 seed | T+4 → 13 | 4 acc, FSDP 4×L4 | cửa hàng train giữ ngoài: precision báo "thiếu định nghĩa" ≥ 0.9; độ nhạy rulebook của S′ ≥ 0.95 |
| 2 | **B** trên public: W2′ → định tuyến → phán xử trên log đã lưu (offline), rồi lượt chạy lại của S | T+7 → 14 | 2 acc | số câu sai đồng thuận / log giảm (hiện 8–19); số câu đúng bị làm hỏng ≤ 1/3 số câu sửa được |
| 3 | **C** trên public: S′ × 2 seed, cùng bộ phán xử; so ghép cặp với `f5a` / `f5b` | T+13 → 20 | 2–3 acc | tăng ghép cặp z ≥ 2 và hai seed cùng dấu; không đạt → giữ S |
| 4 | Chọn cấu hình; đường container × 2; thời gian 1,000 câu; cập nhật Docker, `README_SOLUTION` | T+20 → 31 | 2 acc | hai lần chạy lệch ≤ 0.03; trong ngân sách 6 giờ (hiện 56–75 phút / 4×L4) |
| Lùi | Thành phần nào không qua cổng → tắt bằng cấu hình; mọi thứ trượt → nộp bản 0a | T+31 → 35 | — | — |

Mọi so sánh ghép cặp trên cùng câu; A và B đo được trên log S cố định nên không có nhiễu sinh; C cần ≥ 2 seed (nhiễu giữa lần chạy ~0.01).

## 6. Đánh giá khách quan về mục tiêu 1.0

- **Đã chứng minh:** bộ phán xử huấn luyện +0.010–0.015 (kế hoạch 5); trần 986–987 với S hiện tại, 991 khi gộp nhiều lần chạy; 13 câu sai ở mọi log; hai biến thể phán xử mất 60–80 % phán quyết; S đúng khi được nối đúng dòng rulebook.
- **Giả thuyết:** bộ kiểm định tiêu chí đủ chính xác để định tuyến mà không phá câu đúng; S′ học dùng rulebook mà không hồi quy như v4; dữ liệu tương phản chuyển được sang quy ước viết của public.
- **Dự báo cộng dồn:** A → 0.954–0.958; thêm B hoặc C → 0.962–0.972; thêm E / D → tối đa ~0.975–0.98. Sau khi đối chiếu đáp án chuẩn, không còn câu nào "không biết đáp án": mọi câu sai đều có đáp án xác định, nên 1.0 không bị chặn về nguyên tắc. Nhưng nó đòi hỏi sửa hết 13 câu sai ở mọi log, cả các câu chỉ phụ thuộc quy ước viết ("4/26"), mà không làm hỏng câu nào trong 957 câu đang ổn định: **xác suất vẫn thấp**. Mốc thực tế: ≥ 0.965 trước đóng băng.
- **Rủi ro:** 35 giờ cho ba lần huấn luyện và lưới đánh giá; chính sách chọn trên chính public lạc quan 0.003–0.005; prompt dài hơn làm giảm số request đồng thời.

## 7. Nguồn tham khảo

1. Ni et al., *LEVER: Learning to Verify Language-to-Code Generation with Execution*, ICML 2023 — arXiv 2302.08468; mã: github.com/niansong1996/lever
2. Scholak et al., *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding*, EMNLP 2021 — arXiv 2109.05093
3. Zhang et al., *R-Tuning: Instructing LLMs to Say "I Don't Know"*, NAACL 2024 — arXiv 2311.09677; mã: github.com/shizhediao/R-Tuning
4. Lee et al., *TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring* — arXiv 2403.15879; mã: github.com/glee4810/TrustSQL
5. Han et al., *R³-SQL: Ranking Reward and Resampling for Text-to-SQL*, ACL Findings 2026 — arXiv 2604.25325
6. Wei et al., *Symbol Tuning Improves In-Context Learning in Language Models*, EMNLP 2023 — arXiv 2305.08298
7. Li et al., *Large Language Models with Controllable Working Memory* (KAFT), ACL Findings 2023 — arXiv 2211.05110
8. Kaushik et al., *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*, ICLR 2020 — arXiv 1909.12434
9. Li et al., *BIRD: Can LLM Already Serve as a Database Interface?*, NeurIPS 2023 — https://bird-bench.github.io
10. Krogh & Vedelsby, *Neural Network Ensembles, Cross Validation, and Active Learning*, NeurIPS 1994
11. Pourreza et al., *CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection*, ICLR 2025 — arXiv 2410.01943 (bỏ phiếu 68.84 → bộ chọn so cặp 73.01, trần 82.79)
12. Gao et al., *XiYan-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL* — arXiv 2411.08599; mã: github.com/XGenerationLab/XiYan-SQL
13. Chang et al., *Dr.Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL Robustness*, ICLR 2023 — arXiv 2301.08881
14. Nguyen et al., *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text*, EACL 2024
15. Fan et al., *Grounding NL to SQL Translation with Data-Based Self-Explanations* (CycleSQL), ICDE 2025 — arXiv 2411.02948; mã: github.com/Kaimary/CycleSQL
16. Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 — arXiv 2310.01798
17. vLLM, *LoRA Adapters* và *Structured Outputs* — https://docs.vllm.ai/en/stable/features/lora/ ; mã: github.com/vllm-project/vllm

Bối cảnh: [15] kiểm chương trình bằng cách diễn đạt ngược về ngôn ngữ tự nhiên — hướng mở rộng cho bộ phán xử nếu còn thời gian; [16] giải thích vì sao phán xử zero-shot trên mọi câu làm hỏng câu đúng, nên bộ phán xử phải được huấn luyện và chỉ xét câu có tín hiệu. Tài liệu nền khác: `docs/11_LITERATURE.md`.
