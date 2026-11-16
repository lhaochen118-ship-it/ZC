# 04 — Các bài toán tương tự DataMate và cách các nghiên cứu khác giải quyết

*Bản 2026-11-16. Tổng hợp từ tìm kiếm web (abstract, trang paper, leaderboard, một số bài tổng hợp thứ cấp). Con số
lấy từ abstract hoặc tóm tắt; chưa đọc hết toàn văn từng paper. Nguồn nào chỉ là blog/vendor được ghi rõ. Bổ sung cho
§2 của `01_PLAN.md` (danh sách paper 2026 ở đó chưa được kiểm lại trong bản này).*

---

## 1. Bài toán chung là gì

DataMate là giao của **ba dòng nghiên cứu**:

| Dòng | Câu hỏi cốt lõi | Đại diện |
|---|---|---|
| **Table QA / Spreadsheet QA** | Đọc và suy luận trên bảng (kể cả bảng "con người làm": header nhiều tầng, ô gộp, nhiều bảng trong một sheet) | TableBench, SheetBench/SheetBrain, SpreadsheetLLM, HCT-QA, DataBench (SemEval-2025 Task 8) |
| **Data agent / text-to-SQL / text-to-code** | Biến câu hỏi tự nhiên thành chương trình (SQL/pandas) nhiều bước, chạy trong sandbox, dùng tài liệu nghiệp vụ | DABstep, InfiAgent-DABench, DA-Code, DSBench, Spider 2.0, BIRD |
| **Selective prediction / abstention** | Biết khi nào *không* trả lời; câu không trả lời được; điểm phạt khi sai | TrustSQL, EHRSQL, TrustDABench, khảo sát "Know Your Limits" |

**Phát biểu hình thức (dùng chung):** cho gói dữ liệu bẩn $D$ (nhiều file/sheet), tri thức miền $K$ (README,
rulebook, lịch âm–dương), câu hỏi $q$ và mốc thời gian $t$; agent xuất $a \in \mathcal{A}_{\text{typed}} \cup
\{\texttt{unanswerable}, \texttt{abstain}\}$ để tối đa hoá tiện ích kỳ vọng với +1 đúng, 0 abstain, −1 sai. Bài toán
tách thành hai bài con:

1. **Sinh đáp án đúng** (độ chính xác $p$ của chương trình) — vấn đề của Table QA + data agent.
2. **Ước lượng $p$ có hiệu chuẩn và quyết định** (trả lời khi $p > 0.5$) — vấn đề của selective prediction.

Phần lớn benchmark chỉ đo (1). DataMate, TrustSQL và EHRSQL-2024 đo cả (2), và (2) là chỗ làm điểm âm.

---

## 2. Các bài toán / benchmark giống DataMate

Xếp theo độ gần, giảm dần.

| Benchmark | Giống DataMate ở đâu | Khác ở đâu | Mức khó báo cáo |
|---|---|---|---|
| **DABstep** (Adyen + Hugging Face, 2025) [1] | File CSV/JSON + **tài liệu Markdown chứa luật nghiệp vụ**; đáp án factoid, chấm tự động bằng khớp chính xác; nhiều biến thể của cùng một câu (đổi khoảng thời gian, tên merchant) | Một miền (thanh toán), dữ liệu khá sạch, không có abstain | Hard (84 % số câu): agent tốt nhất **14.55 %**; Easy 76.39 % |
| **TrustSQL** [2] / **EHRSQL 2024 shared task** [3] | **Thước đo gần như trùng PA**: đúng +1, abstain 0, sai (hoặc trả lời câu không trả lời được) bị phạt; câu unanswerable đa dạng | SQL trên DB sạch, tiếng Anh | **Không phương pháp nào thắng ổn định baseline "abstain tất cả"** |
| **SemEval-2025 Task 8 DataBench** [4] | QA trên bảng thật, đáp án có kiểu (boolean, số, category, list), chấm khớp chính xác; nhiều đội sinh code pandas/SQL | Một bảng mỗi câu, bảng sạch hơn, không abstain | Hạng 2/100 đội: 86.21 % (ensemble + voting) |
| **TrustDABench** (2026) [5] | Spreadsheet/CSV; đo **từ chối / hỏi lại khi không có đường bằng chứng hợp lệ** và **bất biến với cách trình bày bảng** (19 phép biến đổi) | Không chấm theo PA | Reliability tốt nhất 24.21 % MRS; mô hình hiếm khi phát hiện bằng chứng mâu thuẫn |
| **InfiAgent-DABench** [6] | Câu hỏi đóng trên CSV, agent viết code chạy trong sandbox, đáp án được ép định dạng để chấm tự động | CSV sạch, câu đơn bảng | GPT-4 ≈ 79 % |
| **DA-Code** [7] / **DSBench** [8] | Agent khoa học dữ liệu nhiều bước (wrangling, EDA), sandbox SQL + Python, nhiều bảng | Gồm cả ML, không có abstain | DA-Code ≈ 30.5 %; DSBench 34.12 % |
| **Spider 2.0** [9] | Workflow doanh nghiệp, schema lớn, cần đọc tài liệu ngoài, nhiều bước | DB lớn chuẩn, SQL dialect | o1-preview ≈ 21 % (vs 91.2 % Spider 1.0); ReFoRCE v5 ≈ 36 % |
| **BIRD** [10] | Giá trị bẩn, cần "evidence" (tri thức ngoài) để hiểu cột | Có sẵn evidence cho từng câu | SOTA ≈ 72–73 % EX |
| **TableBench** [11], **SheetBench** [12], **HCT-QA** [13] | Bảng công nghiệp / spreadsheet phức tạp (đa bảng, header phân cấp), suy luận số | Không có quy trình làm sạch đa file | TableBench: GPT-4 vẫn kém người nhiều |
| **AutoDCWorkflow** [14] | Làm sạch bảng bẩn *theo mục đích câu hỏi* (định dạng, kiểu, trùng lặp) rồi đánh giá bằng đáp án | Chỉ phần làm sạch | Mô hình 9–27B cải thiện rõ so với baseline |
| **TAG / TAG-Bench** [15] | Câu hỏi trên DB cần cả tri thức thế giới / ngữ nghĩa ngoài SQL | 80 câu | Phương pháp chuẩn ≤ 20 % |
| **ViText2SQL** [16], **ViBIRD** [17] | Tiếng Việt | Dịch từ Spider/BIRD, không khẩu ngữ, không dữ liệu bẩn | — |

**Nhận xét:** DataMate khó hơn từng benchmark riêng lẻ vì cộng dồn: dữ liệu bẩn kiểu SheetBench/HCT-QA + luật nghiệp
vụ kiểu DABstep + thước đo phạt kiểu TrustSQL + tiếng Việt khẩu ngữ + mô hình ≤ 15B offline. Trên DABstep-Hard và
Spider 2.0, cả mô hình frontier cũng chỉ đạt 15–36 %; độ chính xác ~37 % mỗi chương trình của ZC (14B) vì vậy không
bất thường, và chỉ ra rằng **bộ quyết định trả lời/abstain quan trọng ngang bộ sinh đáp án**.

---

## 3. Khó khăn trong quá trình suy nghĩ và hành động của agent

Xếp theo các bước của một vòng agent. Cột "Thấy ở ZC" dẫn chiếu `handoff/01_SCIENTIFIC_FINDINGS.md`.

| # | Bước | Khó khăn | Bằng chứng ngoài | Thấy ở ZC |
|---|---|---|---|---|
| K1 | **Nhận thức dữ liệu** | Layout bẩn: header lệch, header lặp, dòng tổng, ô gộp, nhiều bảng một sheet, một danh sách tách nhiều file; số/ngày nhiều định dạng | SpreadsheetLLM cần mô hình riêng để dò bảng (F1 78.9 %); TrustDABench: mô hình đổi đáp án khi đổi cách trình bày bảng | Đã giải phần lớn bằng `ingest.py` (78 → 0 cột đọc sai); lỗi `1,134tr` |
| K2 | **Grounding schema** (schema linking) | Gọi sai tên cột, sai bảng; schema lớn làm loãng ngữ cảnh | CHESS và Spider 2.0 đều coi schema linking là nút thắt chính | Sai tên cột (`'Thời gian'` 59 lần, `'Ngày bán'` 21) |
| K3 | **Grounding giá trị / thực thể** | Tên sản phẩm/khách viết nhiều kiểu; giá trị đặc biệt ("Khách lẻ", "kl"); trạng thái `ok`/`Xong`/`Đã hoàn thành` | CHESS dùng LSH trên giá trị DB vì câu hỏi không khớp chính xác giá trị lưu trữ | "Khách lẻ" bị xếp vào top khách hàng; số đơn huỷ sai (170 vs 61) |
| K4 | **Hiểu định nghĩa nghiệp vụ** | "Doanh thu", "đơn hợp lệ", "khách trung thành"… có định nghĩa riêng; luật đổi theo split | DABstep: phần khó là áp dụng luật trong tài liệu; báo cáo thực hành trên BIRD/DABstep: lỗi nằm ở *diễn giải định nghĩa*, không phải cú pháp SQL [18] | Đếm dòng thay vì đơn (147 vs 59); doanh thu không lọc ngày |
| K5 | **Suy luận thời gian** | Quy đổi "tuần rồi", "cùng kỳ năm ngoái" theo as-of date; tuần bắt đầu thứ mấy; âm lịch; ngày không tồn tại | Nghiên cứu tokenisation ngày (DateAugBench) cho thấy LLM yếu ở số học ngày; TRAVELER phân loại mốc ngầm "tháng này" là khó | ~165–231 lỗi "day is out of range" |
| K6 | **Lập kế hoạch nhiều bước** | Câu ghép (lọc → nhóm → so sánh kỳ → xếp hạng với tie-break) dễ quên một bước | DABstep Hard 14.55 %; DA-Code ≈ 30 %; Data Interpreter đề xuất kế hoạch dạng đồ thị vì kế hoạch tuyến tính gãy | Compile bảng chuẩn chỉ thành công 3/14 DN |
| K7 | **Hành động: sinh và chạy code** | Lỗi API (`.str` trên cột số), không gán biến kết quả, đọc file trái lệnh; vòng sửa lỗi có thể "sửa cho chạy được" nhưng sai nghĩa | CHESS/ReFoRCE đều có vòng tự sửa theo phản hồi thực thi | Sửa lỗi tăng 79 → 93 % mẫu chạy được |
| K8 | **Định dạng đáp án** | Khớp chính xác: sai kiểu (string vs percent), làm tròn, thứ tự list, table vs list | InfiAgent-DABench phải ép format để chấm; DataBench các đội có bước dự đoán kiểu đáp án | `"Giảm 1.79%"` thay vì `-1.79`; list −0.83 PA |
| K9 | **Biết khi nào không trả lời** | Phân biệt *unanswerable* (dữ liệu không có) với *không chắc* (abstain); mô hình "sai mà tự tin"; code chạy được nhưng không có bằng chứng | TrustSQL: không thắng abstain-all; TrustDABench: tiếp tục phân tích "chạy được nhưng không có căn cứ"; "Entropy Alone is Insufficient": ảo giác entropy thấp lọt qua [19] | PA −0.14; 4/4 phiếu chỉ đúng 65 %; 2/4 phiếu đúng 24 % |
| K10 | **Chọn ứng viên** | Majority vote thất bại khi lỗi tương quan (mọi mẫu cùng hiểu sai một định nghĩa) | CHASE-SQL thay vote bằng selector học được; ReFoRCE chỉ vote rồi đào sâu khi không đồng thuận | Đồng thuận thấp; nhánh A và B lệch nhau |
| K11 | **Ràng buộc tài nguyên** | ≤ 15B, offline, 6 giờ, tất định | DataBench: mô hình lớn hơn vượt trội; Arctic-R1: 7–14B tiệm cận GPT-4o sau RL | Thông lượng thấp, compile chậm |

Có thể gom các khó khăn trên thành **ba loại lỗi gốc**:
- **lỗi hiểu** (K2–K5): agent hiểu sai *câu hỏi muốn gì* so với *dữ liệu có gì*;
- **lỗi làm** (K6–K8): hiểu đúng nhưng chương trình hoặc định dạng sai;
- **lỗi tự đánh giá** (K9–K10): không biết mình sai.

Các báo cáo DABstep/BIRD và TrustDABench đều cho thấy loại đầu và loại cuối chiếm phần lớn khi mô hình đã viết code
tốt. Phân tích lỗi của ZC cũng cho thấy như vậy.

---

## 4. Các phương pháp phổ biến

| Nhóm | Ý tưởng | Đại diện |
|---|---|---|
| **M1. Biểu diễn và profile dữ liệu** | Dò bảng/header, nén spreadsheet, profile cột (kiểu, giá trị rời rạc, phạm vi ngày) trước khi hỏi; biên dịch tri thức schema một lần (offline) | SpreadsheetLLM (SheetCompressor, giảm 96 % token) [20], AutoDCWorkflow [14], ProfiliTable [21], "Analyst-first knowledge compilation" [22] |
| **M2. Suy luận bằng chương trình** | LLM sinh SQL/pandas/Python, kết quả do máy tính ra (không để LLM tự viết số) | Binder, DATER, Program-of-Thought, TableLLM, TableGPT2 [23][24] |
| **M3. Thao tác bảng từng bước** | LLM chọn một phép toán bảng mỗi bước từ tập định nghĩa sẵn (DSL) | Chain-of-Table [25] |
| **M4. Schema linking và value retrieval** | Lọc cột/bảng liên quan; chỉ mục giá trị (LSH, vector) để khớp tên thực thể; khám phá cột lặp | CHESS [26], ReFoRCE (column exploration) [27], LinkAlign |
| **M5. Đưa tri thức miền vào** | Evidence theo câu (BIRD); "semantic layer" / định nghĩa metric dạng code; skill pack theo miền | BIRD [10], DABstep docs [1], blog semantic layer [18], skills [22] |
| **M6. Vòng ReAct với phản hồi thực thi** | Suy nghĩ → chạy → đọc lỗi/kết quả → sửa; kiểm tra logic ngoài lỗi runtime | ReAct, Data Interpreter [28], DA-Agent [7], CHESS candidate generator |
| **M7. Sinh nhiều ứng viên + chọn** | Self-consistency; đa dạng hoá đường giải (divide-and-conquer, query plan, few-shot tổng hợp); chọn bằng vote, selector cặp đôi đã fine-tune, unit test | CHASE-SQL [29], ReFoRCE, Samsung @ SemEval-25 (ensemble + vote) [4] |
| **M8. Abstention và hiệu chuẩn** | Ngưỡng trên điểm tin cậy (đồng thuận, entropy, beam score), classifier lỗi riêng, đánh giá risk–coverage; RL với reward có hành động từ chối | TrustSQL [2], EHRSQL [3], selective classification text-to-SQL [30], khảo sát Know Your Limits [31], Abstain-R1 [32], RLSR (ICML 2026) |
| **M9. Huấn luyện mô hình nhỏ bằng phản hồi thực thi** | SFT trên chương trình đúng (STaR/rejection sampling), sau đó GRPO với reward = đúng khi thực thi | Arctic-Text2SQL-R1 [33], OmniSQL |

---

## 5. Nghiên cứu khác đã giải từng vấn đề như thế nào

Mỗi mục: **cách làm → bằng chứng → bài học cho ZC** (kèm hiện trạng repo).

### 5.1 Dữ liệu bẩn và nhiều định dạng (K1)

- **SpreadsheetLLM** [20]: dùng "structural anchors" để tìm ranh giới bảng, gom ô cùng định dạng số, nén sheet thành
  biểu diễn nhỏ (giảm 96 % token); fine-tune mô hình dò bảng (F1 78.9 %). QA theo "Chain-of-Spreadsheet": tìm bảng
  liên quan trước, rồi mới trả lời.
- **AutoDCWorkflow** [14]: làm sạch *có mục đích*. LLM chọn cột cần cho câu hỏi → kiểm tra chất lượng cột → chọn phép
  làm sạch (định dạng, kiểu, trùng lặp) dưới dạng thao tác OpenRefine. Chấm bằng việc bảng sạch có ra đáp án đúng không.
- **TrustDABench** [5]: đo độ bền với 19 phép biến đổi cách trình bày. Mô hình nhạy với các biến đổi đổi ranh giới
  quan sát hoặc quan hệ giữa các bảng.
- **Bài học ZC:** hướng "làm sạch tất định, tổng quát, có test tổng hợp" (`ingest.py`) đi trước literature ở điểm không
  để LLM tự làm sạch (nhanh, tất định). Còn thiếu so với [5]: kiểm tra độ bền (bất biến) bằng cách tự biến đổi bảng
  train (đổi thứ tự cột, tách sheet, đổi định dạng số) rồi xem đáp án có đổi không.

### 5.2 Schema linking và khớp thực thể (K2, K3)

- **CHESS** [26]: (i) trích từ khoá trong câu hỏi; (ii) tìm giá trị gần đúng trong DB bằng MinHash/LSH + vector index
  (xây một lần khi tiền xử lý); (iii) lọc schema 3 tầng (lọc cột → chọn bảng → chọn cột cuối). Kết quả: 71.1 % BIRD
  test với ít lời gọi LLM hơn ~83 %.
- **ReFoRCE** [27]: nén schema, khi không chắc thì *chạy truy vấn khám phá cột* (xem giá trị thật) trước khi viết
  truy vấn cuối.
- **Bài học ZC:** lỗi sai tên cột → đưa **schema card có danh sách cột chính xác + giá trị mẫu** vào prompt và
  **kiểm tra tĩnh trước khi chạy** (tên cột trong code ∈ schema; không thì yêu cầu sửa ngay, không tốn một lần chạy).
  Lỗi "Khách lẻ" → **entity index** (giá trị rời rạc của cột tên khách/trạng thái, đánh dấu sẵn giá trị đặc biệt như
  khách lẻ/vãng lai) — đã có trong `01_PLAN.md` [A] nhưng chưa làm.

### 5.3 Luật nghiệp vụ và định nghĩa metric (K4)

- **DABstep** [1]: cung cấp luật dạng Markdown. Lỗi chủ yếu đến từ việc agent không áp dụng đúng luật trong tài liệu
  qua nhiều bước. Báo cáo thực hành [18] (blog, chưa được bình duyệt): trên 25 câu BIRD + DABstep không có lỗi nào là
  lỗi cú pháp SQL; lỗi nằm ở diễn giải định nghĩa và trả lời câu dữ liệu không trả lời được. Một vendor (MotherDuck)
  báo đạt 100 % DABstep nhờ một **lớp ngữ nghĩa phân cấp** trên dữ liệu thô (chưa kiểm chứng độc lập).
- **BIRD** [10]: gắn "evidence" (ví dụ công thức metric) cho từng câu; các hệ thống tốt nhất đều dùng evidence.
- **Analyst-first** [22]: biên dịch tri thức miền thành "skill pack" + tri thức schema đã kiểm (chạy lại SQL bằng
  chứng cho từng metric).
- **Bài học ZC:** đây là điểm literature ủng hộ mạnh nhất cho ý tưởng gốc [B] "rulebook compiler" trong `01_PLAN.md`
  (hiện đã thay bằng compile bảng chuẩn và mới thành công 3/14 DN). Cách nhẹ hơn, khả thi với 14B: **thư viện helper
  viết tay theo khái niệm tổng quát** (`valid_orders(df)`, `revenue(df, start, end)`, `is_walk_in(name)`), đọc tham
  số từ rulebook lúc chạy, LLM chỉ gọi hàm. Cách này giữ đúng Rule 4 (không theo id). Các lỗi "đếm dòng thay vì
  đơn" và "doanh thu không lọc ngày" là loại lỗi một helper loại bỏ hoàn toàn.

### 5.4 Thời gian tương đối và lịch (K5)

- Literature riêng cho text-to-SQL còn ít. Nhận định chung (DateAugBench, TRAVELER) là LLM yếu ở số học ngày và mốc
  ngầm, nên các hệ thống thực tế **giải mốc thời gian bằng code tất định** chứ không để LLM tính.
- **Bài học ZC:** `resolve_period("tuần rồi", as_of) -> (start, end)` viết tay, phủ các biểu thức hay gặp trong train
  + lịch âm từ `calendar.csv`; `safe_date(y, m, d)` kẹp ngày cuối tháng. Gần như loại bỏ ~200 lỗi "day is out of range".

### 5.5 Lập kế hoạch và sinh chương trình (K6, K7)

- **Data Interpreter** [28]: kế hoạch dạng đồ thị phân cấp (task graph), cập nhật khi dữ liệu trung gian thay đổi;
  kiểm tra tính hợp lý logic của code, không chỉ lỗi runtime; ghi lại kinh nghiệm để tái dùng.
- **Chain-of-Table** [25] / **DSL**: giới hạn hành động vào tập phép toán định sẵn → ít lỗi API hơn, dễ kiểm tra.
- **CHESS / ReFoRCE**: vòng tự sửa theo lỗi thực thi, kèm chuẩn hoá (format restriction).
- **Bài học ZC:** vòng sửa lỗi đã có hiệu quả (+14 điểm % mẫu chạy được). Bước còn thiếu là **kiểm tra ngữ nghĩa sau
  khi chạy**: kết quả 0/rỗng/âm bất thường, doanh thu > tổng doanh thu cả kỳ, ngày ngoài phạm vi dữ liệu. ZC đã có
  self-check cho 0/rỗng; có thể mở rộng thành các *assert kiểu unit test* (CHESS gọi là Unit Tester).

### 5.6 Định dạng đáp án (K8)

- InfiAgent-DABench ép mọi câu về định dạng đóng; các đội DataBench có bước **dự đoán kiểu đáp án** riêng trước khi
  sinh code [4].
- **Bài học ZC:** tách việc chọn kiểu khỏi việc tính toán: một lời gọi ngắn (hoặc luật từ khoá: "bao nhiêu %",
  "tăng/giảm" → percent có dấu; "những…", "top k" → list; "có … không" → boolean), sau đó formatter tất định ép
  output chương trình về kiểu đó. Như vậy `"Giảm 1.79%"` không thể xuất hiện.

### 5.7 Chọn ứng viên (K10)

- **Self-consistency** cơ bản: vote trên kết quả thực thi.
- **CHASE-SQL** [29]: đa dạng hoá *cách sinh* (divide-and-conquer, suy luận theo query plan, few-shot tổng hợp riêng
  cho từng câu) để lỗi ít tương quan; chọn bằng **selector so sánh cặp** đã fine-tune thay vì đa số. 73.0 % BIRD test.
- **ReFoRCE** [27]: vote; nếu không đồng thuận thì *hoãn* và khám phá thêm dữ liệu rồi sinh lại.
- **Samsung @ SemEval-25** [4]: nhiều mô hình/nhiều prompt + vote, kèm diễn đạt lại câu hỏi và vòng sửa lỗi → 86.21 %.
- **Bài học ZC:** với 4/4 phiếu chỉ đúng 65 %, các mẫu đang *sai giống nhau* (cùng hiểu sai định nghĩa). Đa dạng hoá
  nên nằm ở **cách hiểu** (diễn đạt lại câu hỏi; nhánh helper vs nhánh code tự do; nhánh SQL/DuckDB vs pandas), không
  chỉ ở nhiệt độ. Cách của ReFoRCE (không đồng thuận → khám phá thêm, nếu vẫn không đồng thuận → abstain) hợp với PA.

### 5.8 Abstention và câu không trả lời được (K9) — quan trọng nhất cho PA

- **TrustSQL** [2]: mô hình chọn sinh SQL hoặc abstain; điểm phạt cho SQL sai và cho việc trả lời câu unanswerable
  (với hệ số phạt 1 thì giống PA). So sánh hai họ phương pháp: *pipeline* (bộ lọc câu unanswerable + bộ sinh + bộ dò
  lỗi riêng) và *mô hình thống nhất*. **Không cách nào thắng ổn định baseline abstain-tất-cả.**
- **EHRSQL 2024** [3]: yêu cầu trả lời câu làm được, abstain câu unanswerable hoặc dễ sai. [34] dùng beam score và
  entropy làm tín hiệu bất định, kèm lọc theo n-gram câu hỏi, và chỉ ra rằng đánh giá kiểu này dễ bị bias của dữ liệu
  chi phối (bộ lọc học thuộc mẫu câu), nên cần đánh giá không thiên lệch.
- **Selective classification cho text-to-SQL** [30]: điểm bất định + đánh giá risk–coverage; tăng recall cho câu
  unanswerable nhưng tổng quát kém với câu dài/phức tạp.
- **"Entropy Alone is Insufficient"** [19]: nhiều câu sai có độ tự tin cao; AUROC tốt chưa đủ, cần đo ở ngưỡng rủi ro
  chặt (AURC).
- **Abstain-R1** [32], **RLSR** (ICML 2026): RL với reward có hành động từ chối, tối ưu trực tiếp risk–coverage.
- **TrustDABench** [5]: mô hình hiếm khi nhận ra "không có đường bằng chứng" và vẫn phân tích tiếp.
- **Bài học ZC:**
  1. Kết quả −0.14 của ZC lặp lại đúng phát hiện của TrustSQL: khi $p$ ~ 0.4, mọi ngưỡng lỏng đều thua abstain-all.
     **Ngưỡng phải chọn trên DN holdout bằng PA**, không đặt τ = 0.5 theo lý thuyết, vì tỷ lệ phiếu chưa hiệu chuẩn.
  2. Tỷ lệ phiếu là một đặc trưng, không phải xác suất. Ghép thêm đặc trưng rẻ (kiểu đáp án, loại câu, có lỗi/sửa lỗi
     không, kết quả 0/rỗng, nhánh A và B có khớp không, ngày truy vấn có nằm trong phạm vi dữ liệu không) vào một
     hồi quy logistic nhỏ trên train → $p$ đã hiệu chuẩn. Đây là [F] trong `01_PLAN.md`.
  3. `unanswerable` cần **tín hiệu tất định** (khái niệm không có cột tương ứng; khoảng thời gian ngoài dữ liệu)
     thay vì để LLM tự đoán, đúng như cách pipeline của TrustSQL tách bộ lọc unanswerable riêng.
  4. Abstain theo loại: các loại có PA âm ổn định trên holdout (list −0.83, string −0.71) có thể abstain mặc định cho
     tới khi được sửa.

### 5.9 Mô hình nhỏ, giới hạn tính toán (K11)

- **Arctic-Text2SQL-R1** [33]: GRPO với reward đơn giản (đúng khi thực thi + điểm cú pháp), khởi đầu từ mô hình đã
  mạnh (Qwen2.5-Coder / OmniSQL); 14B đạt 70.04 % BIRD test, 7B ngang GPT-4o trên 6 benchmark. GRPO tốt hơn PPO; khởi
  đầu yếu thì RL không bù được.
- **TableLLM / TableGPT2** [23][24]: SFT trên dữ liệu bảng tổng hợp (chuyển SQL → pandas) giúp mô hình 7–13B mạnh lên
  với code pandas.
- **Bài học ZC:** hướng E7 (STaR → SFT → GRPO với reward = PA) có cơ sở, và reward có hành động abstain là điểm mới
  so với Arctic-R1. Nhưng E7 tốn nhiều giờ GPU và cần pipeline đã ổn định. Nó không phải đòn bẩy rẻ nhất.

---

## 6. Ánh xạ sang hệ thống ZC: ưu tiên theo chi phí / lợi ích

Hạn nộp Docker là **16/11 23:59 (GMT+7)**, đúng hôm nay; sau đó solution bị đóng băng. Vì vậy bảng tách ra những việc
còn làm kịp và những việc chỉ có ý nghĩa cho các vòng sau.

| Ưu tiên | Việc | Giải quyết | Cơ sở từ literature | Chi phí |
|---|---|---|---|---|
| 1 | Chọn τ (và abstain theo type) trên holdout bằng PA; mặc định bảo thủ (chỉ trả lời khi đồng thuận tuyệt đối) | K9 | TrustSQL, EHRSQL, kết quả 4/4 → +0.12 trong mẫu | Rất thấp |
| 2 | Formatter ép kiểu theo luật từ khoá câu hỏi | K8 | DataBench, InfiAgent | Thấp |
| 3 | Helper ngày an toàn + resolver mốc thời gian | K5, K7 | Thực hành chung | Thấp |
| 4 | Kiểm tra tĩnh tên cột trước khi chạy; nhắc `.astype(str)` | K2, K7 | CHESS, ReFoRCE | Thấp |
| 5 | Helper nghiệp vụ (đơn hợp lệ, doanh thu, khách lẻ) đọc tham số từ rulebook | K3, K4 | DABstep, semantic layer, BIRD evidence | Trung bình |
| 6 | Entity/value index cho cột tên & trạng thái | K3 | CHESS LSH | Trung bình |
| 7 | Bộ quyết định logistic trên nhiều đặc trưng | K9 | Selective classification [30] | Trung bình |
| 8 | Đa dạng hoá cách hiểu (diễn đạt lại, SQL vs pandas) + "không đồng thuận → khám phá → abstain" | K10 | CHASE-SQL, ReFoRCE | Trung bình–cao |
| 9 | SFT/GRPO với reward PA | K6–K9 | Arctic-R1, Abstain-R1 | Cao |

Mục 1–4 không đổi kiến trúc, nên còn kịp trước hạn nếu có thời gian kiểm lại trên holdout. Mục 1 là bảo hiểm quan
trọng nhất: một agent abstain nhiều vẫn có PA ≥ 0, còn agent trả lời lỏng tay thì đang âm.

---

## 7. Tài liệu tham khảo

1. DABstep: Data Agent Benchmark for Multi-step Reasoning — <https://arxiv.org/abs/2506.23719>; leaderboard <https://huggingface.co/spaces/adyen/DABstep>
2. TrustSQL: A Reliability Benchmark for Text-to-SQL Models with Diverse Unanswerable Questions — <https://arxiv.org/abs/2403.15879>
3. Overview of the EHRSQL 2024 Shared Task on Reliable Text-to-SQL — <https://arxiv.org/html/2405.06673v2>; EHRSQL <https://arxiv.org/pdf/2301.07695>
4. SemEval-2025 Task 8: Question Answering over Tabular Data — <https://preview.aclanthology.org/setup/2025.semeval-1.324>; Samsung R&D Poland <https://research.samsung.com/blog/SemEval-2025-Task-8-LLM-Ensemble-Methods-for-QA-Over-Tabular-Data>; Agentic LLMs for QA over Tabular Data <https://arxiv.org/abs/2509.09234>
5. TrustDABench: Benchmarking Reliability and Robustness of LLMs for Structured Data Analysis — <https://arxiv.org/abs/2608.24145>
6. InfiAgent-DABench — <https://arxiv.org/abs/2401.05507>
7. DA-Code (EMNLP 2024) — <https://aclanthology.org/2024.emnlp-main.748>
8. DSBench (ICLR 2025) — <https://arxiv.org/abs/2409.07703>
9. Spider 2.0 (ICLR 2025) — <https://arxiv.org/abs/2411.07763>
10. BIRD (qua LinkAlign và CHESS) — <https://arxiv.org/pdf/2503.18596>
11. TableBench (AAAI 2025) — <https://arxiv.org/abs/2408.09174>
12. SheetBrain / SheetBench — <https://arxiv.org/pdf/2510.19247>
13. HCT-QA: Human-Centric Tables — <https://arxiv.org/abs/2504.20047>
14. AutoDCWorkflow (Findings EMNLP 2025) — <https://arxiv.org/abs/2412.06724>
15. Text2SQL is Not Enough: TAG (CIDR 2025) — <https://arxiv.org/abs/2408.14717>
16. ViText2SQL (Findings EMNLP 2020) — <https://preview.aclanthology.org/fix_video/2020.findings-emnlp.364.pdf>
17. ViBIRD — <https://huggingface.co/datasets/hoadm/vibird>
18. Blog thực hành, không bình duyệt: "Semantic Layer = Map for Chat-BI Agents" <https://juhache.substack.com/p/semantic-layer-map-for-chat-bi-agents>; MotherDuck <https://motherduck.com/blog/oops-maybe-we-do-need-semantic-layers/>
19. Entropy Alone is Insufficient (selective prediction) — <https://arxiv.org/abs/2603.21172>
20. SpreadsheetLLM — <https://arxiv.org/abs/2407.09025>
21. ProfiliTable: Profiling-Driven Tabular Data Processing via Agentic Workflows — <https://arxiv.org/pdf/2605.12376>
22. From Question-First to Analyst-First: Domain-Expert Skills and Verified Knowledge Compilation — <https://arxiv.org/abs/2608.28594>
23. TableLLM — <https://arxiv.org/abs/2403.19318>
24. TableGPT2-7B — <https://huggingface.co/tablegpt/TableGPT2-7B>
25. Khảo sát Table Reasoning with LLMs (Chain-of-Table, Binder, DATER) — <https://arxiv.org/abs/2402.08259>; LLM for Table Processing survey <https://arxiv.org/abs/2402.05121>
26. CHESS — <https://arxiv.org/abs/2405.16755>
27. ReFoRCE — <https://arxiv.org/abs/2502.00675>
28. Data Interpreter — <https://arxiv.org/abs/2402.18679>
29. CHASE-SQL (ICLR 2025) — <https://arxiv.org/abs/2410.01943>
30. Confidence Estimation for Error Detection in Text-to-SQL — arXiv 2501.09527 (tóm tắt: <https://themoonlight.io/review/confidence-estimation-for-error-detection-in-text-to-sql-systems>)
31. Know Your Limits: A Survey of Abstention in LLMs (TACL) — <https://arxiv.org/abs/2407.18418>
32. Abstain-R1 — <https://arxiv.org/html/2604.17073>
33. Arctic-Text2SQL-R1 — <https://arxiv.org/abs/2505.20315>
34. Towards Unbiased Evaluation of Detecting Unanswerable Questions in EHRSQL — <https://arxiv.org/abs/2405.01588>
