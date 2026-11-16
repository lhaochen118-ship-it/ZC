# BỔ SUNG NGHIÊN CỨU: 14 BIẾN THỂ CHỌN LỌC CHO 7 GIẢI PHÁP DATAMATE P5

**Ngày đánh giá:** 15/11/2026 (GMT+7)  
**Repository:** `lhaochen118-ship-it/ZC`, nhánh `claude/datamate-v3-0dny90`  
**Mục tiêu:** Lựa chọn hai biến thể có cơ chế thực sự khác nhau cho từng hướng N1–N7, đối chiếu kiến trúc P5 hiện tại và thực hiện các kiểm tra khả thi có thể chạy được.  
**Trạng thái:** Đã kiểm tra mã nguồn GitHub và lịch sử score; đã chạy 12/12 bài kiểm tra CPU tổng hợp; **chưa chạy GPU/Kaggle và chưa có kết quả D1 mới của N1–N7**.

> **Quy tắc cứng:** Không thêm luật nghiệp vụ viết tay; Qwen phải tự suy luận và sinh luật áp dụng cho mỗi ngữ cảnh, schema và dữ liệu mới. Chỉ dùng những phép kiểm tra toán học, dữ liệu, type, cú pháp và bảo mật mang tính tổng quát. Tất cả model/adapter/probe triển khai phải có tổng parameters thực tế **dưới 15B**. Cấm dùng câu trả lời public/private test để học/chọn ngưỡng hoặc viết điều kiện đặc thù.

## 1. Baseline đã thực sự có trong GitHub

Từ [`Process/scores_p5zs.csv`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/Process/scores_p5zs.csv), hai lượt chạy **P5 strict zero-shot** đã hoàn thành:

| Lượt | Cấu hình | Đúng / Sai / Abstain | Penalized accuracy | Thời gian |
|---|---|---|---:|---:|
| `b00s0` | `flow_p5_zs_strict.json` | 832 / 79 / 89 | **0.753** | 110 phút |
| `b00s1` | `flow_p5_zs_strict_s1.json` | 831 / 76 / 93 | **0.755** | 113 phút |
| **B00-S** | **2 seeds, trung bình** | — | **0.754** | — |

Mốc **P5 tuned 0.952** (`configs/flow_p5_final.json`, V3A × 0.70 + WL1 + WJR1, D1) **chỉ là reference**. Baseline cho tất cả thử nghiệm zero-shot N1–N7 phải là **B00-S = 0.754** trên cùng D1 protocol và seed matching.

Trong hai lượt baseline, phần câu hỏi thuộc nhóm ngành có PA khoảng **0.580–0.595**, thấp hơn phần còn lại khoảng **0.811–0.819**. Đây là tín hiệu cho việc kiểm tra schema linking, đơn vị và việc Qwen viết luật cho rulebook mới; **chưa phải bằng chứng** các lỗi đó là nguyên nhân chính.

Một số kết quả đã thực sự chạy khác trong repo: P02 pruned MoE **0.781**, P03 **0.773**, P01 REAP-64 **0.429**, M01 Qwen3-14B **0.709**. Chúng là thí nghiệm *khác*, **không** chứng minh hiệu quả cho N1–N7. Các lượt `l02a*` dùng trọng số fine-tuned nên không được xếp chung nhóm zero-shot; `b00i` dùng đường có hàm `industry` viết tay nên không đáp ứng nguyên tắc model tự sinh luật.

## 2. Những module đã có và lý do phải chọn biến thể kỹ

| Module trong repository | Đã hỗ trợ | Điều mới còn thiếu / rủi ro chồng lặp |
|---|---|---|
| `src/perturb.py` | Biến đổi **câu hỏi** (noise, bỏ dấu) để stress test | N1 mới phải là **biến đổi dữ liệu** với quan hệ toán học có điều kiện, không làm lại question perturbation |
| `src/precheck.py` | Kiểm AST, cú pháp, API, một số heuristic có flags | N1/N3/N7 không nên biến thành tập hợp luật nghiệp vụ mới; `unit_check` hiện bị tắt trong P5 strict |
| `src/units.py` | Qwen-generated unit function, kiểm bằng chạy trên nhiều kỳ | N7 chỉ nên bổ sung **dimension của kết quả/đại lượng**, không viết lại unit function verification |
| `src/automap.py` | Canonical tables và mapping, nhiều heuristic diễn giải dữ liệu | N4 không làm lại mapping tên vai trò; thử graph từ khóa/giá trị **trung lập** trên raw & canonical, để Qwen chọn ý nghĩa |
| `src/retrieval.py` | Few-shot retrieval, `schema_sig` Jaccard | N4 phát hiện **đường kết nối bảng**, không chỉ similarity của column set |
| `src/sqlesc.py` | DuckDB SQL SELECT an toàn trên DataFrames | N6 tạo **cross-engine equivalence tests**, không chỉ bật SQL engine đã có |
| `src/hybrid.py`, `src/judge.py` | Sampling, sửa lỗi, voting và judge | N3 là **sinh phản ví dụ / ràng buộc để phân biệt ứng viên**, không phải gọi thêm judge đoán đúng/sai |
| `src/ingest.py`, `src/dm.py` | Dữ liệu raw → canonical, primitives | N2 lineage có thể khó nếu ID nguồn biến mất khi canonical hóa; cần đo coverage trước |

**Lưu ý compliance:** `src/automap.py` có những phương pháp `build_sales`, `build_conversions`, `classify` chứa logic ánh xạ kiểu dữ liệu và nghiệp vụ. `hide_api: industry` chỉ là một phần của chế độ H; nó **không tự chứng minh mọi thành phần pipeline đều không chứa heuristic nghiệp vụ viết tay**. Cần audit riêng nhưng không thay đổi source baseline trong vòng so sánh đầu tiên. Biến thể mới không được thêm logic nghiệp vụ hard-code vào các module đó.

## 3. Mười bốn biến thể — hai phương án có mục đích khác nhau cho mỗi hướng

### N1. Metamorphic Data Testing

**N1-A — Permutation invariance có chứng minh tiền điều kiện (P1).**

- **Vị trí:** Sau `src/hybrid.py` thực thi candidate; wrapper dữ liệu được tạo tạm bên ngoài sandbox chính.
- **Cơ chế:** Nếu thao tác trong chương trình là đại số bất biến theo thứ tự (ví dụ tổng trên tập bản ghi), tạo bản dữ liệu hoán vị dòng và chạy lại đúng chương trình. Một vi phạm là tín hiệu không ổn định; Qwen quyết định có cần sửa.
- **Đối chứng:** Tắt metamorphic check, giữ nguyên K programs/votes/seed; metric `violation_rate`, `wrong_rejected`, `correct_lost`, ΔZS.
- **Cảnh báo:** Cấm áp dụng cho `first/last`, window functions, phép xếp hạng có tie không được xác định, time ordering, hoặc thao tác có side-effects.

**N1-B — Partition-recomposition law cho aggregate (P2).**

- **Cơ chế:** Với aggregate có phép ghép đã được chứng minh (SUM/COUNT bằng cộng; MIN/MAX bằng min/max; AVG bằng `(sum,count)`), chia dữ liệu đã lọc thành các phân vùng và kiểm kết quả gộp bằng phép toán tương ứng.
- **Khác N1-A:** Phát hiện sai aggregation/join multiplicity mà việc đảo dòng không phát hiện. Không suy từ từ khóa câu hỏi; xác định aggregate bằng AST/program + Qwen-generated spec.
- **Cổng:** Chỉ thử nếu tỷ lệ candidate thuộc subset hỗ trợ >5–10% và N1-A tìm được ít nhất vài lỗi thực trên train/dev.

### N2. Data Provenance & Evidence Certificates

**N2-A — Canonical-row lineage (P2).**

- **Vị trí:** Instrument những thao tác lọc/nhóm tổng hợp đi qua `C[table]` ở `dm`/sandbox; gắn ID không ảnh hưởng ngữ nghĩa.
- **Cơ chế:** Cung cấp tập source IDs tham gia kết quả và kiểm bằng replay subset; Qwen tự so dấu vết với mục tiêu câu hỏi. Bắt đầu với một bảng, SUM/COUNT để giới hạn phạm vi.
- **Cổng:** >80% queries thuộc subset thử có lineage hợp lệ; instrumentation không đổi output baseline.

**N2-B — Raw-to-canonical lineage + join-multiplicity audit (P3).**

- **Cơ chế:** Propagate `_source_file`, `_sheet`, `_row_id` từ `ingest` qua `automap` sang canonical C, ghi số lần mỗi source row tham gia JOIN. Phát hiện fan-out bất thường và đối chiếu cách tổng hợp với plan Qwen.
- **Khác N2-A:** Kiểm được lỗi xuất hiện ở quá trình chuẩn hóa và nối bảng, không chỉ trong program cuối; sửa sâu nên chỉ làm khi N2-A cho thấy giá trị.
- **Cổng:** Có thể chứng minh bản raw→canonical không mất row IDs quan trọng, không thay đáp án chuẩn vì instrumentation.

### N3. Counterexample-Guided Program Synthesis (CEGIS)

**N3-A — Counterexample discrimination giữa các candidate (P1/P2).**

- **Vị trí:** Khi ít nhất hai chương trình trả cùng số trên dữ liệu thật nhưng có cấu trúc khác nhau, sinh bảng nhỏ phù hợp schema để tìm đầu vào khiến chúng phân kỳ.
- **Cơ chế:** CEGIS tìm **chứng cứ hai cách tính không tương đương**, rồi yêu cầu Qwen base chọn biểu thức nào khớp câu hỏi/rulebook. Không dùng solver để tự khẳng định chương trình đúng.
- **Điểm khác:** Khác voting/judge có sẵn; phương pháp tìm phản ví dụ toán học thay vì thêm một lời đánh giá LLM.
- **Cổng:** Chỉ gọi khi candidate equivalence class đáng ngờ, timeout nhỏ (≤1–3 giây/câu nếu CPU), không suy ý định từ mẫu giả.

**N3-B — Local AST hole filling theo Qwen-generated specification (P3).**

- **Cơ chế:** Qwen nêu chương trình khung và tập phép biến đổi ứng viên ở một node (aggregation/filter/join key). Dùng những test constraints từ train hoặc do Qwen tạo và kiểm độc lập để chọn bản thỏa ràng buộc.
- **Khác N3-A:** Có **sửa localized program** thay vì chỉ phân biệt hai chương trình. Không tự tạo nhãn đúng từ public/private test; không liệt kê luật nghiệp vụ theo ngành.
- **Cổng:** Có bằng chứng N3-A phát hiện vấn đề có thể quy về 1 node; nếu không, dừng.

### N4. Graph-Based Schema Linking

**N4-A — Structural join candidate graph (P0/P1).**

- **Vị trí:** `src/ingest.py` và `src/retrieval.py` lấy thông tin raw/canonical: tên bảng, khóa, overlap giá trị, uniqueness, null profile; tạo graph candidates (không áp luật ngành).
- **Cơ chế:** Thêm 3–5 edges có confidence vào prompt để Qwen tự quyết định JOIN; tránh ép dùng graph khi schema canonical đã đủ.
- **Cổng:** Trên train/dev, so số lỗi JOIN, số correct-to-wrong; không thêm logic mapping question→table dưới dạng regex.

**N4-B — Join path + cardinality-aware planner, Qwen approval (P1/P2).**

- **Cơ chế:** Khi Qwen yêu cầu nhiều bảng, chọn các đường nối ngắn/ít rủi ro theo edge graph; kèm ước lượng cardinality và nêu các lựa chọn để Qwen quyết định. Có thể thực thi trên raw SQL qua `src/sqlesc.py`.
- **Khác N4-A:** Không chỉ cung cấp edges trong prompt, mà tạo đường JOIN có dự báo fan-out, giúp hạn chế lỗi nhân bản dòng.
- **Cổng:** Chỉ làm nếu N4-A tăng correct hoặc giảm JOIN-related errors; test rows count, nullable keys, many-to-many.

### N5. Activation Steering

**N5-A — Global contrastive steering train-only (P3).**

- **Cơ chế:** Trên Qwen3.5-9B pretrained của P5 zero-shot, dùng các cặp chương trình đúng/sai trên train để tính contrastive activation vector tại một số lớp. Thử ít hệ số/lớp được chọn từ train-dev, không sweep vô hạn.
- **Khác hidden-state probe H01:** **Thay hidden activation trong quá trình generation**, không đọc score từ hidden states để xếp hạng candidate.
- **Phân loại:** Không LoRA fine-tuning nhưng có task-specific calibration từ nhãn; ghi là `CALIBRATED`, không phải strict zero-shot.

**N5-B — Conditional steering khi Qwen bất đồng (P3, có điều kiện).**

- **Cơ chế:** Chỉ bật steering cho các câu có candidate disagreement, vote entropy cao hoặc lỗi execution; sử dụng vector đã học từ N5-A để giảm runtime/side effects của steering toàn cục.
- **Cổng:** N5-A phải chứng minh đúng định hướng (giảm lỗi chọn hàm) trước; kiểm ảnh hưởng đến việc Qwen tự sinh business rule, tránh ép model vào xu hướng cố định.
- **Blocker:** vLLM production không mặc định hỗ trợ can thiệp activation; nếu cần backend PyTorch riêng làm vượt 6 giờ, không triển khai.

### N6. Cross-Engine Differential Execution

**N6-A — Deterministic Pandas vs DuckDB shadow execution (P1).**

- **Vị trí:** `src/sqlesc.py` đã có DuckDB và `src/dm.py`/Python chạy Pandas. Chọn subset phép toán có thể dịch ngữ nghĩa chính xác (lọc, COUNT/SUM, đơn giản JOIN); chạy hai implementations và đối chiếu.
- **Cơ chế:** Khi lệch kết quả, ghi vấn đề type/NULL/duplicate; tuyệt đối không mặc định engine nào đúng.
- **Cổng:** Trước hết test định nghĩa arithmetic, NULL, date, decimal và row multiplicity; chỉ trigger cho subset có semantic parity.

**N6-B — Independent program rendering from one Qwen plan (P2).**

- **Cơ chế:** Qwen mô tả plan (không phải compiler luật nghiệp vụ); hai renderers được xây độc lập thành Pandas và DuckDB; đối chiếu cả intermediate row counts và final. Nếu bất đồng, Qwen tái xác nhận intent thay vì bỏ phiếu số.
- **Khác N6-A:** Hai cài đặt độc lập giúp phát hiện bug do transcompiler chung có thể bỏ sót, nhưng tăng chi phí và phạm vi code.
- **Cổng:** Chỉ làm khi N6-A phát hiện khác biệt thật có thể xử lý trên train/dev; không chạy trên mọi câu.

### N7. Semantic Unit & Dimensional Reasoning

**N7-A — Qwen-generated unit certificate trên kết quả (P0/P1).**

- **Vị trí:** Sau chương trình; Qwen gốc sinh metadata `output_unit`, `input_units`, `aggregation`, `period`, `denominator` từ schema/rulebook hiện tại; không có bảng đơn vị sản phẩm hard-code.
- **Cơ chế:** Kiểm bằng đại số thứ nguyên đơn giản. Ví dụ `item × currency/item = currency`, không thể tự kết luận sản phẩm nào có bao nhiêu item/bịch.
- **Cổng:** Đo rate certificate hợp lệ và việc sửa đúng *unit confusion*; không override một đáp án đúng chỉ vì Qwen tự viết sai certificate.

**N7-B — Dimensional AST type-checking (P2).**

- **Cơ chế:** Parse AST của chương trình, truyền unit dimensions qua multiply/divide/add/aggregation; yêu cầu các nhánh cộng phải có cùng dimension và answer type phải khớp mục tiêu Qwen đã khai báo.
- **Khác N7-A:** Kiểm **từng phép toán trung gian**, giúp khoanh vùng lỗi, không chỉ kiểm đầu ra.
- **Cổng:** Chỉ mở rộng nếu N7-A chứng minh phần lớn câu có thể xác định dimension từ ngữ cảnh; không bật các heuristic `unit_check` viết tay cũ.

## 4. Bảng tổng hợp và quyết định sơ bộ

| ID | Biến thể | Mức can thiệp | Zero-shot thuần? | Ưu tiên | Nhận định trước thực nghiệm |
|---|---|---|---|---|---|
| N1-A | Row order invariance | Wrapper data / replay | Có | P1 | Dễ thử, giá trị nếu code phụ thuộc order ngoài ý muốn |
| N1-B | Partition recomposition | Aggregate analyzer / replay | Có | P2 | Chỉ có ý nghĩa trên subset additive và grouping rõ |
| N2-A | Canonical lineage | Instrument data outputs | Có | P2 | Có ích kiểm trace; độ phủ chưa biết |
| N2-B | Raw→canonical lineage, fan-out | Sửa ingest/automap | Có | P3 | Chi phí code cao; chỉ sau N2-A |
| N3-A | Counterexample discrimination | Sinh fixture + Qwen adjudicate | Có | P1/P2 | Giải quyết ứng viên khác nghĩa, tốn thêm generation |
| N3-B | Local AST repair | Synth + constraint checker | Có | P3 | Dễ overfit vào spec sai; chỉ sau N3-A |
| N4-A | Structural join graph | Prompt/schema supplement | Có | **P0/P1** | Sát lỗi schema/JOIN và ít sửa pipeline |
| N4-B | Graph + cardinality path planning | Path selection / SQL | Có | P1/P2 | Đáng thử nếu N4-A tạo gain |
| N5-A | Contrastive activation steering | Runtime hidden states | **Không** (calibrated) | P3 | Rủi ro backend và generalization cao |
| N5-B | Conditional activation steering | Runtime + gating | **Không** (calibrated) | P3 | Chỉ thử nếu N5-A qua gate |
| N6-A | Pandas/DuckDB parity | Shadow execution | Có | P1 | Hợp các lỗi thực thi; phạm vi subset hạn chế |
| N6-B | Independent dual rendering | Hai implementations | Có | P2 | Chi phí cao hơn; chỉ nếu N6-A phát hiện lỗi |
| N7-A | Generated unit certificate | Qwen output metadata + algebra | Có | **P0/P1** | Có tiềm năng trên lỗi đại lượng/đơn vị |
| N7-B | Dimensional AST checker | Expression typeflow | Có | P2 | Khả năng giảm lỗi toán học, phụ thuộc metadata |

**Đánh giá ưu tiên sau đọc repo:** Bắt đầu **N4-A + N7-A** (không cần đổi backbone), cùng N1-A/N6-A làm checks rẻ; N3-A khi nhiều candidates bất đồng; N2-A kiểm định instrumentation; N5-A/B và N2-B/N3-B/ N6-B để sau vì chi phí/rủi ro cao. Đặc biệt, mặc dù N1/N2/N6 giảm false positives, chúng có thể khiến P5 abstain nhiều hơn; penalized score phải là tiêu chí cuối chứ không phải số lỗi phát hiện.

## 5. Kiểm tra đã thực sự chạy ở môi trường này

Đã viết và chạy một **CPU-only synthetic feasibility harness**, không lấy dữ liệu challenge, không load Qwen và **không tích hợp vào P5 runtime**. Kết quả:

| Nhóm | Các kiểm tra cơ chế đã chạy | Kết quả |
|---|---|---|
| N1 | Hoán vị dòng không đổi SUM; partition law | **2/2 PASS** |
| N2 | Source row lineage; join multiplicity | **2/2 PASS** |
| N3 | Sinh phản ví dụ phân biệt SUM/MEAN; constraint hole search | **2/2 PASS** |
| N4 | Join edge từ key overlap; đường nối graph | **2/2 PASS** |
| N5 | Cần Qwen+activation hooks+GPU; không thể kiểm CPU hữu ích | **CHƯA CHẠY** |
| N6 | Pandas/SQLite parity; phát hiện khác biệt integer division | **2/2 PASS** |
| N7 | Kiểm đại số đơn vị đầu ra; dimensional expression typing | **2/2 PASS** |
| **Tổng** | 12 smoke tests của 6 hướng | **12/12 PASS trên dữ liệu tổng hợp** |

**Giải thích quan trọng:** `PASS` ở bảng trên chỉ xác nhận thuật toán mẫu hoạt động trên ví dụ giả định, **không chứng minh khả năng tích hợp P5 hoặc tăng score D1**. N6 dùng SQLite trong smoke do môi trường này chưa có DuckDB; implementation chính thức phải dùng DuckDB của repo. N5 không được tính là `FAIL` hay `PASS`.

Artifacts đi kèm: `new7_cpu_feasibility.py`, `new7_cpu_feasibility_results.json`.

## 6. Cách tổ chức chạy thật trên Kaggle mà không spam GPU

### Giai đoạn 0 — Freeze và audit (CPU)

- Freeze `configs/flow_p5_zs_strict.json`, giữ nguyên weights base Qwen và tất cả công cụ trung lập. Không giữ cache rules do model tuned viết, `hide_api=industry`, `lrl_regen=1`; bảo đảm Qwen base tự sinh luật ở mọi biến thể.
- Mốc đã đo **0.754** (2 seeds) được tái sử dụng nếu commit, dataset và evaluation protocol khớp. Nếu code/data thay đổi đáng kể, chạy lại B00.
- Chỉ khai thác các logs train/dev hợp lệ để nghiên cứu lỗi; dùng D1 đánh giá công bằng theo policy hiện hành, không dùng D1 làm tập tinh chỉnh điều kiện/alpha.
- Audit `automap.py`/`hints.py`/`precheck.py` về heuristics viết tay; không mở cờ logic nghiệp vụ có sẵn chỉ để tăng score.

### Giai đoạn 1 — CPU-first: 5 biến thể ít GPU

N1-A, N2-A, N4-A, N6-A, N7-A. Làm offline replay trên các candidate programs đã lưu, hoặc phần train/dev dữ liệu được cấp phép. Nếu chỉ có đáp án cuối mà không có trace/program hoặc raw dataset, **không thể replay tương đương**: phải thu logs P5 ở run mới.

Dự toán ban đầu: mỗi feasibility/replay **0.25–2 giờ CPU wall-time**, phụ thuộc log/canonical data; đây là **ước tính**, không phải thời gian đo trên repo. Không cần cấp cả 4 L4 cho những bài toán chỉ dùng Pandas.

### Giai đoạn 2 — 200 câu train/dev hợp lệ, 1 seed để sàng lọc

Tập trung N4-A, N7-A, N1-A (và N3-A nếu có disagreement); chạy các biến thể trên cùng `question_id` và seed, có `matched rule source` và `candidate pool` càng cố định càng tốt. Chỉ những biến thể dùng prompt/Qwen calls mới cần GPU. Có thể chia độc lập mỗi L4 một variant worker khi từng model vừa VRAM.

Dự toán per experiment end-to-end trên 200 câu: **0.3–1.0 giờ wall-clock với 4×L4**, ước tính ngoại suy từ B00 ~110–113 phút/1,000 câu và thêm overhead rule generation; **cần pilot 20–30 câu để xác minh**. N3 thường phát sinh thêm model calls; N5 có backend/cost riêng và chưa định giờ.

### Giai đoạn 3 — 1–3 finalists, full D1 + held-out, 2 seeds

- Promote nếu paired ΔPA thực sự ổn định, không phải score của samples khác seeds/cases. Báo `c/w/a`, số lỗi khôi phục, số câu đúng bị mất, `PA_pub_ind`, `PA_pub_noind`, p95 latency, overall wall-clock, peak VRAM.
- Mục tiêu so sánh là **B00 strict mean 0.754**, không phải P5 tuned 0.952. Báo thêm khoảng cách đến 0.952 chỉ khi cùng D1 protocol, không dùng để chọn winners zero-shot.
- Cổng chọn chỉ mở khi không thấy leakage, không hard-coded domain rules, tổng model params <15B, và full inference vẫn đáp ứng 6 giờ.
- Với mức ±0.001 seed variability đã thấy ở B00, chỉ coi gain đáng tin nếu **paired confidence interval** không âm và không có systematic regression ở câu ngành/unseen; một gain +0.002 không đủ kết luận chắc chắn.

### Nhóm dừng sớm (tránh hao GPU)

- **N1**: ít hơn 5–10% chương trình được chứng minh an toàn cho phép biến đổi hoặc lỗi phát hiện không chuyển thành score gain.
- **N2**: lineage không bảo toàn qua canonicalization, nguồn row IDs mất hoặc chi phí truy vết cao.
- **N3**: phản ví dụ chỉ chứng minh chương trình khác nhau nhưng không thể phân giải ý định, gây nhiều LLM calls mà không giảm sai.
- **N4**: phần lớn câu đã được `automap` đưa về canonical C không cần JOIN; graph thêm prompt noise hoặc đề xuất JOIN sai.
- **N5**: backend vLLM không cho activation hooks đủ nhanh, steering làm rule generation kém ổn định hoặc vượt 6 giờ.
- **N6**: SQL↔Pandas subset quá nhỏ hoặc parity false alarms do NULL/dtype/datetime semantics.
- **N7**: certificates do cùng Qwen sinh mắc lỗi tương quan; checker không có tín hiệu độc lập ngoài việc tự xác nhận mình.

## 7. Hạ tầng và những việc bị chặn hiện tại

Repository đã kết nối qua GitHub plugin (read-only trong bước đánh giá), nhưng môi trường thực thi cục bộ **không có Kaggle CLI, Kaggle API credentials, NVIDIA GPU, hoặc checkout đầy đủ các bộ dữ liệu `data/raw`**. Plugin directory không tìm thấy tích hợp Kaggle. Do đó **chưa thể khởi chạy 14 variants trên D1**, và báo cáo này **không được diễn giải là đã thực thi Kaggle/P5**.

Repo có `automation/launch_many.py`, `automation/quota.py`, `training/soup_job.py`, các kernel và config zero-shot P5. Khi access Kaggle sẵn sàng, sử dụng chúng trước, không viết lại hệ orchestration. Chạy 4×L4 per session hoặc 1 variant/GPU tùy memory; kiểm quota, 12-hour session time cap, ghi checkpoints và bảo mật token.

**Nguồn tham khảo code trực tiếp:**
- [`configs/flow_p5_final.json`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/configs/flow_p5_final.json)
- [`configs/flow_p5_zs_strict.json`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/configs/flow_p5_zs_strict.json)
- [`Process/scores_p5zs.csv`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/Process/scores_p5zs.csv)
- [`src/automap.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/automap.py)
- [`src/perturb.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/perturb.py)
- [`src/sqlesc.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/sqlesc.py)
- [`src/units.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/units.py)
- [`src/retrieval.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/retrieval.py)
- [`src/hybrid.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/hybrid.py)
- [`src/judge.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/judge.py)
- [`src/precheck.py`](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/src/precheck.py)

## 8. Quyết định cuối và thứ tự thực thi

**Ưu tiên ngay (do khớp P5 và ít tốn resources):** N4-A → N7-A → N1-A, N6-A có thể chạy song song trên CPU/train-dev. N3-A chỉ gọi khi có candidate disagreement; N2-A là bước nghiên cứu instrumentation riêng. Không mở ngay N2-B, N3-B, N4-B, N5-A/B, N6-B, N7-B, N1-B nếu các biến thể đơn giản không chứng minh được tiềm năng.

**Thành công không có nghĩa số CPU PASS cao:** chỉ khi full P5 zero-shot đánh giá trên D1/held-out có **paired ΔPA dương, ổn định**, không dùng luật viết tay và không làm tăng thời gian suy luận vượt giới hạn, mới kết luận một giải pháp phù hợp để triển khai. Đến thời điểm lập báo cáo, **chưa có bằng chứng rằng bất kỳ N1–N7 nâng score DataMate**.
