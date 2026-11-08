# 08 — Kế hoạch tổng thể: hybrid framework, mục tiêu public = 1.0, thời gian ngắn nhất

*Lập 08/11/2026 theo yêu cầu user. Đây là kế hoạch chính cho khung chat kế tiếp; thay thế §7 (E0–E5) của
`docs/06_HYBRID_FRAMEWORK.md` (thiết kế module ở đó vẫn đúng). Số citation: Semantic Scholar; số stars: GitHub — cùng tra
ngày 08/11/2026, thay đổi theo thời gian.*

## 0. Tóm tắt

1. **Hệ đích:** Agent (Qwen3.5-9B, fine-tune LoRA) sinh code cho **mọi** câu; luật tổng quát chỉ chuẩn bị ngữ cảnh: dựng schema
   (`automap`), làm sạch câu, gói gợi ý (kỳ thời gian, thực thể khớp data, kiểu đáp án, thuật ngữ rulebook), truy hồi ví dụ;
   code tính toán (toolkit `dm` / `industry`, pandas / DuckDB SQL dự phòng), kiểm tra trước khi chạy, Agent đọc kết quả, chọn
   theo phiếu + điểm tin cậy hiệu chỉnh, sinh lại hoặc abstain.
2. **Đòn bẩy chính, theo thứ tự lợi / chi phí:** (a) schema từ `automap` (bỏ lỗi ánh xạ — nguồn lỗi lớn nhất của v28);
   (b) gói gợi ý + ví dụ đã kiểm; (c) **fine-tune LoRA** trên 2681 cặp (câu train → chương trình đã kiểm) + paraphrase +
   dữ liệu tổng hợp "chương trình trước, câu sau" (OmniSQL); (d) chọn / kiểm / tin cậy (self-consistency thích ứng, LEVER);
   (e) vòng sửa lỗi tổng quát trên public.
3. **Tìm cấu hình huấn luyện tốt nhất bằng thí nghiệm song song:** mỗi phiên Kaggle 4×L4 chạy **4 nhánh QLoRA độc lập (mỗi
   GPU một nhánh)** khác nhau về dữ liệu / mô hình thầy, cùng đánh giá trên bộ dev nhanh; loại dần (successive halving);
   nhánh thắng huấn luyện đầy đủ trên 4 GPU.
4. **Thời gian:** 8 ngày tới hạn 16/11 23:59 GMT+7; quota GPU ≈ 75 GPU-giờ tuần này + 90 từ 14/11. Lộ trình §6.
5. **Trung thực về mục tiêu:** public = 1.0 là mục tiêu của vòng phát triển (public dùng để phân tích lỗi, **không** để train
   hay đưa vào prompt). Mọi sửa phải là cơ chế chung và **không làm giảm held-out**; điểm private mới là đích cuối.
   Ước lượng (chưa đo): ~0.93–0.96 sau bước (a)+(b), ~0.97–0.99 sau (c), 1.0 cần vòng sửa lỗi (e) — không bảo đảm.

## 1. Mục tiêu, thước đo, ràng buộc

| Thước đo | Tập | Vai trò | Mục tiêu |
|---|---|---|---|
| PA public gốc | 1000 câu, so với `data/raw/public_gt.jsonl` (`src/score.py`) | dev — được phân tích lỗi | **1.0** |
| PA paraphrase public k=0 | 692 câu | dev — được phân tích lỗi | tăng đều |
| PA paraphrase public k=1,2 | 1373 câu | **held-out** — không phân tích từng câu | không giảm khi sửa; > 0.602 (v34) |
| PA train held-out theo DN | 6 DN train (~600 câu gốc + paraphrase), loại khỏi dữ liệu huấn luyện và ngân hàng ví dụ khi đo | **held-out** | không giảm |
| PA DN tổng hợp ngành mới | DN do ta sinh (§4.2 D4), nếu kịp | held-out (gần private: ngành mới) | — |
| Bền vững | đổi tên cột, README trắng, câu nhiễu (`tests/eval_all.sh`) | kiểm phần luật / schema | giữ |
| Thời gian | s / câu / lần chạy; tổng cho 1000 câu | ≤ 6 h trên 2×RTX 5090 cho private | ≤ 2 h / 1000 câu (dư cho private lớn hơn) |
| Tất định | 2 lần chạy cùng cấu hình | luật BTC: \|ΔPA\| ≤ 0.03 | ≤ 0.01 |

Ràng buộc (user + BTC, không hỏi lại): mọi câu có code do mô hình sinh; luật chỉ là ngữ cảnh / gợi ý / kiểm tra; không vá
theo id DN / câu; không dùng public / private test (kể cả paraphrase của chúng) để train, không đưa đáp án public vào
`src/` hay prompt; tổng tham số lúc inference ≤ 15B (Qwen3.5-9B = 9.65B → còn ~5B cho mô hình phụ); mô hình (kể cả thầy)
có trọng số public trước 01/10/2026; offline; ghi đáp án dần; dataset ngoài theo `docs/07_DATASETS.md`; nộp kèm **code
train** (luật BTC: solution gồm code train + inference + dữ liệu).

## 2. Tham khảo: paper / repo uy tín → bài học → áp dụng

### 2.1 Agent sinh code truy vấn, đa tác tử, phân tích dữ liệu

| Công trình | Uy tín | Ý chính | Áp dụng |
|---|---|---|---|
| ReAct (ICLR 2023) | 13,667 cit. | suy luận xen hành động / quan sát | vòng sinh code → chạy → đọc kết quả → sửa |
| Program of Thoughts (TMLR 2023), PAL (ICML 2023) | 1,587 / 911 cit. | LLM viết chương trình, **trình thông dịch tính**, LLM không tự tính | nguyên tắc cốt lõi: số liệu chỉ từ code; định dạng đáp án bằng code |
| Binder (ICLR 2023) | 368 cit., 324★ | ngôn ngữ ký hiệu + API gọi mô hình | toolkit `dm` = API nghiệp vụ trong ngôn ngữ chương trình |
| Teaching LLMs to Self-Debug (ICLR 2024) | 1,519 cit. | sửa code theo lỗi thực thi + tự giải thích | vòng sửa (`--repairs`), thông điệp lỗi giàu (LookupError kèm tên gần nhất) |
| DIN-SQL (NeurIPS 2023) | 998 cit. | tách bước: liên kết schema → phân loại độ khó → sinh → tự sửa | gói gợi ý (liên kết thực thể / kỳ) trước khi sinh |
| MAC-SQL (COLING 2025) | 350 cit. | selector (lọc schema) / decomposer / refiner | lọc schema theo câu để prompt ngắn |
| CHESS (2024) | 278 cit., 288★ | truy hồi giá trị / thực thể từ data, lọc schema | liên kết thực thể với giá trị thật (đã có nhiều tầng) |
| CHASE-SQL (ICLR 2025) | 244 cit. | nhiều đường sinh đa dạng + mô hình chọn | đa dạng hoá prompt (trực tiếp / kế hoạch trước / theo gợi ý) |
| XiYan-SQL | 93 cit., 1,019★ | lọc schema → nhiều bộ sinh (fine-tune khác nhau) → bộ chọn | nhiều nhánh LoRA có thể làm nhiều bộ sinh |
| ReFoRCE (Spider 2.0 top) | 42 cit., 143★ | nén schema, thăm dò cột / giá trị, đồng thuận, tự sửa | tool thăm dò giá trị (§3 M8) |
| DS-STAR (Google, 2025) | 13 cit. (mới), top DABStep | mô tả mọi file → Planner / Coder / Verifier lặp | **data card** do Agent viết; Verifier đọc kết quả |
| Data Interpreter (ACL), MetaGPT | 322 cit., 71,215★ | kế hoạch dạng đồ thị, kiểm từng bước | câu nhiều bước (tăng trưởng, so sánh kỳ) |
| TaskWeaver (Microsoft) | 115 cit., 6,160★ | agent code-first + plugin | plugin = hàm toolkit, có mô tả chữ ký |
| Vanna | 23,801★ | RAG trên DDL + tài liệu + cặp câu-SQL đã kiểm | ngân hàng ví dụ đã kiểm + tài liệu nghiệp vụ trong prompt |
| WrenAI | 18,106★ | **semantic layer** (định nghĩa chỉ số) giữa LLM và dữ liệu | toolkit = semantic layer; rulebook → định nghĩa chỉ số |
| PandasAI | 23,938★ | LLM sinh pandas trên DataFrame, chạy, thử lại | đường dự phòng pandas cho ngành chưa có hàm |
| DB-GPT / DB-GPT-Hub | 20,273★ / 2,015★ | khung + fine-tune text-to-SQL (LoRA) | công thức SFT cho mô hình mở |

### 2.2 Ngữ cảnh và ví dụ

| Công trình | Uy tín | Áp dụng |
|---|---|---|
| DAIL-SQL (PVLDB 2024) | 868 cit., 642★ | chọn ví dụ theo **khung câu đã che thực thể / số** + tổ chức ví dụ gọn → truy hồi ví dụ theo khung câu (fewshot đã che số; thêm che thực thể / kỳ) |
| What Makes Good In-Context Examples (DeeLIO 2022) | 1,972 cit. | ví dụ gần nghĩa nhất > ngẫu nhiên |
| RESDSQL (AAAI 2023), CodeS (SIGMOD 2024) | 448 / 385 cit. | tách liên kết schema khỏi sinh khung; mô hình mở fine-tune cho text-to-SQL |

### 2.3 Chọn, kiểm, tin cậy, tiết kiệm mẫu

| Công trình | Uy tín | Áp dụng |
|---|---|---|
| Self-Consistency (ICLR 2023) | 8,228 cit. | bỏ phiếu theo kết quả thực thi (đã có) |
| Adaptive-Consistency (EMNLP 2023), Early-stopping SC (ICLR 2024) | 193 / 164 cit. | **dừng sớm khi các mẫu đầu đồng ý** → giảm thời gian 2–3× ở câu dễ |
| LEVER (ICML 2023) | 415 cit. | **bộ kiểm học được** trên (câu, chương trình, kết quả) để xếp hạng → tín hiệu tin cậy |
| CodeT (ICLR 2023) | 683 cit. | sinh phép thử để chọn code → phép kiểm bất biến (tổng ≥ thành phần, tỉ lệ ∈ [0,100]) |
| Language Models (Mostly) Know What They Know; Just Ask for Calibration (EMNLP 2023) | 2,320 / 1,091 cit. | xác suất / tự đánh giá của mô hình là tín hiệu, cần hiệu chỉnh |
| TrustSQL; selective prediction text-to-SQL (2026) | 26 cit.; mới | chấm có phạt như PA; self-consistency một mình không đủ → kết hợp tín hiệu |

### 2.4 Huấn luyện / chưng cất

| Công trình / công cụ | Uy tín | Áp dụng |
|---|---|---|
| LoRA (ICLR 2022), QLoRA (NeurIPS 2023) | 25,528 / 6,318 cit. | fine-tune 9B trên L4 (QLoRA 1 GPU / nhánh) |
| STaR (NeurIPS 2022), ReST-EM (TMLR 2024) | 1,241 / 434 cit. | **tự huấn luyện có lọc**: mô hình sinh nhiều chương trình, giữ cái chạy ra đúng, huấn luyện lại, lặp |
| Distilling Step-by-Step (ACL Findings 2023) | 1,259 cit. | học trò nhỏ học cả lời giải thích của thầy → ít dữ liệu hơn |
| OmniSQL (PVLDB 2025) | 149 cit., 457★ | tổng hợp dữ liệu **chương trình trước → câu sau → CoT**, kiểm thực thi; 7B fine-tune ngang mô hình lớn |
| Arctic-Text2SQL-R1 (ACL Findings 2026) | 61 cit., 293★ (ArcticTraining) | RL (GRPO) chỉ thưởng đúng khi thực thi; 7B 68.5 BIRD test |
| SQL-R1 (NeurIPS 2025), Reasoning-SQL | 107 / 96 cit. | RL cho sinh truy vấn — chỉ làm nếu dư thời gian |
| DeepSeekMath (GRPO) | 10,833 cit. | thuật toán GRPO |
| LlamaFactory (ACL 2024), Unsloth, TRL, verl, OpenRLHF | 75,935★ / 79,188★ / 19,690★ / 24,163★ / 10,145★ | công cụ SFT / RL; ưu tiên TRL + PEFT (ít phụ thuộc, dễ đóng gói offline) hoặc Unsloth (nhanh, ít VRAM) |
| vLLM / PagedAttention (SOSP 2023) | 9,947 cit., 96,542★ | suy luận, prefix cache, LoRA |

### 2.5 Tiếng Việt

PhoBERT (Findings EMNLP 2020; 534 cit., 810★) — mô hình phụ khả dĩ (phân loại ý định / bộ kiểm, ~135M tham số);
ViText2SQL (Findings EMNLP 2020; 68 cit.) — dữ liệu câu hỏi truy vấn tiếng Việt.

### 2.6 Tám nguyên tắc rút ra

1. Số liệu chỉ từ code; mô hình chọn / kiểm, không chép lại số (PoT / PAL).
2. Ngữ cảnh dữ liệu giàu nhưng đã lọc: schema chuẩn + data card + giá trị thật của thực thể (DS-STAR, CHESS, MAC-SQL).
3. Ngữ nghĩa nghiệp vụ trong lớp chỉ số đã kiểm (semantic layer = toolkit) (WrenAI).
4. Ví dụ đã kiểm, chọn theo khung câu (DAIL-SQL, Vanna).
5. Nhiều ứng viên đa dạng + chọn; dừng sớm khi đồng thuận (CHASE-SQL, Adaptive-Consistency).
6. Tự sửa theo phản hồi thực thi (Self-Debug, ReFoRCE).
7. Tin cậy = kết hợp nhiều tín hiệu, hiệu chỉnh trên dev, ngưỡng tối đa PA (LEVER, TrustSQL).
8. Mô hình nhỏ mạnh lên nhờ **dữ liệu tổng hợp có kiểm thực thi + fine-tune / tự huấn luyện** (OmniSQL, STaR / ReST, Arctic-R1).

## 3. Kiến trúc đích (module → ai làm → hiện có → việc cần làm)

M = mô hình, C = code / luật logic (được phép).

| # | Module | Ai | Hiện có | Việc cần làm |
|---|---|---|---|---|
| M1 | Đọc data → bảng thô + card | C | `ingest.py` | giữ |
| M2 | Schema chuẩn C | C (+M dự phòng) | `automap.py`; LLM ánh xạ + `canon.secondary_checks` + `consensus.py` | dùng automap mặc định; LLM ánh xạ khi cảnh báo nặng / thiếu bảng |
| M3 | Diễn giải rulebook | M + C | `changed_terms`, định nghĩa trong toolkit | M đọc rulebook của split → bảng định nghĩa (JSON) so với mặc định; thuật ngữ mới / đổi → đưa nổi bật vào prompt + cho phép pandas ghi đè |
| M4 | Data card | M | — | mỗi DN một lần: mô tả ≤ 300 token (cột có / thiếu, quy ước lạ), cache, đặt trong phần prompt chung |
| M5 | Làm sạch câu + gói gợi ý | C | `qnorm.py`; `qparse.parse_period`, `named_*`, liên kết thực thể nhiều tầng | `src/hints.py`: kỳ thời gian quy ra ngày, thực thể đã khớp tên thật, kiểu đáp án dự kiến, thuật ngữ rulebook, dấu hiệu "dữ liệu không có" — **không** đưa mẫu / chương trình của `qparse` |
| M6 | Truy hồi ví dụ | C | `fewshot.py` (2681 cặp đã kiểm, n-gram ký tự, che số) | che thêm thực thể / kỳ (khung câu, DAIL-SQL); thêm ví dụ tổng hợp; loại DN held-out khi đo |
| M7 | Sinh chương trình | M | `agent.py` (`--k`, `--kc`, `--shots`, `--pipeline`) | prompt có gợi ý + data card; 2–3 kiểu prompt đa dạng; mô hình LoRA; ngôn ngữ dự phòng pandas / DuckDB SQL cho ngành chưa có hàm |
| M8 | Tool (chỉ khi cần) | M gọi, C thực hiện | LookupError gợi ý tên | `lookup_values`, `describe(table)`, `sample_rows` nếu phân tích lỗi cho thấy thiếu thông tin giá trị |
| M9 | Kiểm tra trước khi chạy | C | grounding, `answer_kind_ok` | `src/precheck.py`: AST, hàm / tham số đúng chữ ký, thực thể và kỳ khớp gợi ý, kiểu trả về; vi phạm → Agent sửa |
| M10 | Chạy | C | sandbox trong `pipeline.py` | giữ (timeout, read-only) |
| M11 | Đọc kết quả | M | — | chỉ khi nghi ngờ (bất đồng, 0, rỗng, âm, bất thường, lỗi): OK / SỬA / KHÔNG TRẢ LỜI ĐƯỢC |
| M12 | Chọn + tin cậy | C + M | `vote.py --min-conf`, `--escalate` | dừng sớm khi đồng thuận; điểm tin cậy hiệu chỉnh (logistic) từ: tỉ lệ phiếu, số đáp án khác nhau, cờ precheck, lỗi chạy, log-prob, tự đánh giá M11, (tuỳ chọn) chương trình luật đồng ý hay không — **chỉ là đặc trưng, không chọn đáp án**; dưới θ₁ sinh lại, dưới θ₂ abstain |
| M13 | Định dạng + ghi | C | `fmt_answer`, `merge_sub.py` | giữ (append + flush, gộp mỗi 20 s) |

Chương trình do `qparse` dựng **không bao giờ** được đưa vào prompt hay chọn làm đáp án (quyết định user 08/11).

## 4. Huấn luyện / chưng cất

### 4.1 Nguồn dữ liệu (được phép / cấm)

| Mã | Dữ liệu | Quy mô | Nhãn từ đâu | Ghi chú |
|---|---|---|---|---|
| D1 | Câu train → chương trình đã kiểm | 2681 (+ unanswerable) | `src/refs/answer_examples.jsonl` (chương trình chạy ra đúng đáp án train) | sẵn có; bỏ 6 DN held-out |
| D2 | Paraphrase câu train | 5689 (`<acc>/zc-para/para_train.jsonl`) | chương trình của câu gốc, **giữ khi chạy lại ra cùng đáp án** | kiểm đổi nghĩa bằng vòng lặp ngược (D3 bước 4) |
| D3 | Tổng hợp "chương trình trước, câu sau" trên DN train | 10–30k | toolkit chạy chương trình → đáp án | §4.2 |
| D4 | DN tổng hợp ngành mới (ta tự sinh data) | 3–6 DN, 2–5k câu | đáp án tính từ bộ sinh / toolkit / pandas đã kiểm | dạy đường dự phòng pandas; gần private (ngành mới) |
| D5 | Lời giải thích (CoT) của thầy cho D1–D4 | như trên | thầy sinh, giữ khi chương trình kèm theo chạy đúng | Distilling step-by-step |
| D6 | Dataset ngoài (`docs/07_DATASETS.md` §4, không cần xin phép) | 5–20 % hỗn hợp | nhãn của dataset | ViText2SQL (tiếng Việt), TableBench-TableInstruct (Python), BIRD / Spider train, SynSQL-2.5M (mẫu nhỏ) — chỉ giữ khi nhánh thắng |
| D7 | Tự huấn luyện (ReST) | lấy mẫu K chương trình của chính mô hình trên D1–D4 | giữ chương trình chạy ra đúng | vòng 2 nếu còn thời gian |
| **Cấm** | câu / data / paraphrase của public test và private test | — | — | luật BTC §6 |

Dùng logic luật (bộ sinh chương trình, toolkit) **để tạo dữ liệu huấn luyện offline** là hard code logic (được phép); chúng
không tham gia trả lời lúc chạy. Nếu user không đồng ý điểm này: chỉ dùng nhãn từ đáp án train + chương trình do thầy /
mô hình sinh có kiểm thực thi (D1 vẫn hợp lệ vì chương trình được giữ nhờ khớp đáp án train).

### 4.2 Dây chuyền tổng hợp D3 (kiểu OmniSQL, chạy trên CPU Kaggle + GPU cho thầy)

1. **Lấy mẫu chương trình** từ ngữ pháp toolkit (hàm × tham số × kỳ × thực thể thật của DN train), phân bố theo tỉ lệ kiểu
   câu của train; loại kết quả suy biến (rỗng, toàn 0, lỗi).
2. **Chạy** → đáp án (CPU; song song trên 5 phiên CPU / acc).
3. **Thầy viết câu tiếng Việt** cho chương trình theo nhiều văn phong (trang trọng, chat, viết tắt, không dấu, từ địa phương,
   hỏi gián tiếp), kèm data card của DN để dùng tên thật.
4. **Lọc vòng lặp ngược:** mô hình (9B hiện tại hoặc thầy) đọc câu → viết chương trình → chạy; giữ cặp khi đáp án trùng
   (bỏ câu đổi nghĩa). Ghi tỉ lệ giữ của từng thầy (chỉ số chất lượng thầy).
5. (tuỳ) thầy viết lời giải ngắn (D5).

### 4.3 Mô hình thầy ứng viên (kiểm model card: ngày phát hành < 01/10/2026, giấy phép; kiểm có trên Kaggle Models)

| Thầy | Kích thước | Giấy phép / phát hành (nguồn thứ ba, cần kiểm) | Chạy trên 4×L4 |
|---|---|---|---|
| Qwen3.5-9B (chính nó, tự chưng cất) | 9.65B | Apache-2.0, 02/2026 | đã có |
| Qwen3.5-27B | 27B đặc | Apache-2.0, 02/2026 | FP8, TP 2 × 2 bản |
| Qwen3.5-35B-A3B | 35B MoE (~3B hoạt động) | Apache-2.0, 02/2026 | FP8, TP 2; nhanh |
| Qwen3.5-122B-A10B | 122B MoE | Apache-2.0, 02/2026 | INT4 (~61 GB), TP 4 — cần bản INT4 trên Kaggle |
| Gemma 4 31B / 26B MoE | 31B / 26B | Apache-2.0, 04/2026 | FP8, TP 2 (Google đăng trên Kaggle Models) |
| gpt-oss-20b / 120b | 21B / 117B MoE | Apache-2.0, 08/2025 | 20b: 1 GPU; 120b: cần kiểm kernel MXFP4 trên L4 |

Container dev chỉ có ~24 GB đĩa → không tải trọng số lớn về container; chỉ dùng trọng số đã có trên Kaggle (Models /
Datasets).

### 4.4 Học trò và phương pháp

- **Học trò:** Qwen3.5-9B (mô hình chạy thật). QLoRA (4-bit, r 16–64) cho nhánh thử trên 1 GPU; bản cuối LoRA bf16 / QLoRA
  DDP 4 GPU. Prompt huấn luyện **ngắn** (schema đã lọc theo câu + gợi ý + ≤ 2 ví dụ, ~1.5–2k token) để huấn luyện nhanh và
  suy luận nhanh.
- **Thứ tự:** SFT (D1 + D2 + D3 [+ D4, D5, D6]) → ReST một vòng (D7) nếu tăng held-out → GRPO thưởng thực thi chỉ khi dư
  thời gian (khả năng thấp trước 16/11).
- **Mô hình phụ (tuỳ, ≤ ~5B còn lại):** chỉ khi đo được lợi: bộ kiểm kiểu LEVER (Qwen3.5-0.8B / 2B hoặc PhoBERT fine-tune
  trên (câu, chương trình, kết quả) → đúng / sai) làm một tín hiệu tin cậy. Mặc định dùng chính 9B làm bộ kiểm (không thêm
  tham số).

### 4.5 Tìm cấu hình tốt nhất bằng nhánh song song (tiết kiệm quota)

- **Một phiên 4×L4 = 4 nhánh:** mỗi GPU một nhánh QLoRA độc lập (khác dữ liệu / thầy / siêu tham số), cùng số bước; sau
  huấn luyện mỗi GPU gộp LoRA và chạy đánh giá nhanh của nhánh mình. Chi phí quota như một lần chạy.
- **Đánh giá nhanh (xếp hạng nhánh):** 400 câu cố định = public 150 (phân tầng theo DN / kiểu) + k=0 150 + train held-out
  100; tham lam (K = 1) để xếp hạng; K = 4 cho 2 nhánh đứng đầu.
- **Vòng 1 (dữ liệu):** A1 = D1; A2 = D1 + D2; A3 = D1 + D2 + D3 (câu từ chính 9B); A4 = A3 + D6 (10 %).
- **Vòng 2 (thầy):** cùng hỗn hợp tốt nhất, D3 do 3–4 thầy khác nhau viết câu (§4.3) + một nhánh trộn các thầy.
- **Vòng 3 (siêu tham số):** r ∈ {16, 64}, epoch ∈ {1, 2}, có / không D5 (CoT).
- **Chốt:** nhánh thắng (held-out cao nhất; hoà → public / k=0) → huấn luyện đầy đủ DDP 4 GPU → đánh giá đầy đủ.
- **Ghi lại** mọi nhánh: dữ liệu, thầy, siêu tham số, thời gian, điểm (bảng trong `EXPERIMENT_LOG.md`).

### 4.6 Đóng gói cho Docker

Gộp LoRA vào trọng số (merge) rồi lượng tử FP8 như hiện nay (tránh rủi ro vLLM chưa hỗ trợ LoRA cho kiến trúc Qwen3.5);
đăng trọng số / adapter lên Kaggle (dataset private của acc) để `docker/fetch_assets.sh` tải; kiểm sha256. Code huấn luyện
+ dữ liệu tổng hợp + cấu hình nằm trong repo (`training/`) vì BTC yêu cầu nộp code train.

## 5. Đánh giá và ghi log

- Bộ dev / held-out cố định ở §1; tạo file danh sách id một lần (`tests/splits/*.txt`), không đổi giữa các lần đo.
- Mỗi câu lưu: gợi ý, ví dụ, mọi chương trình, kết quả, cờ precheck, phiếu, tin cậy, thời gian → **chấm lại / đổi ngưỡng /
  đổi cách chọn offline trong vài giây** (`src/replay.py`, `src/vote.py`), không chạy lại GPU.
- Cổng giữ thay đổi: held-out (k=1,2 + train held-out) không giảm, public + k=0 tăng hoặc giữ, thời gian trong ngân sách,
  test CPU + preflight qua.

## 6. Lộ trình (08/11 → 16/11) và phân bổ tài nguyên

**Song song ba luồng:** (A) code suy luận trên container (CPU); (B) dữ liệu trên container + phiên CPU Kaggle (5 / acc);
(C) GPU Kaggle (2 phiên / acc; dồn đủ một acc rồi sang acc khác). Quota tuần này: quynhtran35 30, phatle1578 30, yuan1103
~15 GPU-giờ; reset 14/11 07:00 GMT+7 (+90). Một phiên 4×L4 ≈ 4 GPU-giờ / giờ.

| Ngày | Luồng A (suy luận) | Luồng B (dữ liệu) | Luồng C (GPU) | Cổng quyết định |
|---|---|---|---|---|
| 08–09/11 | luồng hybrid v1: bỏ stage 1 trả lời thẳng; `hints.py`, `precheck.py`, prompt có gợi ý; cờ bật / tắt; test + preflight | D1 / D2 xuất định dạng SFT; tách 6 DN held-out; bộ lấy mẫu chương trình D3 + chạy | **E0**: `agent.py --prefer-map fmt:automap --qw 0`, không `--confirm` trên 400 câu nhanh + public đủ (yuan1103, ~8 GPU-h) | có mốc E0 |
| 10/11 | data card (M4), đọc kết quả (M11), log đặc trưng tin cậy | D3 phần câu do 9B viết + lọc ngược; tải D6 | **E1** (hybrid v1) trên bộ nhanh; **SFT vòng 1** (4 nhánh, quynhtran35); dò thầy: mỗi thầy viết câu cho 1k chương trình + lọc ngược (phatle1578) | chọn hỗn hợp dữ liệu; xếp hạng thầy theo tỉ lệ giữ |
| 11/11 | sửa lỗi tổng quát từ E1 (phân loại §7) | D3 đầy đủ bằng thầy tốt nhất (10–30k) | **SFT vòng 2** (thầy); sinh D3 đầy đủ | chọn thầy |
| 12/11 | tích hợp mô hình LoRA vào luồng; hiệu chỉnh tin cậy trên dev | (tuỳ) D4 ngành mới | **SFT đầy đủ** (DDP 4 GPU) + đánh giá đầy đủ (public, k=0, held-out) | LoRA > nền trên held-out? không → giữ nền |
| 13/11 | vòng sửa lỗi 2; ngưỡng abstain; đo thời gian; dừng sớm khi đồng thuận | D7 (ReST) nếu còn quota | (tuỳ) ReST vòng 1 | đóng băng tính năng |
| 14/11 | — | — | quota mới: **đánh giá cuối** (public ×2 lần → tất định, k=1,2, train held-out, bền vững, thời gian) | chọn bản nộp |
| 15/11 | đóng băng code; Docker (trọng số đã gộp, `fetch_assets.sh`, `predict.sh`); `submissions/vNN` | — | dự phòng | user build + chạy thử |
| 16/11 | sửa sự cố Docker nếu có | — | — | **user nộp trước 23:59 GMT+7** |

**Dự phòng:** fine-tune không thắng nền trên held-out → nộp hybrid nền (không LoRA). Thầy lớn không có trên Kaggle → tự
chưng cất bằng 9B (ReST) + Gemma 4 / gpt-oss-20b nếu có. Thiếu quota → bỏ vòng 3 và D4.

**Tiết kiệm thời gian:** cache ingest / schema / data card; đo lại offline từ log; nhánh song song mỗi GPU; đánh giá nhanh
400 câu thay vì đầy đủ; một đợt preflight cho nhiều sửa; dừng sớm self-consistency ở suy luận.

## 7. Vòng sửa lỗi để tiến tới public = 1.0

1. Sau mỗi lần đo: lấy câu sai / abstain trên public + k=0 + train (không xem held-out), phân loại gốc lỗi theo
   `docs/06_HYBRID_FRAMEWORK.md` §5 (schema, ý định, kỳ, thực thể, toolkit, thực thi, định dạng, abstain, nhãn / đổi nghĩa).
2. Sửa theo nhóm, ưu tiên: toolkit / định nghĩa đã kiểm → precheck / kiểm kết quả → gợi ý → ví dụ tổng hợp / dữ liệu
   huấn luyện bao phủ nhóm đó (trên DN train / DN tổng hợp, **không** câu public) → hướng dẫn prompt (ít) → tool.
3. Thêm test hồi quy từ train / tổng hợp cho mỗi nhóm; chạy bộ nhanh; giữ khi held-out không giảm.
4. Dừng khi câu sai còn lại chỉ sửa được bằng vá riêng câu đó, hoặc là lỗi nhãn / câu mơ hồ — ghi rõ trong log.

## 8. Rủi ro

| Rủi ro | Cách giảm |
|---|---|
| Quota GPU không đủ | nhánh 4-trong-1, đánh giá nhanh, cắt vòng 3 / D4 |
| Thư viện huấn luyện (PEFT / TRL / bitsandbytes) chưa hỗ trợ kiến trúc Qwen3.5 hoặc thiếu wheel offline | chạy thử nhỏ (`--subset`) trên 1 GPU trước; chuẩn bị wheel trong `zc-assets`; dự phòng Unsloth |
| vLLM chưa phục vụ LoRA cho Qwen3.5 | gộp LoRA vào trọng số trước khi đánh giá / nộp |
| Fine-tune làm hỏng khả năng chung (quên) | trộn D6 nhỏ; LoRA r thấp; đánh giá held-out mỗi nhánh |
| Quá khớp public | public chỉ phân tích, không train; cổng held-out |
| Thời gian suy luận vượt 6 h nếu private lớn | dừng sớm khi đồng thuận; K thích ứng; ngân sách thời gian + ghi dần |
| Không tất định | seed cố định, nhiệt độ thấp ở bước chọn, bỏ phiếu, schema tất định (automap) |
| Container khởi động lại / tác vụ nền 2 h | `run_kernel.py --attach`; ghi trạng thái vào `handoff/STATUS.md` sau mỗi bước |

## 9. Việc của user

- Token Kaggle cho khung chat mới; gửi lại `score_1.jsonl` nếu container mới không có.
- Xác nhận (một lần) việc dùng logic luật để **tạo dữ liệu huấn luyện offline** (§4.1); mặc định: được phép.
- Build Docker và chạy thử ngày 15/11; nộp trước 16/11 23:59 GMT+7.

## Nguồn

Papers (arXiv): ReAct 2210.03629; PoT 2211.12588; PAL 2211.10435; Binder 2210.02875; Self-Debug 2304.05128; DIN-SQL
2304.11015; MAC-SQL 2312.11242; CHESS 2405.16755; CHASE-SQL 2410.01943; XiYan-SQL 2411.08599; ReFoRCE 2502.00675; DS-STAR
2509.21825; Data Interpreter 2402.18679; TaskWeaver 2311.17541; DAIL-SQL 2308.15363; What Makes Good ICE 2101.06804;
RESDSQL 2302.05965; CodeS 2402.16347; Self-Consistency 2203.11171; Adaptive-Consistency 2305.11860; ESC 2401.10480; LEVER
2302.08468; CodeT 2207.10397; Kadavath et al. 2207.05221; Just Ask for Calibration 2305.14975; TrustSQL 2403.15879;
Selective prediction text-to-SQL 2607.06799; LoRA 2106.09685; QLoRA 2305.14314; STaR 2203.14465; ReST-EM 2312.06585;
Distilling Step-by-Step 2305.02301; OmniSQL 2503.02240; Arctic-Text2SQL-R1 2505.20315; SQL-R1 2504.08600; Reasoning-SQL
2503.23157; DeepSeekMath 2402.03300; LlamaFactory 2403.13372; vLLM 2309.06180; PhoBERT 2003.00744; ViText2SQL
(2020.findings-emnlp.364).
Repos (GitHub): vanna-ai/vanna, Canner/WrenAI, sinaptik-ai/pandas-ai, eosphoros-ai/DB-GPT, eosphoros-ai/DB-GPT-Hub,
microsoft/TaskWeaver, FoundationAgents/MetaGPT, XGenerationLab/XiYan-SQL, RUCKBReasoning/OmniSQL, ShayanTalaei/CHESS,
BeachWang/DAIL-SQL, Snowflake-Labs/ReFoRCE, snowflakedb/ArcticTraining, xlang-ai/Binder, hiyouga/LlamaFactory,
unslothai/unsloth, huggingface/trl, verl-project/verl, OpenRLHF/OpenRLHF, vllm-project/vllm, VinAIResearch/PhoBERT.
Mô hình: Qwen3.5 (https://www.verdent.ai/guides/model/qwen-3-5 — nguồn thứ ba), Gemma 4
(https://www.theregister.com/2026/04/02/googles_gemma_4_open_weights/).
