# 08 — Kế hoạch tổng thể + đặc tả triển khai: hybrid framework, mục tiêu public = 1.0

*Lập 08/11/2026, user đã duyệt. **Việc của khung chat kế tiếp: triển khai code đúng theo tài liệu này, gói việc §8 theo thứ
tự WP0 → WP12**, dùng đặc tả §5 (file, hàm, định dạng, ngữ pháp, tool, thuật toán, cờ, huấn luyện) và quy tắc quyết định §10.
Lý do từng lựa chọn: `docs/09_OPTIONS_DEEP_DIVE.md` (căn cứ từ log lỗi, paper / repo); thiết kế gốc: `docs/06_HYBRID_FRAMEWORK.md`;
dataset ngoài: `docs/07_DATASETS.md`. Mọi giả định kiểm được không cần GPU đã kiểm ngày 08/11 (§1); điểm chỉ đo được trên GPU
gom vào WP1 với quy tắc quyết định viết sẵn — không còn câu hỏi mở.*

## 0. Tóm tắt

1. **Hệ đích:** Agent (mô hình chính ≤ 15B, fine-tune LoRA) sinh code cho **mọi** câu. Luật tổng quát chỉ chuẩn bị ngữ cảnh
   (schema `automap`, làm sạch câu, gói gợi ý: kỳ thời gian, thực thể khớp data, kiểu đáp án, thuật ngữ rulebook; ví dụ đã
   kiểm). Code tính toán (toolkit `dm` / `industry`; pandas / DuckDB SQL dự phòng), kiểm tra trước khi chạy, Agent đọc kết
   quả, chọn theo phiếu + tin cậy hiệu chỉnh, sinh lại hoặc abstain.
2. **Thứ tự đòn bẩy:** (a) schema từ `automap` (bỏ lỗi ánh xạ — nguồn lỗi lớn nhất của v28); (b) **sinh có ràng buộc ngữ pháp
   + tool dạng hàm** (35 % mẫu của mô hình gốc hiện hỏng vì định dạng / API / thực thể — log v34) + prompt gọn + gợi ý + ví dụ;
   (c) fine-tune LoRA bf16 trên 2681 cặp đã kiểm + paraphrase + nhiễu + dữ liệu tổng hợp "chương trình trước, câu sau" do mô
   hình thầy viết câu; (d) chọn / kiểm / tin cậy (dừng sớm, bộ kiểm 0.8B, truy hồi ngữ nghĩa 0.6B); (e) vòng sửa lỗi tổng quát
   trên public.
3. **Tìm cấu hình tốt nhất song song:** nhiều nhánh fine-tune cùng lúc (2 nhánh / phiên 4×L4, 2 phiên / acc), khác nhau về
   dữ liệu / mô hình thầy, cùng ngân sách token; đánh giá trên bộ nhanh 400 câu cố định; loại dần; nhánh thắng huấn luyện
   đầy đủ.
4. **Quota:** tuần W46 còn ≈ 75.6 GPU-giờ (quynhtran35 30, phatle1578 30, yuan1103 ≈ 15.6); từ 14/11 07:00 GMT+7 thêm 90.
   Kế hoạch dùng ≈ 70 (W46) + ≈ 45 (W47), dư phòng ≈ 50.
5. **Mục tiêu trung thực:** public = 1.0 là đích của vòng phát triển (public chỉ để phân tích lỗi, không train, không vào
   prompt). Ước lượng (chưa đo): 0.93–0.96 sau (a)+(b), 0.97–0.99 sau (c), 1.0 cần (e); không đánh đổi held-out.

## 1. Đã kiểm ngày 08/11 (không cần kiểm lại)

| Hạng mục | Kết quả | Nguồn |
|---|---|---|
| Qwen3.5-9B | 9,653,104,368 tham số (gồm vision + MTP), Apache-2.0, HF 27/02/2026; 32 lớp: 24 `linear_attention` + 8 `full_attention`; Qwen3_5ForConditionalGeneration; Kaggle danbth/qwen3-5-9b/transformers/default/1 khớp HF 4/4 file (+ sha256 trong `src/model_hashes.json`) | HF API, Kaggle API |
| Gemma 4 12B-it | 11,959,730,224 tham số, Apache-2.0, HF 23/05/2026; `Gemma4UnifiedForConditionalGeneration`; vLLM 0.32 chạy được, **không** có LoRA serving (phải gộp trọng số); Kaggle chính thức google/gemma-4/transformers/gemma-4-12b-it/2 | HF API, docs vLLM v0.32.0, Kaggle |
| Thầy Qwen3.6-27B-FP8 | 27.8B, Apache-2.0, HF 21/04/2026; Kaggle nsalzman/qwen3-6-27b-fp8/transformers/fp8/6 khớp HF 66/66 file (30.9 GB) | HF + Kaggle |
| Thầy Qwen3.6-35B-A3B-FP8 | 36.0B MoE, Apache-2.0, HF 15/04/2026; Kaggle zerabytex/qwen-3-6-35b-a3b-fp8/transformers/default/1 khớp 42/42 (37.5 GB) | HF + Kaggle |
| Thầy Gemma 4 31B-it | 31.3B, Apache-2.0, HF 11/03/2026 (công bố 02/04); Kaggle chính thức google/gemma-4/transformers/gemma-4-31b-it/1 (bf16 62.6 GB) và google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2 (23.3 GB) | HF + Kaggle |
| Thầy gpt-oss-120b / 20b | 116.8B / 20.9B, Apache-2.0, HF 04/08/2025; Kaggle `danielhanchen/gpt-oss-120b` (65.2 GB) / `-20b` (13.8 GB) khớp các file trọng số chính của HF | HF + Kaggle |
| Loại khỏi danh sách thầy | Qwen3.5-122B-A10B-GPTQ-Int4 (78.9 GB, khớp 39/39): trên 4×L4 chỉ còn ~2–3 GB / GPU cho KV → quá chậm; Qwen3.5-397B: quá lớn | tính VRAM |
| Mọi mô hình trên | trọng số public trước 01/10/2026, Apache-2.0 → hợp lệ theo luật BTC §2 | HF |
| vLLM 0.32.0 (image nộp) | phát hành 18/10/2026; yêu cầu torch 2.13.0, transformers ≥ 5.16.1, < 5.20; có LoRA cho Qwen3.5, Qwen3.5-MoE, Gemma 4 (bản thường), GPT-OSS | PyPI, docs v0.32.0 |
| Lượng tử trên L4 (Ada, SM 8.9) | hỗ trợ: FP8 W8A8, GPTQ, AWQ, Marlin (GPTQ / AWQ / FP8 / FP4, gồm MXFP4) | docs vLLM quantization |
| Fine-tune Qwen3.5 | cần transformers v5; **không khuyến nghị QLoRA 4-bit** (sai số lượng tử lớn); LoRA bf16 9B cần ~22 GB (Unsloth) — sát mức 22.5 GB của L4 | docs Unsloth |
| Bộ phiên bản huấn luyện tương thích | torch 2.13.0 (đã có trong wheels vLLM), **transformers 5.17.0** (trong khoảng của vLLM và ≤ 5.17.0 của Unsloth), peft 0.21.2, **trl 1.15.0** (unsloth_zoo yêu cầu ≤ 1.15.0), accelerate 1.15.0, datasets < 5, unsloth 2026.11.3 (tuỳ chọn) | PyPI requires_dist |
| Kernel L4 | **không có internet** → wheels huấn luyện phải nằm trong dataset; trọng số thầy lấy từ Kaggle Models | `automation/make_kernel.py`, `docs/02_KAGGLE_OPS.md` |
| Số đo thật (v34, 4×L4) | stage 2: 422 câu khó / lần chạy trong 3308–3610 s (~8.5 s / câu, ~4.5k token sinh / câu, ~171 tok/s / GPU); nạp 9B FP8 276 s; KV 214,896 token / GPU; GPU bận 73–89 %; VRAM đỉnh 21.9 / 22.5 GB | log v34 |
| Số đo thật (v25, 4×L4) | public 1000 câu / lần chạy 55–75 phút (gồm ánh xạ LLM) | EXPERIMENT_LOG |
| Dữ liệu sẵn có | 2681 cặp (câu train → chương trình đã kiểm) src/refs/answer_examples.jsonl; paraphrase train 5689 / public 2065 trong `<acc>/zc-para` | repo, Kaggle |

## 2. Tham khảo: paper / repo uy tín → bài học → áp dụng

*(Citation: Semantic Scholar; stars: GitHub — tra 08/11/2026.)*

### 2.1 Agent sinh code truy vấn, đa tác tử, phân tích dữ liệu

| Công trình | Uy tín | Ý chính | Áp dụng |
|---|---|---|---|
| ReAct (ICLR 2023) | 13,667 cit. | suy luận xen hành động / quan sát | vòng sinh code → chạy → đọc kết quả → sửa |
| Program of Thoughts (TMLR 2023), PAL (ICML 2023) | 1,587 / 911 cit. | LLM viết chương trình, trình thông dịch tính | số liệu chỉ từ code; định dạng đáp án bằng code |
| Binder (ICLR 2023) | 368 cit., 324★ | ngôn ngữ ký hiệu + API | toolkit `dm` = API nghiệp vụ |
| Self-Debug (ICLR 2024) | 1,519 cit. | sửa code theo lỗi thực thi | `--repairs`, lỗi giàu thông tin (LookupError kèm tên gần nhất) |
| DIN-SQL (NeurIPS 2023) | 998 cit. | liên kết schema → sinh → tự sửa | gói gợi ý trước khi sinh |
| MAC-SQL (COLING 2025) | 350 cit. | selector / decomposer / refiner | lọc schema cho prompt gọn |
| CHESS (2024) | 278 cit., 288★ | truy hồi giá trị / thực thể | liên kết thực thể nhiều tầng |
| CHASE-SQL (ICLR 2025) | 244 cit. | nhiều đường sinh + chọn | 2–3 kiểu prompt đa dạng |
| XiYan-SQL | 93 cit., 1,019★ | nhiều bộ sinh fine-tune + bộ chọn | nhiều nhánh LoRA làm nhiều bộ sinh (tuỳ chọn) |
| ReFoRCE (Spider 2.0 top) | 42 cit., 143★ | thăm dò cột / giá trị, đồng thuận | tool thăm dò giá trị (M8) |
| DS-STAR (Google, 2025) | 13 cit., top DABStep | mô tả file → Planner / Coder / Verifier | data card (M4), đọc kết quả (M11) |
| Data Interpreter (ACL), MetaGPT | 322 cit., 71,215★ | kế hoạch đồ thị, kiểm từng bước | câu nhiều bước |
| TaskWeaver | 115 cit., 6,160★ | agent code-first + plugin | hàm toolkit có chữ ký rõ |
| Vanna / WrenAI / PandasAI / DB-GPT | 23,801★ / 18,106★ / 23,938★ / 20,273★ | RAG ví dụ đã kiểm; semantic layer; pandas agent; khung + SFT | ngân hàng ví dụ; toolkit = semantic layer; đường dự phòng pandas |

### 2.2 Ngữ cảnh, ví dụ, chọn, tin cậy

| Công trình | Uy tín | Áp dụng |
|---|---|---|
| DAIL-SQL (PVLDB 2024) | 868 cit., 642★ | truy hồi ví dụ theo khung câu đã che thực thể / số / kỳ |
| What Makes Good In-Context Examples (DeeLIO 2022) | 1,972 cit. | ví dụ gần nghĩa nhất |
| RESDSQL (AAAI 2023), CodeS (SIGMOD 2024) | 448 / 385 cit. | tách liên kết schema khỏi sinh; mô hình mở fine-tune |
| Self-Consistency (ICLR 2023) | 8,228 cit. | bỏ phiếu theo kết quả thực thi |
| Adaptive-Consistency (EMNLP 2023), ESC (ICLR 2024) | 193 / 164 cit. | dừng sớm khi các mẫu đầu đồng ý |
| LEVER (ICML 2023), CodeT (ICLR 2023) | 415 / 683 cit. | bộ kiểm trên (câu, chương trình, kết quả); phép thử bất biến |
| Kadavath et al.; Just Ask for Calibration (EMNLP 2023) | 2,320 / 1,091 cit. | tự đánh giá là tín hiệu, cần hiệu chỉnh |
| TrustSQL; selective prediction text-to-SQL (2026) | 26 cit.; mới | chấm có phạt như PA; kết hợp nhiều tín hiệu |

### 2.3 Huấn luyện / chưng cất

| Công trình / công cụ | Uy tín | Áp dụng |
|---|---|---|
| LoRA (ICLR 2022), QLoRA (NeurIPS 2023) | 25,528 / 6,318 cit. | LoRA bf16 (QLoRA không khuyến nghị cho Qwen3.5) |
| STaR (NeurIPS 2022), ReST-EM (TMLR 2024) | 1,241 / 434 cit. | tự huấn luyện có lọc theo kết quả thực thi |
| Distilling Step-by-Step (ACL Findings 2023) | 1,259 cit. | học cả lời giải thích của thầy |
| OmniSQL (PVLDB 2025) | 149 cit., 457★ | tổng hợp "chương trình trước → câu sau → CoT", kiểm thực thi |
| Arctic-Text2SQL-R1 (ACL Findings 2026), SQL-R1 (NeurIPS 2025) | 61 / 107 cit. | RL thưởng thực thi (ngoài phạm vi trước 16/11) |
| LlamaFactory (ACL 2024), Unsloth, TRL, verl | 75,935★ / 79,188★ / 19,690★ / 24,163★ | TRL + PEFT (chính), Unsloth (nếu vừa 1 GPU) |
| vLLM (SOSP 2023) | 9,947 cit., 96,542★ | suy luận, prefix cache, LoRA |
| PhoBERT; ViText2SQL (Findings EMNLP 2020) | 534 / 68 cit. | mô hình phụ khả dĩ; dữ liệu câu truy vấn tiếng Việt |

**Tám nguyên tắc:** số liệu chỉ từ code; ngữ cảnh dữ liệu giàu nhưng đã lọc; ngữ nghĩa trong lớp chỉ số đã kiểm; ví dụ đã
kiểm chọn theo khung câu; nhiều ứng viên + chọn + dừng sớm; tự sửa theo thực thi; tin cậy = nhiều tín hiệu đã hiệu chỉnh;
mô hình nhỏ mạnh lên nhờ dữ liệu tổng hợp có kiểm thực thi + fine-tune.

## 3. Thước đo, bộ đánh giá, cổng

| Bộ | Nội dung (cố định, file id trong `tests/splits/`) | Vai trò |
|---|---|---|
| `pub_full` | 1000 câu public, chấm với `data/raw/public_gt.jsonl` | dev — mục tiêu 1.0 |
| `pub_k0` | 692 paraphrase k=0 | dev |
| `pub_ho` | 1373 paraphrase k=1,2 | **held-out** |
| `tr_ho` | 6 DN train: biz_105 (sổ tay theo tháng), biz_109 (POS), biz_114 (CSV phẳng), biz_123 (SQLite), biz_127 (nhiều CSV), biz_130 (sổ chi tiết xlsx) — 631 câu gốc + paraphrase; **loại khỏi dữ liệu huấn luyện và ngân hàng ví dụ** khi đo | **held-out** |
| `quick400` | 150 public (phân tầng theo DN × kiểu đáp án, seed 20261108) + 150 k=0 + 100 `tr_ho` | xếp hạng nhánh nhanh |
| `pub_noind` | 112 câu public dùng hàm ngành (dev_011–dev_014) chạy với hàm ngành bị ẩn khỏi API / ngữ pháp | mô phỏng ngành mới của private |
| `pub_noacc` | câu public bỏ dấu (`src/perturb.py`) | bền vững với câu không dấu |
| `robust` | `tests/eval_all.sh` (đổi tên cột, README trắng, nhiễu) | phần schema / làm sạch |

**Cổng giữ một thay đổi:** `pub_ho` và `tr_ho` không giảm quá nhiễu (±0.5 điểm trên quick400; ±0.3 trên bộ đầy đủ); `pub_full`
và `pub_k0` tăng hoặc giữ; thời gian trong ngân sách; `tests/run_all.sh` ALL PASS; preflight OK.
**Thời gian đích:** ≤ 2 h / 1000 câu trên 2×RTX 5090 (≈ ≤ 6 h trên 4×L4); tất định |ΔPA| ≤ 0.01 giữa 2 lần chạy.

## 4. Kiến trúc đích

### 4.1 Luồng chạy (thay `src/predict_flow.py` rule-first)

**Mỗi split (một lần):** M3 diễn giải rulebook → `rulebook_terms.json`.
**Mỗi DN (một lần, cache):** M1 ingest → M2 schema `automap` (LLM ánh xạ dự phòng khi cảnh báo nặng / thiếu bảng) → M4 data
card → API spec + ngữ pháp của DN (§5.4) → phần prompt chung của DN (prefix cache).
**Mỗi câu:** M5 gói gợi ý → M6 truy hồi ví dụ → M7 sinh vòng 1 (2 mẫu, có ngữ pháp) → M9 precheck → M10 chạy → T7 postcheck →
dừng sớm nếu đạt (§5.7) → nếu không: vòng 2 (thêm mẫu, biến thể prompt, lượt sửa + khám phá bằng tool) → M11 đọc kết quả khi
nghi ngờ → M12 chọn (phiếu + bộ kiểm) + tin cậy → trả lời / abstain → M13 ghi ngay.

**Bỏ khỏi luồng nộp:** stage 1 luật trả lời thẳng (`rules_first.py`), stage 1.5 (`canon_llm.py`), `--confirm` dựa trên luật,
`--qw` > 0 (chương trình luật không là phiếu). Các module luật vẫn giữ làm thư viện: `qnorm` (làm sạch), `qparse.parse_period`
và hàm liên kết thực thể (gợi ý), `automap` (schema), `rules_first` / `qparse` (gán nhãn dữ liệu huấn luyện offline, đánh giá).

### 4.2 Module (M = mô hình, C = code / luật logic được phép)

| # | Module | Ai | File |
|---|---|---|---|
| M1 | Đọc data | C | `src/ingest.py` (giữ) |
| M2 | Schema chuẩn C | C (+M dự phòng) | `src/automap.py`; dự phòng: đường ánh xạ LLM hiện có trong `src/pipeline.py` + `src/canon.py` + `src/consensus.py` |
| M3 | Diễn giải rulebook | C + M | `src/rulebook.py` (mới): `changed_terms` (code) + M tóm tắt định nghĩa đổi / mới thành JSON |
| M4 | Data card | M | `src/datacard.py` (mới) |
| M5 | Làm sạch + gói gợi ý | C | `src/qnorm.py` (giữ) + `src/hints.py` (mới) |
| M6 | Truy hồi ví dụ | C (+S1) | `src/retrieval.py` (mới; thay phần chọn ví dụ của `src/fewshot.py`) |
| M7 | Sinh chương trình | M | `src/agent.py` (prompt gọn, ngữ pháp, biến thể prompt, multi-LoRA) |
| M8 | Tool T1–T9 | C (M gọi) | `src/tools.py` (mới) |
| M9 | Precheck | C | `src/precheck.py` (mới) |
| M10 | Chạy | C | sandbox hiện có (`src/pipeline.py`, `dm.run_program`) + vết lọc (`dm` trace, §5.5 T6) |
| M11 | Đọc kết quả | M | `src/review.py` (mới) |
| M12 | Chọn + tin cậy | C + S3 | `src/confidence.py` (mới), `src/verifier.py` (mới), `src/vote.py` (giữ cho nhiều lần chạy) |
| M13 | Định dạng + ghi | C | `fmt_answer`, `src/merge_sub.py` (giữ) |
| — | Luồng | C | `src/predict_flow.py` (viết lại), cấu hình `configs/*.json` |

## 5. Đặc tả kỹ thuật (để triển khai chính xác)

### 5.1 File mới / sửa

| File | Nội dung chính (hàm công khai) |
|---|---|
| `src/api_spec.py` | `build_api_spec(shop, today, hide=()) -> list[dict]`: duyệt namespace `dm.api(shop, today)` (PERIOD_API, NUM_API, `industry.HELPERS`, `shop_api(shop)`, tool T1–T9) → `{name, signature, params:[{name, kind: period / item / customer / staff / supplier / channel / payment / status / group / vehicle / number / text / bool / lambda, required}], returns, doc, example}`; `doc` lấy từ `dm.API_DOC` + docstring; `example` từ ngân hàng ví dụ đã kiểm. `hide` dùng cho bộ `pub_noind` (ẩn hàm ngành) |
| `src/grammar.py` | `build_grammar(api_spec, entity_values, max_enum=5000, escapes=("python",)) -> str` (EBNF cho xgrammar, §5.4); cache theo (DN, hash spec) |
| `src/hints.py` | `build_hints(question, as_of, shop, normalizer, rule_terms) -> dict` (§5.2) |
| `src/tools.py` | T1–T9 (§5.5): `bind_tools(shop, today, rulebook, data_end) -> dict[name, fn]`, được chèn vào namespace chương trình |
| `src/precheck.py` | `precheck(code, api_spec, hints, shop) -> list[{level: error / warn, rule, msg, suggestion}]` (§5.6) |
| `src/datacard.py` | `make_datacard(engine, bid, schema_card, readme) -> str` (≤ 300 token, cache `<cache>/<bid>.datacard.txt`) |
| `src/rulebook.py` | `interpret(split_dir, engine) -> {"changed": [...], "terms": [...]}` (cache theo hash rulebook) |
| `src/retrieval.py` | `Bank.load(path)`, `skeleton(question, hints) -> str` (che thực thể / số / kỳ), `retrieve(q, k=4, exclude_bids=(), mode="ngram" / "embed") -> list[example]`; MMR + không trùng họ hàm |
| `src/prompting.py` | `system_block(api_spec)`, `business_block(schema_card, datacard, rule_changes)`, `question_block(examples, hints, question, variant)`; `parse_output(text) -> {plan, code, escape}` (§5.3) |
| `src/review.py` | `needs_review(cands) -> bool`, `review(engine, question, cand, explain_text) -> {"verdict": OK / FIX / NONE, "code"}` |
| `src/confidence.py` | `features(cands, ...) -> dict`, `Calibrator.fit(X, y)` / `.predict(x)` (logistic numpy), `choose_thresholds(dev) -> (θ_regen, θ_abstain)`, `conformal_check(...)` |
| `src/verifier.py` | `Verifier(engine_or_path).score(question, code, result_summary) -> P(đúng)` (mô hình sinh: P(token "1")) |
| `src/predict_flow.py` | luồng §4.1, đọc `configs/<name>.json` + cờ CLI (§5.8); ghi `submission.jsonl` dần + `logs/<id>.json` |
| `src/agent.py` | thêm cờ §5.8; dùng `prompting.py`, `StructuredOutputsParams(grammar=...)`, multi-LoRA (`enable_lora`, `LoRARequest`), logprobs |
| `src/dm.py` | vết lọc: mỗi bước lọc ghi (tên bộ lọc, giá trị, số dòng trước / sau) vào `shop.trace` khi bật |
| `src/eval_flow.py` | chạy luồng trên các bộ (`--sets quick400,pub_full,...`), dựng split con (như `stress_flow.py`), ẩn hàm ngành cho `pub_noind` |
| `tests/make_splits.py` | tạo `tests/splits/{pub_full,pub_k0,pub_ho,tr_ho,quick400,pub_noind,pub_noacc}.txt` (seed 20261108) |
| `tests/score_sets.py` | bảng PA / đúng / sai / abstain / thời gian theo bộ từ log của luồng |
| `training/` | `data/d1_train_pairs.py`, `d2_paraphrase.py`, `d2n_noise.py` (H2), `d3_sample_programs.py`, `d3_teacher_questions.py`, `d3_roundtrip.py`, `d4_tool_traj.py` (H4), `d5_rationales.py`, `d6_external.py`, `d7_rest.py`, `verifier_data.py`; `sft.py` (TRL + PEFT), `merge.py` (PEFT `add_weighted_adapter` + `merge_and_unload`), `verifier_train.py`, `teacher_gen.py`, `smoke.py`, `embed_bank.py`; `fsdp2.yaml`, `fsdp4.yaml`; `README.md` (cách tái lập — yêu cầu nộp code train) |
| `configs/` | `e0.json`, `e1.json`, …, `flow_final.json` (cấu hình luồng được chọn ở WP10) |
| `automation/jobs/` | `smoke_v35.jobs`, `v35_e0.jobs`, `v35_s0.jobs`, `arms_r1a.jobs`, `arms_r1b.jobs`, `teacher_probe.jobs`, `d3_gen.jobs`, `arms_r2.jobs`, `verifier.jobs`, `sft_full.jobs`, `final_eval.jobs` |
| `src/run_kaggle.py`, `automation/make_kernel.py` | chế độ `--train`: cài wheels huấn luyện từ dataset `<acc>/zc-train-wheels` (không internet), chạy `accelerate launch`; gói thêm `training/` + `configs/` vào code dataset |

### 5.2 Định dạng dữ liệu

- **Gợi ý (M5):** `{"period": {"text", "code", "from", "to"} | null, "periods": [...] (câu có nhiều kỳ), "entities": [{"text",
  "kind", "data_value", "how": exact / no_accent / regional / abbrev / typo / semantic, "candidates": [...]}], "answer_type":
  number / percent / list / date / string / boolean / table / unanswerable / null, "rule_terms": [{"term", "definition"}],
  "missing_data": [...], "measure_words": [...]}`. Không chứa tên hàm toolkit, tên mẫu câu hay chương trình.
- **API spec:** như §5.1 (`src/api_spec.py`).
- **Mẫu SFT (JSONL):** `{"id", "business_id", "source": D1 / D2 / D2N / D3:<teacher> / D4 / D6:<name> / D7, "messages":
  [system, user, assistant], "answer": {...}, "program": "..."}`; assistant = đúng định dạng đầu ra §5.3.
- **Mẫu bộ kiểm (JSONL):** `{"question", "program", "result_summary", "label": 0 / 1, "business_id", "source"}`.
- **Log mỗi câu của luồng:** câu, gợi ý, id ví dụ, mọi mẫu (prompt variant, code, lỗi, kết quả, cờ precheck / postcheck,
  logprob), phán quyết M11, điểm bộ kiểm, đặc trưng tin cậy, quyết định, thời gian — đủ để chấm lại / đổi ngưỡng offline.

### 5.3 Prompt gọn và đầu ra

Thứ tự (phần chung trước, dùng prefix cache): `# ROLE` → `# API` (tài liệu rút gọn từ API spec; ẩn hàm theo `hide`) →
`# OUTPUT FORMAT` → `# BUSINESS` (card schema chuẩn ≤ 1,000 token: bảng / cột có thật + 2 giá trị ví dụ; data card ≤ 300;
thay đổi rulebook) → `# EXAMPLES` (2–4) → `# HINTS` (JSON §5.2, rút gọn) → `# QUESTION` (câu gốc + ngày hỏi).
Biến thể prompt cho vòng 2: `direct` (mặc định), `rephrase` (I5: mô hình viết lại câu trong `<plan>` trước), `explore` (được
gọi T2 / T5 / T9 trước khi viết code, tối đa 2 lượt).
**Đầu ra mô hình (không suy nghĩ dài):**
```
<plan>đại lượng | kỳ | bộ lọc (một dòng)</plan>
<code>
ANSWER = {"type": "number", "answer": revenue(month(2025, 7), item="Cá lóc")}
</code>
```
Nhánh thoát: `<python>…</python>` (pandas trên `C` + toolkit + tool) hoặc `<sql>…</sql>` (DuckDB, WP12). Tin cậy không do mô
hình tự khai (bỏ `<conf>`); dùng logprob của chương trình làm đặc trưng. Nhãn huấn luyện: `<plan>` sinh tự động từ chương trình
("hàm | kỳ | bộ lọc"); D5 thay bằng lời giải ngắn của thầy (nếu QĐ8 giữ).

### 5.4 Ngữ pháp (xgrammar EBNF qua `StructuredOutputsParams(grammar=...)`)

```
root      ::= "<plan>" plan "</plan>\n" (code | py | sql)
code      ::= "<code>\n" stmt* "ANSWER = {\"type\": \"" atype "\", \"answer\": " expr "}\n</code>"
py        ::= "<python>\n" free "\n</python>"            # nhánh thoát (không ràng buộc nội dung)
sql       ::= "<sql>\n" free "\n</sql>"                  # chỉ khi bật WP12
stmt      ::= name " = " expr "\n"
expr      ::= call | number | string | list | dict | name | expr binop expr | "(" expr ")" | attr | subscript | lambda
call      ::= <một quy tắc riêng cho mỗi hàm trong API spec: tên + đúng tên tham số + kiểu giá trị theo kind>
ent_item  ::= "\"" ("Cá lóc" | "Nếp cái hoa vàng" | …) "\""   # giá trị thật của DN; > max_enum → string tự do
method    ::= ".sum()" | ".mean()" | ".max()" | ".idxmax()" | ".head(" int ")" | ".tolist()" | …   # danh sách cho phép
lambda    ::= "lambda d: " expr                          # d["<cột có thật>"] …
```
Sinh tự động từ API spec + giá trị thực thể của DN. **Nghiệm thu:** 100 % trong 2,681 chương trình đã kiểm + chương trình D3
khớp ngữ pháp của DN tương ứng (`tests/test_grammar.py`); thời gian dựng < 2 s / DN; giải mã chậm hơn ≤ 15 %.

### 5.5 Tool T1–T9 (tất định; đầu ra cắt ≤ 300 token)

| Tool | Hành vi |
|---|---|
| T1 `find(kind, text, k=5)` | khớp tầng: chính xác sau chuẩn hoá → không dấu → địa phương (`regional_key`) → viết tắt / tiền tố dòng máy → gõ sai (difflib) → ngữ nghĩa (S1 nếu bật); trả `[(giá trị thật, kiểu khớp, điểm, số dòng)]` |
| T2 `values(table, column, contains=None, k=20)` | giá trị khác nhau + số lần, lọc chuỗi con không dấu |
| T3 `period(text, today=TODAY)` | dùng `qparse.parse_period` + lịch `calendar.csv` + rulebook → `Period`; không hiểu → lỗi kèm ví dụ cụm hợp lệ |
| T4 `rule(term)` | định nghĩa trong rulebook của split (`rulebook.py`) + hàm toolkit tương ứng nếu có |
| T5 `schema(table=None)`, `peek(table, n=3, where=None)` | cột, kiểu, % trống, ví dụ; vài dòng |
| T6 `explain(code)` | chạy với `shop.trace` bật → số dòng sau mỗi bộ lọc, bộ lọc khớp 0 dòng, đơn vị quy đổi |
| T7 `check(answer, expected_type)` | postcheck §5.6 |
| T8 `help(name)` | chữ ký + mô tả + 1 ví dụ đã kiểm từ API spec |
| T9 `has_data(concept)` | concept ∈ {tồn kho, giá vốn, nợ, nhân viên, ca, kênh, khách hàng, nhà cung cấp, lô / hạn dùng, bảo hành, trả góp, phiếu sửa…} → có / không + bảng nguồn (từ C) |

### 5.6 Precheck / postcheck

**Precheck (trước khi chạy):** (1) parse AST; (2) chỉ dùng tên có trong API spec / tool / builtins cho phép (`round, len, sum,
min, max, sorted, abs, int, float, str, list, dict, set, enumerate, zip, range`); (3) tham số đúng tên / số lượng theo spec; (4)
chuỗi thực thể có trong data (sau chuẩn hoá) — nếu không: lỗi kèm gợi ý T1; (5) kỳ trong code khác kỳ gợi ý (khoảng ngày khác)
→ cảnh báo (trừ câu nhiều kỳ); (6) `type` khác `answer_type` gợi ý → cảnh báo; (7) không import / mở file / mạng.
**Postcheck (sau khi chạy, T7):** số đếm nguyên ≥ 0; tỉ trọng ∈ [0, 100]; tiền ≥ 0 trừ đại lượng chênh lệch / lãi lỗ; list
top-N đủ N (hoặc ít hơn khi data ít hơn); ngày trong khoảng data; kết quả rỗng / None khi `type` ≠ unanswerable → cờ; kết quả 0
khi một bộ lọc khớp 0 dòng (T6) → cờ.

### 5.7 Thuật toán sinh, chọn và tin cậy

```
cands = sample(variant="direct", n=2, temps=[0.0, 0.7], grammar=G)          # vòng 1
for c in cands: c.pre = precheck(c); if not error: c.res = run(c); c.post = postcheck(c)
if two valid, same answer, no flags, verifier ≥ v_fast (nếu bật): accept     # dừng sớm
else:                                                                       # vòng 2 (ngân sách thời gian cho phép)
    cands += repair(failed, error_msg, tools, max_turns=2)                  # Self-Debug / CodeAct
    cands += sample(variant in ["direct", "rephrase"], n=K_max-2, temps=0.7)
    if suspicious(top): cands += review(top, explain(top))                  # M11 / CRITIC
groups = group_by_answer(valid cands)
score(g) = vote_share(g) + λ·mean_verifier(g)                               # λ chọn trên dev; λ = 0 nếu tắt S3
best = argmax score; conf = Calibrator(features(best, groups, flags, logprob, verifier, review))
answer if conf ≥ θ_abstain else abstain                                      # θ chọn tối đa PA trên dev, kiểm conformal
```
Mặc định trước khi tinh chỉnh: `K_max = 8`, `v_fast = 0.8`, `max_turns = 2`, một lần chạy (`--runs 1`) với seed cố định.

### 5.8 Cờ CLI (`src/agent.py` / `src/predict_flow.py`) — mặc định ban đầu

`--prompt compact|full` (compact) · `--hints 0|1` (1) · `--grammar 0|1` (1) · `--escape python|python,sql|none` (python) ·
`--tools find,period,check,help[,values,schema,explain,has_data,rule]` · `--explore-turns 2` · `--variants direct,rephrase` ·
`--k-fast 2 --k-max 8` · `--early-stop 1` · `--review 0|1` (1) · `--retriever ngram|embed` (ngram cho tới QĐ11) ·
`--embed-model PATH` · `--verifier PATH` · `--verifier-weight λ` · `--calibrator PATH` · `--theta-abstain θ` · `--lora PATH[,PATH…]`
(nhiều adapter = multi-LoRA) · `--exclude-bids biz_105,…` (đánh giá `tr_ho`) · `--hide-api industry` (bộ `pub_noind`) ·
`--qw 0` · không `--confirm` · `--runs 1` · `--budget S` · `--config configs/<name>.json` (ghi đè mặc định).

### 5.9 Huấn luyện

- **Ngăn xếp:** torch 2.13.0 (wheels vLLM 0.32), transformers 5.17.0, peft 0.21.2, trl 1.15.0, accelerate 1.15.0, datasets 4.x;
  wheels → dataset `<acc>/zc-train-wheels`.
- **SFT (`training/sft.py`):** LoRA bf16, `target_modules="all-linear"` trong mô hình ngôn ngữ (không vision, không
  `lm_head`); nhánh: r 32, alpha 64, lr 2e-4, 1 epoch, ngân sách ~6M token; bản cuối: r 64, alpha 128, lr 1e-4, 2 epoch;
  dropout 0.05; cosine, warmup 3 %; chuỗi ≤ 2,048; loss chỉ trên phần assistant; batch hiệu dụng 64; gradient checkpointing;
  seed 20261108; FSDP `full_shard` (`training/fsdp2.yaml` cho 2 GPU, `fsdp4.yaml` cho 4 GPU) — theo QĐ1.
- **Hai nhánh / phiên:** `CUDA_VISIBLE_DEVICES=0,1` và `2,3`, hai cổng master khác nhau; mỗi nhánh xong → gộp adapter → đánh
  giá `quick400` K = 1 trên GPU của nhánh (vLLM, `--lora` hoặc trọng số đã gộp theo QĐ3).
- **Gộp nhánh (`training/merge.py`):** PEFT `add_weighted_adapter` với `linear`, `ties`, `dare_ties` (density 0.5) trên 2–3
  nhánh tốt nhất → đánh giá như một nhánh (QĐ9).
- **Bộ kiểm (`training/verifier_train.py`):** Qwen3.5-0.8B, LoRA r 16, đầu vào ≤ 512 token (câu + chương trình + tóm tắt kết
  quả), nhãn "1" / "0", cân bằng lớp, 1 epoch, 1 GPU.
- **Dò thầy / sinh câu (`training/teacher_gen.py`):** vLLM với TP theo bảng §6.2; sau đó tải thầy ra, nạp học trò để lọc vòng
  lặp ngược trong cùng phiên.

### 5.10 Mô hình phụ lúc chạy

- **S1 Qwen3-Embedding-0.6B:** ngân hàng ví dụ (train đã kiểm + D3 đã kiểm, dạng khung câu đã che) nhúng trước bằng
  `training/embed_bank.py` → `refs/bank/embeddings.npy` + `meta.jsonl` đóng vào image; lúc chạy chỉ nhúng câu hỏi (vLLM pooling
  hoặc transformers trên CPU).
- **S3 bộ kiểm Qwen3.5-0.8B:** engine vLLM thứ hai trên cùng GPU (`gpu_memory_utilization` 0.08; mô hình chính 0.82) hoặc chấm
  theo lô sau khi sinh xong mỗi DN; điểm = P("1") từ logprobs.
- **Tổng tham số lúc chạy:** 9.65B + 0.60B + 0.87B = 11.12B.

### 5.11 Docker

`docker/fetch_assets.sh`: Qwen3.5-9B (HF, sha256), adapter / trọng số gộp (dataset Kaggle của dự án, sha256), Qwen3-Embedding-0.6B
(HF), bộ kiểm (dataset Kaggle); `docker/Dockerfile`: thêm `duckdb` nếu WP12 giữ, chép `configs/`, `refs/bank/`; `docker/predict.sh`
→ `src/predict_flow.py --config configs/flow_final.json`; `docker/README_SOLUTION.md`: mô hình + tham số, dataset ngoài, cách
huấn luyện (`training/README.md`).

## 6. Mô hình

### 6.1 Mô hình chính (≤ 15B lúc chạy)

| Ứng viên | Tham số | Ghi chú |
|---|---|---|
| **Qwen3.5-9B** (mặc định) | 9.65B | hạ tầng sẵn, sha256 đã kiểm, vLLM LoRA có; còn ~5.3B cho mô hình phụ |
| Gemma 4 12B-it | 11.96B | 05/2026; vLLM 0.32 chạy được, không LoRA serving → gộp trọng số; còn ~3B (S1 + S3 0.8B vẫn vừa) |
| Qwen3.5-4B | 4.66B | dùng nếu thời gian suy luận vượt ngân sách |

Chọn bằng S0 (WP3) theo QĐ4.

### 6.2 Mô hình thầy (offline, không tính vào 15B)

| Thầy | Nguồn Kaggle (đã đối chiếu HF) | Chạy trên 4×L4 |
|---|---|---|
| Qwen3.6-27B-FP8 | nsalzman/qwen3-6-27b-fp8/transformers/fp8/6 | TP 2 × 2 bản |
| Qwen3.6-35B-A3B-FP8 | zerabytex/qwen-3-6-35b-a3b-fp8/transformers/default/1 | TP 4 |
| Gemma 4 31B-it | google/gemma-4/transformers/gemma-4-31b-it/1 (`--quantization fp8`) hoặc google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2 | TP 2 × 2 bản |
| gpt-oss-120b (dự phòng 20b) | danielhanchen/gpt-oss-120b/transformers/default/1 (`-20b` cho bản nhỏ) | TP 4 (20b: 1 GPU) |
| Qwen3.5-9B (tự chưng cất) | danbth/qwen3-5-9b/transformers/default/1 | DP 4 |

Chưng cất on-policy mức token (H6, P2) dùng được với thầy Qwen3.6 vì tokenizer trùng hoàn toàn với Qwen3.5 (đã kiểm).

## 7. Dữ liệu huấn luyện

| Mã | Dữ liệu | Quy mô (ước) | Nhãn |
|---|---|---|---|
| D1 | câu train → chương trình đã kiểm (bỏ 6 DN `tr_ho`) | ~2,090 | `src/refs/answer_examples.jsonl` |
| D2 | paraphrase câu train (bỏ `tr_ho`), giữ khi lọc vòng lặp ngược ra cùng đáp án | ~4,400 | chương trình câu gốc |
| D2N | **tăng cường nhiễu** (H2): không dấu, gõ sai, viết tắt, chat, từ địa phương (`src/perturb.py`) trên D1 + D2 | ~6,000 | như câu gốc |
| D3 | "chương trình trước, câu sau" trên 24 DN train: lấy mẫu chương trình theo ngữ pháp toolkit → chạy → thầy viết 2 câu / chương trình → lọc vòng lặp ngược | 10–20k | toolkit |
| D4T | **quỹ đạo dùng tool** (H4): mẫu lỗi thật (LookupError, sai tham số, sai cột) → gọi T1 / T5 / T8 → chương trình đúng | ~2,000 | thực thi |
| D5 | lời giải ngắn của thầy cho `<plan>` | = D1–D3 | giữ khi chương trình đi kèm chạy đúng |
| D6 | ngoài (`docs/07_DATASETS.md` §4): ViText2SQL, TableBench-TableInstruct, BIRD train | ≤ 10 % hỗn hợp | nhãn của dataset |
| D7 | ReST: mẫu của chính mô hình trên D1–D3, giữ cái chạy đúng | tuỳ | thực thi |
| DV | dữ liệu bộ kiểm: K mẫu của mô hình trên D1 + D3 có nhãn đúng / sai theo thực thi | 20–50k | thực thi |
| **Cấm** | câu / data / paraphrase của public test, private test | — | — |

Dùng logic luật (bộ sinh chương trình, toolkit) để **tạo dữ liệu huấn luyện offline** là hard code logic theo ranh giới user
đã chốt. Ví dụ trong prompt huấn luyện được truy hồi kiểu leave-one-out (không lấy chính câu đó hay paraphrase của nó).

## 8. Gói việc (WBS) — làm theo đúng thứ tự; mỗi gói: sản phẩm, test, nghiệm thu, GPU

**WP0 — Chuẩn bị (09/11, CPU).** Xin token; tải `nhatlong1103/zc-2026`; `score_1.jsonl` → `data/raw/public_gt.jsonl`; ingest
cache; tải `<acc>/zc-para`; `tests/make_splits.py`; `pip download` bộ huấn luyện §5.9 → tạo dataset `<acc>/zc-train-wheels` ở
3 acc L4. *Nghiệm thu:* `tests/eval_all.sh quick` 3000 / 3000, 1000 / 1000; file split đúng số câu.

**WP1 — Chạy thử GPU (09/11, yuan1103, ~5 GPU-h).** `training/smoke.py`: (a) LoRA bf16 9B 30 bước trên 200 mẫu D1 — 1 GPU
(Unsloth), FSDP 2, FSDP 4: VRAM đỉnh, tok/s, tên module LoRA; (b) gộp → vLLM `--quantization fp8` sinh 20 câu; adapter qua
`--lora` → so tốc độ; (c) nạp Qwen3.6-27B-FP8 (TP 2) và Gemma 4 31B-it (fp8 động, TP 2), sinh 50 câu mỗi thầy; (d) ngữ pháp
mẫu qua `StructuredOutputsParams` trên 50 câu — đo chậm thêm. *Kiểm trước:* `--dry-run` trên CPU; review; `--no-preflight`.
*Nghiệm thu:* số đo vào `EXPERIMENT_LOG.md`; áp QĐ1–QĐ3, QĐ12.

**WP2 — Luồng hybrid v1 (09–10/11, CPU) — P0.** `api_spec.py`, `grammar.py`, `hints.py`, `tools.py` (T1, T3, T7, T8),
`precheck.py`, `prompting.py`, cờ §5.8 trong `agent.py`, `predict_flow.py` viết lại (§4.1, chưa có M4 / M11 / S1 / S3), `dm` trace.
*Test mới:* `test_api_spec` (mọi hàm trong `dm.api` có trong spec), `test_grammar` (100 % chương trình đã kiểm khớp), `test_hints`
(kỳ đúng ≥ 99 %, thực thể ≥ 98 % trên 200 câu train có nhãn; không chứa tên hàm), `test_tools` (tất định, ≤ 300 token, T1 tìm
đúng tên trên các ca địa phương / không dấu / gõ sai), `test_precheck` (bắt hàm / tham số / thực thể sai; 0 báo nhầm trên 2,681
chương trình đã kiểm), `test_flow_hybrid` (mô hình giả: mọi câu qua mô hình, không câu nào lấy đáp án luật, ghi dần, hết giờ giữ
đáp án). Cập nhật `test_rules_first` / `test_canon_flow` thành test thư viện. *Nghiệm thu:* `tests/run_all.sh` ALL PASS;
preflight OK cho `v35_e0.jobs`.

**WP3 — Mốc E0 + chọn mô hình chính S0 + A/B (10/11, ~14 GPU-h).** E0 (yuan1103): luồng v1, Qwen3.5-9B, `quick400` + `pub_full`.
S0 + A/B (quynhtran35): Gemma 4 12B-it trên `quick400`; prompt `compact` vs `full`; ngữ pháp bật / tắt; `pub_noind` (ẩn hàm ngành)
với nhánh thoát pandas; `pub_noacc`. *Nghiệm thu:* bảng `tests/score_sets.py`; áp QĐ4, QĐ5, QĐ12.

**WP4 — Nhà máy dữ liệu (09–11/11, container + phiên CPU Kaggle).** Script §5.1 `training/data/*`; D1, D2, D2N, D3 (bước lấy
mẫu + chạy), D4T; định dạng §5.2. *Test:* `test_data_factory` (mẫu chạy lại ra đúng nhãn; không id public / DN `dev_*` / DN
`tr_ho`; phân bố kiểu đáp án lệch ≤ 5 điểm %; trùng < 1 %), `test_sft_format` (đúng prompt §5.3, nhãn chỉ ở assistant, khớp ngữ
pháp). *Nghiệm thu:* `training/data/out/*.jsonl` + thống kê.

**WP5 — Dò thầy (11/11, quynhtran35, ~5 GPU-h).** 1,000 chương trình D3 + 300 câu train cần paraphrase, giống nhau cho mọi thầy
(§6.2). Đo tỉ lệ giữ sau lọc vòng lặp ngược, distinct-2, tỉ lệ câu tiếng Việt có dấu chuẩn, tok/s. *Nghiệm thu:* QĐ6.

**WP6 — Nhánh fine-tune (11–12/11).** Vòng 1 (phatle1578, 2 phiên song song, ~16 GPU-h): A1 = D1; A2 = D1 + D2 + D2N; A3 = A2 +
D3 (câu do 9B viết); A4 = A3 + D4T + D6 (10 %). Vòng 2 (quynhtran35, ~8–16 GPU-h): hỗn hợp tốt nhất với D3 do thầy 1 / thầy 2 viết,
± D5. Sau mỗi vòng: gộp 2–3 nhánh tốt nhất (linear / TIES / DARE) và thử multi-LoRA. Đánh giá `quick400` K = 1 rồi K = 4 cho 2
ứng viên đầu. *Nghiệm thu:* bảng nhánh trong `EXPERIMENT_LOG.md`; QĐ7–QĐ9.

**WP7 — Huấn luyện đầy đủ + tích hợp (12–13/11, phatle1578, ~12 GPU-h).** Cấu hình thắng, dữ liệu đầy đủ, FSDP 4; gộp hoặc
adapter (QĐ3); cập nhật `src/model_hashes.json`; đăng trọng số / adapter vào dataset private. *Nghiệm thu:* `quick400` ≥ nhánh
thắng; s / câu đo trên 4×L4; QĐ10.

**WP8 — Tin cậy, bộ kiểm, truy hồi ngữ nghĩa, đọc kết quả (12–13/11) — P1.** `datacard.py` (M4), `review.py` (M11), tool T2 / T5
/ T6 / T9 / T4, `rulebook.py` (M3); DV + `verifier_train.py` (S3, quynhtran35 ~2 GPU-h); `embed_bank.py` + `retrieval.py` mode
`embed` (S1); `confidence.py` (đặc trưng, logistic, θ, conformal). *Test:* `test_confidence` (đơn điệu, ngưỡng, hồi quy trên log
có sẵn), `test_retrieval` (đánh giá offline: tỉ lệ top-4 cùng họ hàm), `test_verifier_format`. *Nghiệm thu:* QĐ11, QĐ13, QĐ14;
token / câu giảm ≥ 30 % nhờ dừng sớm mà PA held-out không giảm.

**WP9 — Vòng sửa lỗi tới public = 1.0 (10–14/11, lặp).** Sau mỗi lần đo: phân loại câu sai / abstain ở `pub_full`, `pub_k0`,
train (không xem held-out) theo 9 nhóm gốc lỗi (`docs/06` §5); sửa theo nhóm, thứ tự: toolkit / định nghĩa → precheck /
postcheck → gợi ý / tool → dữ liệu huấn luyện / ví dụ tổng hợp (DN train / tổng hợp, không câu public) → hướng dẫn prompt (ít).
Mỗi nhóm thêm test hồi quy từ train / tổng hợp. *Dừng khi:* câu sai còn lại chỉ sửa được bằng vá riêng câu hoặc là lỗi nhãn /
câu mơ hồ (ghi danh sách vào log).

**WP10 — Đánh giá cuối (14/11 sau 07:00 GMT+7, quota mới, ~25 GPU-h).** `pub_full` × 2 (tất định), `pub_ho`, `tr_ho`, `pub_noind`,
`pub_noacc`, `robust`, thời gian 1000 câu; so với v34 / v28; ghi `configs/flow_final.json`. *Nghiệm thu:* QĐ15.

**WP11 — Docker + nộp (15/11).** §5.11; `submissions/vNN/` + README; cập nhật `handoff/STATUS.md`. **User** build, chạy
`run_public_test.sh`, gửi dòng cuối `run.log` + thời gian; nộp trước 16/11 23:59 GMT+7.

**WP12 — Nhánh thoát DuckDB SQL (11–13/11, CPU + chung lần đo WP3 / WP8) — P1.** `<sql>` trong ngữ pháp, chạy DuckDB in-process
trên bảng C + bảng thô (read-only), prompt nêu tên bảng / cột; đo trên `pub_noind` so với nhánh pandas. *Nghiệm thu:* QĐ12.

**Thứ tự cắt nếu thiếu thời gian (cắt từ trên xuống):** P2 (JSON IR, on-policy, DSPy, PhoBERT, khôi phục dấu, draft) → WP12 →
multi-LoRA → T9 / T4 / M3 LLM → S3 bộ kiểm → M4 data card → WP6 vòng 2. **Không bao giờ cắt:** WP2 (P0), WP3 E0, WP6 vòng 1,
WP7, WP10, WP11.

## 9. Lịch và quota

| Ngày (GMT+7) | Container / CPU | yuan1103 (≈15.6) | quynhtran35 (30) | phatle1578 (30) |
|---|---|---|---|---|
| T2 09/11 | WP0; WP2; WP4 (D1, D2, D2N) | WP1 smoke (5) | — | — |
| T3 10/11 | WP2 xong + preflight; WP4 D3 lấy mẫu + chạy (CPU Kaggle) | WP3 E0 (8) | WP3 S0 + A/B (6) | — |
| T4 11/11 | WP9 vòng 1; WP4 D3 câu do 9B, D4T; WP12 code | — | WP5 dò thầy (5) | WP6 vòng 1, 2 phiên (16) |
| T5 12/11 | WP8 code; WP9 | — | D3 bằng thầy thắng (8); WP6 vòng 2 (8) | — |
| T6 13/11 | WP8 hiệu chỉnh; WP9 | (dư ~2) | S3 bộ kiểm (2) | WP7 huấn luyện đầy đủ (12) |
| T7 14/11 | WP9 vòng cuối | quota mới → WP10 (~25 tổng, chia 3 acc) | WP10 | WP10 |
| CN 15/11 | WP11; user build Docker | dự phòng | dự phòng | dự phòng |
| T2 16/11 | sửa sự cố; **user nộp trước 23:59** | — | — | — |

Tổng W46 ≈ 70 / 75.6 GPU-h; W47 ≈ 25–45 / 90. Phiên CPU Kaggle (5 / acc) dùng cho D3 bước chạy, lọc thực thi, chấm điểm.

## 10. Quy tắc quyết định (định sẵn)

| Mã | Khi | Quyết định |
|---|---|---|
| QĐ1 | WP1(a) | 1 GPU (Unsloth) VRAM đỉnh ≤ 21.5 GB ở chuỗi 2,048 → 4 nhánh / phiên; ngược lại FSDP 2 GPU → 2 nhánh / phiên; FSDP lỗi → LoRA trên trọng số 8-bit 1 GPU; tất cả lỗi → QLoRA 4-bit và đo chất lượng trước khi dùng |
| QĐ2 | WP1(c) | thầy FP8 khối không chạy trên L4 → bản bf16 `yiyangzheng/qwen3-6-27b` + `--quantization fp8`; gpt-oss-120b lỗi → 20b |
| QĐ3 | WP1(b) | adapter qua `--lora` chậm hơn gộp ≤ 10 % → dùng adapter (gọn, cho phép multi-LoRA); ngược lại gộp + FP8 |
| QĐ4 | WP3 | Gemma 4 12B-it thay Qwen3.5-9B chỉ khi `quick400` cao hơn ≥ 2.0 điểm **và** s / câu ≤ 1.5× |
| QĐ5 | WP3 | prompt gọn giữ khi PA mô hình gốc giảm ≤ 1.0 điểm so với prompt đầy đủ |
| QĐ6 | WP5 | 2 thầy có tỉ lệ giữ sau lọc cao nhất; hoà → tok/s cao hơn |
| QĐ7 | WP6 | nhánh được chọn khi hơn mô hình gốc ≥ 2.0 điểm trên `quick400` và không thua ở phần `tr_ho` / k=0; D6 giữ nếu A4 ≥ A3; D5 giữ nếu + ≥ 0.5 |
| QĐ8 | WP6 | (dự phòng) không nhánh nào đạt QĐ7 → nộp hybrid với mô hình gốc; vẫn làm WP8 |
| QĐ9 | WP6 | bản gộp (linear / TIES / DARE) hoặc multi-LoRA thay nhánh đơn khi ≥ +0.5 điểm `quick400` |
| QĐ10 | sau WP7 | ReST (D7) chỉ chạy nếu còn ≥ 1 ngày trước đóng băng và ≥ 20 GPU-h |
| QĐ11 | WP8 | truy hồi `embed` thay `ngram` khi tỉ lệ top-4 cùng họ hàm + ≥ 3 điểm (train trừ `tr_ho` + `pub_k0`) và PA `quick400` không giảm |
| QĐ12 | WP1(d) / WP3 / WP12 | ngữ pháp giữ khi tỉ lệ mẫu chạy được tăng và PA không giảm, giải mã chậm thêm ≤ 15 %; DuckDB giữ khi thắng pandas trên `pub_noind` |
| QĐ13 | WP8 | bộ kiểm S3 giữ khi AUROC trên dev ≥ 0.75 và PA `quick400` tăng ≥ 0.5 |
| QĐ14 | WP8 | θ_abstain = tối đa PA trên dev (train trừ `tr_ho` + `pub_full` + `pub_k0`); kiểm bằng conformal risk control: tỉ lệ sai trên câu trả lời ở held-out không vượt dev + 2 điểm; nếu vượt → nâng θ |
| QĐ15 | WP10 | bản nộp = held-out (`pub_ho` + `tr_ho`) cao nhất, với `pub_full` ≥ bản tốt nhất trước, thời gian ≤ ngân sách, \|ΔPA\| ≤ 0.01 |

## 11. Kiểm tra và chất lượng

- **Test CPU** (`tests/run_all.sh`): hiện có + `test_api_spec`, `test_grammar`, `test_hints`, `test_tools`, `test_precheck`,
  `test_flow_hybrid`, `test_data_factory`, `test_sft_format`, `test_confidence`, `test_retrieval`, `test_verifier_format`.
- **Preflight** cho mọi job agent / luồng; job huấn luyện / dò thầy: `--dry-run` trên CPU + review; thay đổi chỉ chạy trên
  Kaggle → chạy nhỏ trước (WP1).
- **Chấm:** `src/score.py`, `tests/score_sets.py`; log từng câu đủ để chấm lại offline.
- **Không rò rỉ:** `test_data_factory` kiểm không có id public / DN `dev_*` / DN `tr_ho` trong dữ liệu huấn luyện; `src/` không
  chứa đáp án public.
- **Tái lập:** seed cố định; phiên bản ghim; `training/README.md` + cấu hình + dữ liệu tổng hợp trong repo (hoặc dataset nếu lớn).
- **Sau mỗi WP:** cập nhật `handoff/STATUS.md` + `EXPERIMENT_LOG.md`, commit + push (container có thể khởi động lại).

## 12. Rủi ro

| Rủi ro | Cách giảm |
|---|---|
| Quota không đủ | nhánh song song, `quick400`, thứ tự cắt §8 |
| Ngữ pháp quá chặt / chậm | nhánh thoát `<python>`; nghiệm thu 100 % chương trình đã kiểm khớp; đo chậm thêm (QĐ12) |
| Fine-tune làm quên khả năng chung | D6 nhỏ; r thấp ở nhánh; cổng held-out |
| Quá khớp public | public không vào dữ liệu / prompt; cổng held-out |
| Thời gian suy luận vượt 6 h | dừng sớm, K thích ứng, ngữ pháp giảm lượt sửa, ngân sách + ghi dần |
| Không tất định | seed, nhiệt độ 0 ở mẫu đầu, schema tất định, bỏ phiếu + bộ kiểm |
| Container khởi động lại / tác vụ nền 2 h | `run_kernel.py --attach`; cập nhật STATUS sau mỗi WP |

## 13. Việc của user

Token Kaggle cho khung chat mới; `score_1.jsonl` nếu container mới không có; build + chạy thử Docker ngày 15/11; nộp trước
16/11 23:59 GMT+7.

## Nguồn

Xem §1 (kiểm mô hình / thư viện) và `docs/09_OPTIONS_DEEP_DIVE.md` (paper / repo cho tool, ngữ pháp, mô hình phụ, chưng cất,
chọn / tin cậy). Paper chính của kế hoạch (arXiv): ReAct 2210.03629; PoT 2211.12588; PAL 2211.10435; Self-Debug 2304.05128;
CodeAct 2402.01030; SWE-agent 2405.15793; PICARD 2109.05093; XGrammar 2411.15100; DIN-SQL 2304.11015; CHASE-SQL 2410.01943;
XiYan-SQL 2411.08599; DS-STAR 2509.21825; DAIL-SQL 2308.15363; Self-Consistency 2203.11171; Adaptive-Consistency 2305.11860;
LEVER 2302.08468; Training Verifiers 2110.14168; Conformal Risk Control 2208.02814; LoRA 2106.09685; STaR 2203.14465; ReST-EM
2312.06585; OmniSQL 2503.02240; Model soups 2203.05482; TIES 2306.01708; DARE 2311.03099; GKD 2306.13649; Qwen3 Embedding
2506.05176. Repos: vllm-project/vllm, mlc-ai/xgrammar, huggingface/trl, huggingface/peft, unslothai/unsloth, arcee-ai/mergekit,
QwenLM/Qwen3-Embedding, xingyaoww/code-act, SWE-agent/SWE-agent, duckdb/duckdb.
