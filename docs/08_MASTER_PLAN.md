# 08 — Kế hoạch tổng thể chi tiết: hybrid framework, mục tiêu public = 1.0, thời gian ngắn nhất

*Lập 08/11/2026 (bản chi tiết). Kế hoạch chính cho khung chat kế tiếp. Thiết kế module: `docs/06_HYBRID_FRAMEWORK.md`;
dataset ngoài: `docs/07_DATASETS.md`. Mọi mục "cần kiểm" của bản trước đã được kiểm ngày 08/11 (§1); ba điều chỉ đo được
trên GPU được gom vào **một lần chạy thử đầu tiên** (WP1) với quy tắc quyết định viết sẵn (§9) — không còn câu hỏi mở.*

## 0. Tóm tắt

1. **Hệ đích:** Agent (mô hình chính ≤ 15B, fine-tune LoRA) sinh code cho **mọi** câu. Luật tổng quát chỉ chuẩn bị ngữ cảnh
   (schema `automap`, làm sạch câu, gói gợi ý: kỳ thời gian, thực thể khớp data, kiểu đáp án, thuật ngữ rulebook; ví dụ đã
   kiểm). Code tính toán (toolkit `dm` / `industry`; pandas / DuckDB SQL dự phòng), kiểm tra trước khi chạy, Agent đọc kết
   quả, chọn theo phiếu + tin cậy hiệu chỉnh, sinh lại hoặc abstain.
2. **Thứ tự đòn bẩy:** (a) schema từ `automap` (bỏ lỗi ánh xạ — nguồn lỗi lớn nhất của v28); (b) prompt gọn + gợi ý + ví dụ;
   (c) fine-tune LoRA bf16 trên 2681 cặp đã kiểm + paraphrase + dữ liệu tổng hợp "chương trình trước, câu sau" do mô hình
   thầy viết câu; (d) chọn / kiểm / tin cậy + dừng sớm; (e) vòng sửa lỗi tổng quát trên public.
3. **Tìm cấu hình tốt nhất song song:** nhiều nhánh fine-tune cùng lúc (2 nhánh / phiên 4×L4, 2 phiên / acc), khác nhau về
   dữ liệu / mô hình thầy, cùng ngân sách token; đánh giá trên bộ nhanh 400 câu cố định; loại dần; nhánh thắng huấn luyện
   đầy đủ.
4. **Quota:** tuần W46 còn ≈ 75.6 GPU-giờ (quynhtran35 30, phatle1578 30, yuan1103 ≈ 15.6); từ 14/11 07:00 GMT+7 thêm 90.
   Kế hoạch dùng ≈ 68 (W46) + ≈ 45 (W47), dư phòng ≈ 50.
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
| `robust` | `tests/eval_all.sh` (đổi tên cột, README trắng, nhiễu) | phần schema / làm sạch |

**Cổng giữ một thay đổi:** `pub_ho` và `tr_ho` không giảm quá nhiễu (±0.5 điểm trên quick400; ±0.3 trên bộ đầy đủ); `pub_full`
và `pub_k0` tăng hoặc giữ; thời gian trong ngân sách; `tests/run_all.sh` ALL PASS; preflight OK.
**Thời gian đích:** ≤ 2 h / 1000 câu trên 2×RTX 5090 (≈ ≤ 6 h trên 4×L4); tất định |ΔPA| ≤ 0.01 giữa 2 lần chạy.

## 4. Kiến trúc đích

### 4.1 Module (M = mô hình, C = code / luật logic được phép)

| # | Module | Ai | Hiện có → việc cần làm |
|---|---|---|---|
| M1 | Đọc data | C | `ingest.py` — giữ |
| M2 | Schema chuẩn C | C (+M dự phòng) | `automap.py` mặc định; LLM ánh xạ (`--think-map`) + `canon.secondary_checks` + `consensus.py` khi cảnh báo nặng / thiếu bảng |
| M3 | Diễn giải rulebook | M + C | `changed_terms` → thêm: M đọc rulebook của split một lần → JSON định nghĩa; thuật ngữ mới / đổi → khối `# RULES CHANGED` trong prompt + cho phép pandas ghi đè hàm toolkit |
| M4 | Data card | M | mới: mỗi DN một lần, ≤ 300 token, cache |
| M5 | Làm sạch + gói gợi ý | C | `qnorm.py`; `src/hints.py` mới (§4.2) — **không** đưa mẫu / chương trình `qparse` |
| M6 | Truy hồi ví dụ | C | `fewshot.py` → che thêm thực thể / kỳ (khung câu), thêm ví dụ tổng hợp, loại DN `tr_ho` khi đo |
| M7 | Sinh chương trình | M | `agent.py` → prompt gọn (§4.3), 2–3 kiểu prompt, mô hình LoRA, đường dự phòng pandas / DuckDB SQL |
| M8 | Tool | M gọi, C làm | chỉ thêm nếu WP9 cho thấy thiếu thông tin giá trị: `lookup_values`, `describe`, `sample_rows` |
| M9 | Kiểm tra trước khi chạy | C | `src/precheck.py` mới: AST, hàm / tham số đúng chữ ký, thực thể / kỳ khớp gợi ý, kiểu trả về; vi phạm → Agent sửa |
| M10 | Chạy | C | sandbox `pipeline.py` — giữ |
| M11 | Đọc kết quả | M | mới: chỉ khi nghi ngờ (bất đồng, 0, rỗng, âm, bất thường, lỗi) → OK / SỬA / KHÔNG TRẢ LỜI ĐƯỢC |
| M12 | Chọn + tin cậy | C + M | `vote.py` → dừng sớm khi 2–3 mẫu đầu đồng ý và precheck sạch; tin cậy logistic (§7 WP8); dưới θ₁ sinh lại, dưới θ₂ abstain |
| M13 | Định dạng + ghi | C | `fmt_answer`, `merge_sub.py` (append + flush, gộp mỗi 20 s) — giữ |

### 4.2 Gói gợi ý (`src/hints.py`) — đặc tả

```
{"period": {"text": "tháng 7 năm ngoái", "code": "month(2025, 7)", "from": "2025-07-01", "to": "2025-07-31"},
 "entities": [{"text": "cá tràu", "kind": "item", "data_value": "Cá lóc", "how": "regional"}],
 "answer_type": "number", "rule_terms": [{"term": "Khách sộp nhất", "definition": "..."}],
 "missing_data": ["không có bảng tồn kho"], "measure_words": ["doanh thu"]}
```
Chỉ chứa những gì luật tổng quát nhận diện được (kỳ, thực thể, kiểu, thuật ngữ, dữ liệu thiếu, từ chỉ đại lượng). Không
chứa tên hàm toolkit, mẫu câu hay chương trình — Agent tự chọn hàm.

### 4.3 Prompt gọn (dùng chung cho suy luận và huấn luyện, ~2k token)

Thứ tự (phần chung trước để dùng prefix cache): `[SYSTEM]` vai trò + tham chiếu API toolkit rút gọn + định dạng đầu ra →
`[BUSINESS]` card schema chuẩn rút gọn (≤ 1,000 token: bảng / cột có thật + 2 giá trị ví dụ) + data card (≤ 300) + thay đổi
rulebook → `[EXAMPLES]` 2–4 ví dụ đã kiểm → `[HINTS]` → `[QUESTION]`.
Đầu ra của mô hình (không suy nghĩ dài):
```
<plan>một câu: đại lượng, kỳ, bộ lọc</plan>
<code>answer = revenue(month(2025, 7), item="Cá lóc")</code>
<conf>0.85</conf>
```
Prompt dài hiện tại (rulebook + README + card đầy đủ, `--max-len 24576`) chỉ giữ cho đường dự phòng (M2 LLM ánh xạ, M8).

## 5. Mô hình

### 5.1 Mô hình chính (≤ 15B lúc chạy)

| Ứng viên | Tham số | Ưu / nhược |
|---|---|---|
| **Qwen3.5-9B** (mặc định) | 9.65B | hạ tầng sẵn, sha256 đã kiểm, vLLM có LoRA; còn ~5.3B cho mô hình phụ |
| Gemma 4 12B-it | 11.96B | mới hơn (05/2026), 140+ ngôn ngữ; vLLM 0.32 chạy được nhưng không LoRA serving → gộp trọng số; còn ~3B |
| Qwen3.5-4B | 4.66B | nhanh; dùng nếu thời gian suy luận vượt ngân sách |

Chọn bằng thí nghiệm S0 (WP3) theo quy tắc QĐ4 (§9).

### 5.2 Mô hình thầy (chỉ dùng offline để tạo dữ liệu; không tính vào 15B)

| Thầy | Nguồn Kaggle (đã đối chiếu HF) | Chạy trên 4×L4 | Vai trò |
|---|---|---|---|
| Qwen3.6-27B-FP8 | nsalzman/qwen3-6-27b-fp8/transformers/fp8/6 | TP 2 × 2 bản (15.5 GB / GPU + KV) | viết câu tiếng Việt cho chương trình; paraphrase; CoT |
| Qwen3.6-35B-A3B-FP8 | zerabytex/qwen-3-6-35b-a3b-fp8/transformers/default/1 | TP 4 (9.4 GB / GPU) | như trên, nhanh (3B hoạt động) |
| Gemma 4 31B-it | google/gemma-4/transformers/gemma-4-31b-it/1 (`--quantization fp8`) hoặc .../other/gemma-4-31b-it-qat-w4a16-ct/2 | TP 2 × 2 bản | đa dạng văn phong (họ mô hình khác) |
| gpt-oss-120b | danielhanchen/gpt-oss-120b/transformers/default/1 (MXFP4, Marlin trên Ada) | TP 4 (16.3 GB / GPU) | suy luận mạnh; tiếng Việt cần đo |
| Qwen3.5-9B (tự chưng cất) | sẵn có | DP 4 | ReST, lọc vòng lặp ngược |

### 5.3 Ngăn xếp huấn luyện (cố định)

torch 2.13.0 (từ wheels vLLM 0.32), transformers 5.17.0, peft 0.21.2, trl 1.15.0, accelerate 1.15.0, datasets 4.x; tuỳ chọn
unsloth 2026.11.3 + unsloth_zoo + xformers bản cho torch 2.13. Wheels tải trên container (`pip download`) → dataset private
`<acc>/zc-train-wheels` (mỗi acc L4 một bản). LoRA bf16, `target_modules="all-linear"` trong phần ngôn ngữ (không vision,
không `lm_head`), r 16 / 64, alpha = 2r, lr 1e-4 – 2e-4, cosine, 1–2 epoch, gradient checkpointing, chuỗi ≤ 2,048 token,
chỉ tính loss trên phần đầu ra. Phân tán: FSDP qua `accelerate` (2 GPU / nhánh hoặc 4 GPU cho bản cuối) — theo QĐ1.

## 6. Dữ liệu huấn luyện

| Mã | Dữ liệu | Quy mô (ước) | Nhãn | Script |
|---|---|---|---|---|
| D1 | câu train → chương trình đã kiểm (bỏ 6 DN `tr_ho`) | ~2,090 | `answer_examples.jsonl` | training/data/d1_train_pairs.py |
| D2 | paraphrase câu train (bỏ `tr_ho`) | ~4,400 | chương trình câu gốc, giữ khi lọc vòng lặp ngược đúng | `training/data/d2_paraphrase.py` |
| D3 | tổng hợp "chương trình trước, câu sau" trên 24 DN train | 10–20k | toolkit chạy chương trình | training/data/d3_sample_programs.py, `d3_teacher_questions.py`, `d3_roundtrip.py` |
| D5 | lời giải ngắn của thầy cho D1–D3 | = D1–D3 | giữ khi chương trình đi kèm chạy đúng | training/data/d5_rationales.py |
| D6 | ngoài (`docs/07` §4): ViText2SQL, TableBench-TableInstruct, BIRD train | ≤ 10 % hỗn hợp | nhãn của dataset | `training/data/d6_external.py` |
| D7 | ReST: mẫu của chính mô hình trên D1–D3, giữ cái chạy đúng | tuỳ | thực thi | `training/data/d7_rest.py` |
| D4 (tuỳ) | DN tổng hợp ngành mới (bộ sinh data riêng) | 3–6 DN | bộ sinh / pandas đã kiểm | training/data/d4_synth_business.py |
| **Cấm** | câu / data / paraphrase của public test, private test | — | — | — |

Dùng logic luật (bộ sinh chương trình, toolkit) để **tạo dữ liệu huấn luyện offline** là hard code logic theo ranh giới user
đã chốt (logic hỗ trợ Agent; không tham gia trả lời lúc chạy).

**D3 chi tiết:** (1) lấy mẫu chương trình từ ngữ pháp toolkit (hàm × tham số × kỳ × thực thể thật) theo phân bố kiểu câu
của train; (2) chạy, loại kết quả suy biến (rỗng, toàn 0, lỗi); (3) thầy viết 2 câu / chương trình với văn phong khác nhau
(trang trọng / chat, có / không dấu, viết tắt, từ địa phương, hỏi gián tiếp) kèm data card để dùng tên thật; (4) lọc vòng
lặp ngược: mô hình học trò hiện tại đọc câu → viết chương trình → chạy; giữ khi đáp án trùng; (5) ghi tỉ lệ giữ của từng thầy.

## 7. Gói việc (WBS) — sản phẩm, bước, kiểm tra, nghiệm thu

**WP0 — Chuẩn bị (09/11, CPU, 1 h).** Xin token; tải `nhatlong1103/zc-2026`; nhận `score_1.jsonl` → `data/raw/public_gt.jsonl`;
ingest cache train / public; tải `<acc>/zc-para`; tạo `tests/splits/{pub_full,pub_k0,pub_ho,tr_ho,quick400}.txt` (seed
20261108). *Nghiệm thu:* `S=… bash tests/eval_all.sh quick` in 3000 / 3000, 1000 / 1000 (luật vẫn đúng, kiểm dữ liệu).

**WP1 — Chạy thử GPU (09/11, yuan1103, 1 phiên ~1.2 h ≈ 5 GPU-h).** `automation/jobs/smoke_v35.jobs` gồm 3 phần:
(a) huấn luyện LoRA bf16 9B 30 bước trên 200 mẫu D1, đo VRAM đỉnh và tok/s cho: 1 GPU (Unsloth), FSDP 2 GPU, FSDP 4 GPU;
in tên module LoRA; (b) gộp LoRA → nạp bằng vLLM `--quantization fp8` → sinh 20 câu; đồng thời nạp adapter qua `--lora`
→ so tốc độ; (c) nạp Qwen3.6-27B-FP8 (TP 2) và Gemma 4 31B-it (fp8 động, TP 2), sinh 50 câu mỗi thầy, đo tok/s.
*Kiểm trước khi đẩy:* script chạy được trên CPU ở chế độ khô (`--dry-run`: dựng dữ liệu, cấu hình, không nạp mô hình);
self-review; kernel `--no-preflight` (không chạy agent). *Nghiệm thu:* số đo ghi vào `EXPERIMENT_LOG.md`; áp QĐ1–QĐ3.

**WP2 — Luồng hybrid v1 (09–10/11, CPU).** `src/hints.py` (M5), `src/precheck.py` (M9), prompt gọn (§4.3) trong `agent.py`
sau cờ `--prompt compact|full`, `--hints 0|1`; `src/predict_flow.py` viết lại: bỏ stage 1 trả lời thẳng và stage 1.5;
schema `automap` → per-câu Agent → precheck → chạy → chọn; cờ `--qw 0`, không `--confirm`.
*Test CPU mới:* `tests/test_hints.py` (kỳ / thực thể / kiểu trên 200 câu train có nhãn → độ chính xác kỳ ≥ 99 %, thực thể
≥ 98 %; không chứa tên hàm), `tests/test_precheck.py` (phát hiện hàm sai, tham số sai, thực thể lạ, kỳ lệch; 0 báo nhầm
trên 2681 chương trình đã kiểm), `tests/test_flow_hybrid.py` (mô hình giả: mọi câu đi qua mô hình, không câu nào lấy đáp án
luật; ghi dần; hết giờ giữ đáp án đã có). Cập nhật `test_rules_first.py` / `test_canon_flow.py` (chỉ còn kiểm thư viện).
*Nghiệm thu:* `tests/run_all.sh` ALL PASS; preflight OK với `automation/jobs/v35.jobs`.

**WP3 — Mốc E0 + chọn mô hình chính S0 (10/11, yuan1103 + quynhtran35, ~10 GPU-h).** E0: luồng hybrid v1 với Qwen3.5-9B
trên `quick400` + `pub_full`; S0: cùng luồng với Gemma 4 12B-it trên `quick400`; biến thể prompt `compact` vs `full` trên
`quick400` (QĐ5). *Nghiệm thu:* bảng PA / thời gian / token theo bộ; áp QĐ4, QĐ5.

**WP4 — Nhà máy dữ liệu (09–11/11, container + phiên CPU Kaggle).** Script §6; định dạng SFT JSONL `{messages: [system,
user, assistant]}` theo đúng prompt gọn. *Test:* `tests/test_data_factory.py` — mỗi mẫu chạy lại ra đúng nhãn; không có id
câu public / DN `dev_*`; không có DN `tr_ho`; phân bố kiểu đáp án lệch ≤ 5 điểm % so với train; câu trùng lặp < 1 %.
*Nghiệm thu:* `training/data/out/{d1,d2,d3_self}.jsonl` + thống kê.

**WP5 — Dò thầy (11/11, quynhtran35, ~5 GPU-h).** 1,000 chương trình D3 + 300 câu train cần paraphrase, giống nhau cho mọi
thầy (§5.2). Đo: tỉ lệ giữ sau lọc vòng lặp ngược, độ đa dạng (distinct-2), tỉ lệ câu có dấu tiếng Việt chuẩn, tok/s.
*Nghiệm thu:* xếp hạng thầy; chọn 2 thầy cho vòng 2 (QĐ6).

**WP6 — Nhánh fine-tune (11–12/11).** Cùng ngân sách ~6M token / nhánh, đánh giá `quick400` K = 1 (tham lam) rồi K = 4 cho 2
nhánh đầu.
- Vòng 1 (phatle1578, 2 phiên song song, ~16 GPU-h): A1 = D1; A2 = D1 + D2; A3 = D1 + D2 + D3 (câu do 9B viết);
  A4 = A3 + D6 (10 %).
- Vòng 2 (quynhtran35, ~8–16 GPU-h): hỗn hợp tốt nhất với D3 do thầy 1 / thầy 2 viết (+ một nhánh trộn nếu còn quota);
  + / − D5 (lời giải).
*Nghiệm thu:* bảng nhánh trong `EXPERIMENT_LOG.md` (dữ liệu, thầy, siêu tham số, thời gian, PA từng bộ); áp QĐ7.

**WP7 — Huấn luyện đầy đủ + tích hợp (12–13/11, phatle1578, ~12 GPU-h).** Nhánh thắng, dữ liệu đầy đủ, FSDP 4 GPU; gộp hoặc
adapter theo QĐ3; cập nhật `src/model_hashes.json`, `--lora` / đường dẫn trọng số trong luồng; đăng trọng số / adapter lên
dataset private. *Nghiệm thu:* `quick400` ≥ nhánh thắng; thời gian / câu đo trên 4×L4.

**WP8 — Tin cậy, dừng sớm, đọc kết quả (12–13/11, CPU + chạy chung với WP7).** M11, M12: đặc trưng (tỉ lệ phiếu, số đáp án
khác nhau, cờ precheck, lỗi chạy, log-prob, `<conf>`, phán quyết M11); logistic fit trên `pub_full` + `pub_k0` + train (trừ
`tr_ho`); θ₁ / θ₂ chọn tối đa PA trên dev; dừng sớm khi 2 mẫu đầu đồng ý + precheck sạch + tin cậy ≥ θ₁.
*Test:* `tests/test_confidence.py` (đơn điệu, ngưỡng, hồi quy trên log đã có). *Nghiệm thu:* PA `pub_ho` / `tr_ho` không giảm,
token / câu giảm ≥ 30 %.

**WP9 — Vòng sửa lỗi tới public = 1.0 (10–14/11, CPU, lặp).** Sau mỗi lần đo: phân loại câu sai / abstain ở `pub_full`,
`pub_k0`, train (không xem held-out) theo 9 nhóm gốc lỗi (`docs/06` §5); sửa theo nhóm, thứ tự: toolkit / định nghĩa →
precheck / kiểm kết quả → gợi ý → dữ liệu huấn luyện / ví dụ tổng hợp bao phủ nhóm (trên DN train / tổng hợp) → hướng dẫn
prompt (ít) → tool. Mỗi nhóm thêm test hồi quy từ train / tổng hợp. *Dừng khi:* câu sai còn lại chỉ sửa được bằng vá riêng
câu, hoặc là lỗi nhãn / câu mơ hồ (ghi danh sách vào log).

**WP10 — Đánh giá cuối (14/11 sau 07:00, quota mới, ~25 GPU-h).** `pub_full` × 2 lần (tất định), `pub_ho`, `tr_ho`, `robust`,
đo thời gian 1000 câu; so với v34 / v28. *Nghiệm thu:* QĐ10.

**WP11 — Docker + nộp (15/11).** `docker/fetch_assets.sh` (trọng số / adapter + sha256), `docker/Dockerfile` (thêm
`duckdb` nếu dùng), `docker/predict.sh` (luồng mới), `docker/README_SOLUTION.md` (mô hình, dataset ngoài, code train
`training/`), `submissions/vNN/` + README; cập nhật `handoff/STATUS.md`. **User** build, chạy `run_public_test.sh`, gửi dòng cuối
`run.log` + thời gian; nộp trước 16/11 23:59 GMT+7.

## 8. Lịch và quota

| Ngày (GMT+7) | Container / CPU | yuan1103 (≈15.6) | quynhtran35 (30) | phatle1578 (30) |
|---|---|---|---|---|
| CN 08/11 | kế hoạch (xong) | — | — | — |
| T2 09/11 | WP0, WP2, WP4 (D1 / D2) | WP1 smoke (5) | — | — |
| T3 10/11 | WP2 xong + preflight; WP4 D3 bước 1–2 | WP3 E0 (8) | WP3 S0 + prompt (6) | — |
| T4 11/11 | WP9 vòng 1 (từ E0); WP4 D3 (câu do 9B) | — | WP5 dò thầy (5) | WP6 vòng 1, 2 phiên (16) |
| T5 12/11 | WP8 code; WP9 | — | D3 bằng thầy thắng (8); WP6 vòng 2 (8) | — |
| T6 13/11 | WP8 hiệu chỉnh; WP9 | — | (dư 3) | WP7 huấn luyện đầy đủ (12) |
| T7 14/11 | WP9 vòng cuối | quota mới: WP10 (đến ~25 / 90 mỗi acc) | WP10 | WP10 |
| CN 15/11 | WP11; user build Docker | dự phòng | dự phòng | dự phòng |
| T2 16/11 | sửa sự cố; **user nộp trước 23:59** | — | — | — |

Tổng W46 ≈ 68 / 75.6 GPU-h; W47 ≈ 25–45 / 90. Phiên CPU Kaggle (5 / acc, không tính quota GPU) dùng cho D3 bước 2, lọc
thực thi, chấm điểm.

## 9. Quy tắc quyết định (định sẵn cho mọi điểm chỉ đo được khi chạy)

| Mã | Khi | Quyết định |
|---|---|---|
| QĐ1 | WP1(a) | 1 GPU (Unsloth) VRAM đỉnh ≤ 21.5 GB ở chuỗi 2,048 → 4 nhánh / phiên; ngược lại FSDP 2 GPU → 2 nhánh / phiên; FSDP lỗi → LoRA trên trọng số 8-bit (bitsandbytes) 1 GPU; tất cả lỗi → QLoRA 4-bit và đo chất lượng trước khi dùng |
| QĐ2 | WP1(c) | thầy FP8 khối (Qwen3.6) không chạy trên L4 → dùng bản bf16 `yiyangzheng/qwen3-6-27b` + `--quantization fp8`; gpt-oss-120b MXFP4 lỗi → gpt-oss-20b |
| QĐ3 | WP1(b) | adapter qua vLLM `--lora` chậm hơn gộp trọng số ≤ 10 % → nộp adapter (gọn); ngược lại gộp + FP8 |
| QĐ4 | WP3 | Gemma 4 12B-it thay Qwen3.5-9B chỉ khi PA `quick400` cao hơn ≥ 2.0 điểm **và** thời gian / câu ≤ 1.5× |
| QĐ5 | WP3 | prompt gọn giữ khi PA mô hình gốc giảm ≤ 1.0 điểm so với prompt đầy đủ (fine-tune dùng prompt gọn) |
| QĐ6 | WP5 | chọn 2 thầy có tỉ lệ giữ sau lọc cao nhất; hoà → tok/s cao hơn |
| QĐ7 | WP6 | nhánh được chọn khi hơn mô hình gốc ≥ 2.0 điểm trên `quick400` và không thua ở phần `tr_ho` / k=0; D6 chỉ giữ nếu A4 ≥ A3; D5 chỉ giữ nếu + ≥ 0.5 điểm |
| QĐ8 | WP7 | fine-tune đầy đủ thua mô hình gốc trên held-out → nộp luồng hybrid với mô hình gốc |
| QĐ9 | sau WP7 | ReST (D7) chỉ chạy nếu còn ≥ 1 ngày trước đóng băng và ≥ 20 GPU-h |
| QĐ10 | WP10 | bản nộp = held-out (`pub_ho` + `tr_ho`) cao nhất, với `pub_full` ≥ bản tốt nhất trước, thời gian ≤ ngân sách, |ΔPA| ≤ 0.01 |

## 10. Kiểm tra và chất lượng (tổng hợp)

- **Test CPU** (`tests/run_all.sh`): hiện có + mới `test_hints`, `test_precheck`, `test_flow_hybrid`, `test_data_factory`,
  `test_confidence`, `test_sft_format` (mẫu SFT render đúng prompt gọn, nhãn chỉ ở phần assistant).
- **Preflight** (`automation/preflight.py`) cho mọi job agent; job huấn luyện / dò thầy: chế độ `--dry-run` trên CPU + review;
  thay đổi phần chỉ chạy trên Kaggle → chạy nhỏ trước (WP1).
- **Chấm:** `src/score.py` (luật chấm BTC), `tests/eval_split.py`; mỗi câu lưu log đầy đủ → chấm lại / đổi ngưỡng offline.
- **Không rò rỉ:** script kiểm tự động trong `test_data_factory` (không id public, không DN `dev_*`, không DN `tr_ho`);
  grep `src/` không chứa đáp án public.
- **Tái lập:** seed cố định; phiên bản thư viện ghim (§5.3); cấu hình huấn luyện + dữ liệu tổng hợp nằm trong `training/`.

## 11. Rủi ro

| Rủi ro | Cách giảm |
|---|---|
| Quota không đủ | nhánh song song, `quick400`, cắt vòng 2 / D4 / ReST theo QĐ9 |
| Fine-tune làm quên khả năng chung | trộn D6 nhỏ; r thấp; cổng held-out mỗi nhánh |
| Quá khớp public | public không vào dữ liệu / prompt; cổng held-out |
| Thời gian suy luận vượt 6 h (private lớn) | đầu ra ngắn, dừng sớm, K thích ứng, ngân sách thời gian + ghi dần |
| Không tất định | seed, nhiệt độ thấp ở bước chọn, schema tất định (automap), bỏ phiếu |
| Container khởi động lại / tác vụ nền 2 h | `run_kernel.py --attach`; cập nhật `handoff/STATUS.md` sau mỗi WP |

## 12. Việc của user

Token Kaggle cho khung chat mới; `score_1.jsonl` nếu container mới không có; build + chạy thử Docker ngày 15/11; nộp trước
16/11 23:59 GMT+7.

## Nguồn

HF API (`huggingface.co/api/models/...`, 08/11/2026) cho ngày tạo / giấy phép / số tham số; Kaggle API (`kaggle models
instances versions files`) cho nguồn mô hình; PyPI JSON cho phiên bản thư viện; docs vLLM v0.32.0 (supported models,
quantization); docs Unsloth Qwen3.5 fine-tune (https://unsloth.ai/docs/models/qwen3.5/fine-tune); Qwen3.6
(https://qwen.ai/blog?id=qwen3.6-27b); Gemma 4 (https://www.theregister.com/2026/04/02/googles_gemma_4_open_weights/).
Papers (arXiv): ReAct 2210.03629; PoT 2211.12588; PAL 2211.10435; Binder 2210.02875; Self-Debug 2304.05128; DIN-SQL 2304.11015;
MAC-SQL 2312.11242; CHESS 2405.16755; CHASE-SQL 2410.01943; XiYan-SQL 2411.08599; ReFoRCE 2502.00675; DS-STAR 2509.21825;
Data Interpreter 2402.18679; TaskWeaver 2311.17541; DAIL-SQL 2308.15363; Good ICE 2101.06804; RESDSQL 2302.05965; CodeS
2402.16347; Self-Consistency 2203.11171; Adaptive-Consistency 2305.11860; ESC 2401.10480; LEVER 2302.08468; CodeT 2207.10397;
Kadavath 2207.05221; Calibration 2305.14975; TrustSQL 2403.15879; Selective prediction 2607.06799; LoRA 2106.09685; QLoRA
2305.14314; STaR 2203.14465; ReST-EM 2312.06585; Distilling Step-by-Step 2305.02301; OmniSQL 2503.02240; Arctic-Text2SQL-R1
2505.20315; SQL-R1 2504.08600; DeepSeekMath 2402.03300; LlamaFactory 2403.13372; vLLM 2309.06180; PhoBERT 2003.00744.
Repos: vanna-ai/vanna, Canner/WrenAI, sinaptik-ai/pandas-ai, eosphoros-ai/DB-GPT, microsoft/TaskWeaver, FoundationAgents/MetaGPT,
XGenerationLab/XiYan-SQL, RUCKBReasoning/OmniSQL, ShayanTalaei/CHESS, BeachWang/DAIL-SQL, Snowflake-Labs/ReFoRCE,
snowflakedb/ArcticTraining, xlang-ai/Binder, hiyouga/LlamaFactory, unslothai/unsloth, huggingface/trl, verl-project/verl,
vllm-project/vllm, VinAIResearch/PhoBERT.
