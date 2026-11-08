# 09 — Đào sâu: tool cho Agent, mô hình phụ, chưng cất và các phương án cải tiến pipeline

*Lập 08/11/2026 theo yêu cầu user. Mỗi phương án gắn với một **lỗi đo được** của hệ hiện tại (§0), có bằng chứng (paper /
repo; citation Semantic Scholar, stars GitHub tra 08/11/2026), chi phí, cách thử và quy tắc giữ / bỏ. Phần được chọn đã
đưa vào `docs/08_MASTER_PLAN.md` (§5 đặc tả, §8 gói việc, §10 quy tắc quyết định).*

## 0. Căn cứ: mô hình đang sai ở đâu (log v34, 3 lần chạy × 422 câu khó, Qwen3.5-9B chưa fine-tune)

**Mẫu chương trình (11,260 mẫu) — 35 % không chạy được:**

| Loại lỗi | Số mẫu | Nghĩa | Phương án xử lý |
|---|---|---|---|
| TypeError / AttributeError | 1,019 / 670 | gọi sai kiểu, sai phương thức trên kết quả toolkit | T8 `help`, ngữ pháp có kiểu (§2), fine-tune |
| không có khối code / SyntaxError / IndentationError / không gán ANSWER | 626 / 183 / 31 / 104 | sai định dạng đầu ra | sinh có ràng buộc ngữ pháp (§2) — loại bỏ hoàn toàn |
| ImportError / ModuleNotFound / NameError | 230 / 126 / 216 | bịa module / hàm | ngữ pháp chỉ cho tên hàm có thật (§2), T8 |
| LookupError | 394 | tên hàng / khách không có trong data | T1 `find`, liệt kê thực thể trong ngữ pháp |
| KeyError / ValueError / IndexError | 269 / 158 / 15 | sai tên cột, thao tác pandas sai | T5 `schema` / `peek`, T2 `values` |

**Kết quả từng câu (lần chạy 1, 422 câu):** đúng 188; **sai mà các mẫu bất đồng 118**; **sai mà mọi mẫu đồng ý 63**
(hiểu sai có hệ thống); abstain 53.
→ Ba đòn bẩy: (1) loại mẫu hỏng → nhiều phiếu hợp lệ hơn (§2, §1); (2) chọn tốt hơn / abstain đúng chỗ cho nhóm
"bất đồng" (§4 bộ kiểm, §5.4 tin cậy); (3) sửa hiểu sai có hệ thống bằng dữ liệu huấn luyện + kiểm kết quả (§3, §1 T6).

## 1. Tool cho Agent — chỉ những tool thật sự cần

**Nguyên tắc (SWE-agent, NeurIPS 2024, 2,427 cit., 20.8k★ — "Agent-Computer Interface"):** ít tool, đầu ra ngắn và có
thông tin, có rào chắn; **CodeAct** (ICML 2024, 767 cit., 1.7k★; smolagents CodeAgent 30.3k★): tool là **hàm Python gọi
ngay trong chương trình**, không cần định dạng JSON riêng — hợp với toolkit hiện có. Tool là code tất định (logic được
phép); mô hình quyết định gọi gì.

| Tool | Chữ ký | Đầu ra (≤ 300 token) | Lỗi nhắm tới | Cách dùng |
|---|---|---|---|---|
| T1 `find` | `find(kind, text, k=5)`; kind ∈ item / customer / staff / supplier / vehicle | `[(giá trị thật, kiểu khớp: exact / không dấu / địa phương / viết tắt / gõ sai / ngữ nghĩa, điểm, số dòng)]` | LookupError 394, tên địa phương, viết tắt | trong chương trình (`item=find("item","cá tràu")[0]`) và ở lượt sửa |
| T2 `values` | `values(table, column, contains=None, k=20)` | giá trị khác nhau + số lần | lọc theo kênh / hình thức trả / trạng thái / nhóm sai giá trị | lượt khám phá / sửa |
| T3 `period` | `period("tuần trước")` | khoảng ngày theo lịch + rulebook (tuần, quý, Tết, "đầu tháng đến nay") | kỳ sai | trong chương trình; mô hình tự trích cụm thời gian |
| T4 `rule` | `rule("khách sộp nhất")` | định nghĩa trong rulebook của split + hàm toolkit tương ứng (nếu có) + tham số | rulebook đổi giữa các split | khi câu có thuật ngữ |
| T5 `schema` / `peek` | `schema(table=None)`, `peek(table, n=3, where=None)` | cột, kiểu, % trống, ví dụ | KeyError 269, cột sai | lượt khám phá / sửa |
| T6 `explain` | chạy chương trình kèm chẩn đoán | số dòng còn lại sau từng bộ lọc, bộ lọc khớp 0 dòng, trùng lặp, đơn vị | sai có hệ thống (63 câu), kết quả 0 / rỗng | bước đọc kết quả (M11) |
| T7 `check` | `check(answer, expected_type)` | hợp lệ định dạng + bất biến (phần trăm hợp lý, số đếm ≥ 0, top-N đủ N, ngày trong khoảng data) | sai định dạng, kết quả vô lý | tự động sau khi chạy (postcheck) |
| T8 `help` | `help("revenue")` | chữ ký + 1 ví dụ đã kiểm (kiểu Gorilla: tài liệu API theo yêu cầu) | TypeError / AttributeError / NameError (~2,100 mẫu) | lượt sửa sau lỗi; tài liệu rút gọn trong prompt |
| T9 `has_data` | `has_data("tồn kho")` | có / không ghi dữ liệu này + bảng nguồn | quyết định `unanswerable` | khi câu hỏi về dữ liệu có thể không ghi |

**Không làm:** máy tính (code đã tính), tìm web (offline), tool SQL riêng (gộp vào §6 phương án C).

**Giao thức gọi (giữ nhanh):** đường nhanh = 1 lượt (gợi ý đã tính sẵn + tài liệu API gọn); chỉ khi precheck / chạy lỗi
hoặc tin cậy thấp mới mở **tối đa 2 lượt khám phá** (CodeAct: mô hình viết `print(values(...))`, đọc kết quả, viết lại).
Đầu ra tool cắt ≤ 300 token. Huấn luyện cách dùng tool bằng quỹ đạo tổng hợp (lỗi → gọi tool → sửa), theo Toolformer
(NeurIPS 2023, 6,575 cit.) / ToolLLM (ICLR 2024, 2,601 cit., ToolBench 5.8k★) / Gorilla (NeurIPS 2024, 1,862 cit., 13.1k★).

**Cách thử:** bật / tắt từng tool trên `quick400`; đo tỉ lệ mẫu chạy được, PA, số lượt / câu, thời gian. Giữ tool khi PA
tăng hoặc thời gian giảm mà PA không giảm.

## 2. Sinh có ràng buộc + biểu diễn trung gian (sửa tận gốc lỗi định dạng / API)

**2.1 Ngữ pháp cho DSL toolkit (khuyến nghị làm trước).** vLLM 0.32 hỗ trợ `StructuredOutputsParams(grammar=<EBNF>)` với
backend xgrammar / guidance (đã kiểm docs v0.32.0). Ngữ pháp mô tả đúng dạng
`ANSWER = {"type": <kiểu>, "answer": <biểu thức>}`, biểu thức chỉ gồm hàm toolkit có thật với tham số đúng tên / kiểu;
literal thực thể lấy từ **danh sách giá trị thật của DN** (kênh, hình thức trả, trạng thái, nhân viên, nhóm; tên hàng nếu
≤ ~5k), ngữ pháp dựng một lần mỗi DN và cache. Nhánh thoát `<python>` không ràng buộc cho trường hợp toolkit không đủ
(đánh dấu vào tin cậy).
Bằng chứng: PICARD (EMNLP 2021, 790 cit.) — ràng buộc giải mã tăng mạnh độ chính xác text-to-SQL; Grammar-Constrained
Decoding (EMNLP 2023, 366 cit.); SynCode (TMLR 2024); XGrammar (2024, 117 cit., 2.0k★; backend chính của vLLM); Outlines
(16.0k★), guidance (21.8k★).
Hiệu quả dự kiến: loại ~1,500 mẫu lỗi định dạng / tên (13 % mẫu) và một phần ~1,700 mẫu sai kiểu → nhiều phiếu hợp lệ hơn,
ít lượt sửa hơn (nhanh hơn). Rủi ro: ngữ pháp quá chặt → giữ nhánh thoát; đo thời gian dựng ngữ pháp.

**2.2 Biểu diễn trung gian JSON (semantic layer) — phương án thay thế, thử A/B.** Mô hình sinh "truy vấn" dạng JSON:
`{measure, agg, filters:[{field, op, value}], period, group_by, order_by, limit, compare, answer_type}`; code biên dịch thành
lời gọi toolkit. Bằng chứng: IRNet / SemQL (ACL 2019, 539 cit.), NatSQL (Findings EMNLP 2021, 162 cit.); semantic layer
trong công nghiệp: Cube (21.2k★), dbt MetricFlow (1.9k★), WrenAI MDL (18.1k★). Ưu: không gian đầu ra nhỏ, ràng buộc bằng JSON
schema, tổng hợp dữ liệu dễ, dễ kiểm. Nhược: không phủ hết câu (cần nhánh Python). Quyết định bằng A/B trên `quick400`
(2.1 vs 2.2).

## 3. Mô hình phụ (sub-models) — trong trần 15B

| Mã | Mô hình (đã kiểm) | Tham số | Vai trò | Huấn luyện | Lợi dự kiến |
|---|---|---|---|---|---|
| S1 | **Qwen3-Embedding 0.6B** (Apache-2.0, 06/2025; có trên Kaggle: qwen-lm / qwen-3-embedding / 0.6b) hoặc BGE-M3 (MIT, 2024) | 0.60B | truy hồi ví dụ theo **nghĩa** (khung câu đã che thực thể / số — DAIL-SQL); liên kết thực thể ngữ nghĩa cho T1 | không cần; tuỳ chọn tinh chỉnh tương phản (câu ↔ khung chương trình) | **cao** cho paraphrase (điểm yếu lớn nhất); Qwen3 Embedding 1,884 cit., 2.1k★; M3-Embedding (ACL Findings 2024) 2,184 cit., FlagEmbedding 12.3k★ |
| S2 | Qwen3-Reranker 0.6B (Apache-2.0) | 0.60B | xếp lại ví dụ / ứng viên thực thể | không | thấp–vừa |
| S3 | **Bộ kiểm (verifier)** Qwen3.5 0.8B (Apache-2.0, 02/2026, 0.87B) hoặc Qwen3.5 2B (2.27B) | 0.87–2.27B | chấm P(đúng \| câu, chương trình, tóm tắt kết quả) → xếp hạng ứng viên + tín hiệu abstain | nhãn từ thực thi: lấy K mẫu của mô hình chính trên train (trừ `tr_ho`) + D3 → đúng / sai | **cao** cho 118 câu "bất đồng"; Training Verifiers (Cobbe 2021, 12,023 cit.), LEVER (ICML 2023, 415 cit.), Let's Verify Step by Step (5,375 cit.) |
| S4 | PhoBERT-base (MIT) | 0.135B | phân loại họ hàm toolkit top-3 → gợi ý + lọc truy hồi | nhãn = hàm của chương trình đã kiểm | thấp (S1 đã phủ phần lớn) |
| S5 | BARTpho-syllable (MIT) | 0.40B | khôi phục dấu / chuẩn hoá câu chat làm gợi ý | cặp tổng hợp không dấu → có dấu | chỉ khi bộ câu không dấu cho thấy mô hình tụt |
| S6 | Qwen3.5-0.8B làm **draft** cho giải mã suy đoán (cùng tokenizer) / EAGLE-3 (NeurIPS 2025, 433 cit., 2.6k★) | 0.87B | tăng tốc | EAGLE cần huấn luyện đầu | thấp (đầu ra sau fine-tune đã ngắn; MTP có sẵn chỉ −6 %) |

**Ngân sách tham số:** 9.65 (Qwen3.5-9B) + 0.60 (S1) + 0.87 (S3 bản 0.8B) = **11.12B**; thêm S4 + S5 = 11.66B; dùng S3 bản 2B
= 12.52B — đều ≤ 15B. Nếu mô hình chính là Gemma 4 12B-it (11.96B): chỉ còn ~3B → S1 + S3 (0.8B).

## 4. Chưng cất / huấn luyện — các biến thể

| Mã | Phương án | Bằng chứng | Chi phí | Ưu tiên |
|---|---|---|---|---|
| H1 | SFT trên chương trình đã kiểm (D1–D3) | OmniSQL (PVLDB 2025), DB-GPT-Hub | đã có trong kế hoạch | P0 |
| H2 | **Tăng cường nhiễu** cho câu huấn luyện: không dấu, gõ sai, viết tắt (`dthu`, `sl`), văn chat, từ địa phương (`src/perturb.py` sẵn có) | Spider-Syn (ACL 2021), Dr.Spider (ICLR 2023) | ~0 GPU | P0 |
| H3 | Lời giải ngắn của thầy (`<plan>`) | Distilling Step-by-Step (1,259 cit.) | thầy sinh thêm ~100 token / mẫu | P1 |
| H4 | **Quỹ đạo dùng tool** (lỗi → gọi T1 / T5 / T8 → sửa) | Toolformer, ToolLLM, CodeAct | tổng hợp bằng code + lọc thực thi | P1 |
| H5 | Tự huấn luyện có lọc (STaR / ReST-EM): mô hình sinh K chương trình, giữ cái chạy đúng, huấn luyện lại | STaR (1,241), ReST-EM (434) | 1 vòng ≈ 8 GPU-h | P1 (QĐ9) |
| H6 | **Chưng cất on-policy mức token** (GKD): học trò tự sinh, thầy Qwen3.6 chấm từng token (reverse KL). **Đã kiểm: Qwen3.6 và Qwen3.5 dùng chung vocab 248,044 token, cùng merges, cùng 26 token đặc biệt** → làm được | On-Policy Distillation / GKD (ICLR 2024, 1,281 cit.; `trl.GKDTrainer`), MiniLLM (ICLR 2024) | cao: cần log-prob của thầy (tính trước bằng vLLM trên mẫu của trò) | P2 (sau khi SFT ổn) |
| H7 | **Gộp mô hình** các nhánh LoRA thắng (trung bình / TIES / DARE) thay vì chỉ chọn một | Model soups (ICML 2022, 1,918 cit.), TIES (NeurIPS 2023, 1,091), DARE (ICML 2024, 911); mergekit 7.4k★; PEFT `add_weighted_adapter` có sẵn | ~0 GPU (CPU gộp + 1 lần đánh giá) | P1 |
| H8 | **Nhiều LoRA cùng phục vụ** (vLLM multi-LoRA), mỗi mẫu từ một adapter → đa dạng bộ sinh | XiYan-SQL (multi-generator) | +VRAM nhỏ | P1 (thử cùng H7) |
| H9 | RL thưởng thực thi (GRPO) | Arctic-Text2SQL-R1, SQL-R1, DeepSeekMath; TRL `GRPOTrainer`, verl 24.2k★ | cao, rủi ro | sau 16/11 |
| H10 | Tối ưu prompt / chọn ví dụ tự động | DSPy (1,166 cit., 39.0k★), TextGrad (290 cit., 3.8k★) | vừa (gọi LLM nhiều lần) | P2 |
| H11 | Huấn luyện bộ kiểm S3 từ mẫu đúng / sai (sản phẩm phụ của H5) | LEVER, Cobbe 2021 | 0.8B trên 1 GPU ≈ 30–60 phút | P1 |

## 5. Suy luận, chọn đáp án, tin cậy

| Mã | Phương án | Bằng chứng | Ghi chú |
|---|---|---|---|
| I1 | Self-consistency thích ứng + dừng sớm | Self-Consistency (8,228), Adaptive-Consistency (193), ESC (164) | đã có trong kế hoạch (WP8) |
| I2 | **Best-of-N có bộ kiểm** (S3 xếp hạng nhóm đáp án, kết hợp với phiếu) | Snell et al. "Scaling test-time compute" (2,392 cit.), LEVER | cho nhóm "bất đồng" |
| I3 | **Tự phê bình có tool** (CRITIC): mô hình dùng T6 `explain` kiểm kết quả trước khi chốt | CRITIC (ICLR 2024, 1,088), Self-Verification (EMNLP Findings 2023, 469), Chain-of-Verification (672) | cho nhóm "đồng ý mà sai" (63 câu) |
| I4 | **Postcheck bất biến** (T7): phần trăm, số đếm, top-N, khoảng ngày, kỳ không rỗng | CodeT (683) | code, rẻ — P0 |
| I5 | **Viết lại câu trước khi sinh** (Rephrase-and-Respond): mô hình nêu lại câu theo văn phong chuẩn rồi mới viết code | Rephrase and Respond (170 cit.); bằng chứng nội bộ: câu viết lại giúp đọc đúng 90 % (entry 26) | A/B prompt |
| I6 | Tách câu nhiều phần (Plan-and-Solve / Least-to-Most) | 937 / 2,025 cit. | chỉ cho câu phát hiện nhiều ý |
| I7 | **Ngưỡng abstain có kiểm soát rủi ro** (conformal risk control) bên cạnh chọn θ tối đa PA | Conformal Risk Control (ICLR 2024, 468), Conformal Language Modeling (ICLR 2024, 215) | kiểm độ bền ngưỡng |
| I8 | Prompt gọn (thông tin quan trọng không nằm giữa ngữ cảnh dài) | Lost in the Middle (TACL, 5,919 cit.) | ủng hộ prompt gọn ~2k token |
| I9 | Tăng tốc: prefix cache, FP8, MTP / draft, ít lượt sửa nhờ §2 | vLLM, EAGLE-3, Medusa (1,082) | đo trong WP10 |

## 6. Thay thế / bổ sung kiến trúc

| Phương án | Mô tả | Khi nào dùng |
|---|---|---|
| A (khuyến nghị) | DSL Python trên toolkit + ngữ pháp (§2.1) + tool dạng hàm (§1) | mặc định |
| B | JSON IR + trình biên dịch (§2.2) | nếu A/B trên `quick400` thắng ≥ 1 điểm |
| C | **DuckDB SQL** trên bảng chuẩn C + bảng thô làm nhánh thoát cho ngành / khái niệm chưa có hàm toolkit (DuckDB 42.9k★; tiền đề SQL của LLM rất mạnh) | thử trên bộ `pub_noind` (dưới) |
| D | Đa tác tử Planner / Coder / Verifier (DS-STAR, MAC-SQL) | chỉ cho phần câu tin cậy thấp (tốn thời gian) |
| E | Khung agent (smolagents 30.3k★, LangGraph 44.6k★, OpenAI Agents SDK 30.6k★) | không dùng (phụ thuộc, khó đóng gói offline); chỉ mượn mẫu CodeAct |

## 7. Bộ đánh giá bổ sung (đo khả năng gần private)

- `pub_noind`: các câu public dùng hàm ngành (dev_011, dev_012, dev_013, dev_014 — 112 câu public gốc) với **tài liệu hàm `industry` bị ẩn** khỏi prompt → mô phỏng ngành
  chưa có hàm (private có ngành mới); so nhánh thoát pandas vs DuckDB SQL (C).
- `rb_shift`: đổi một số định nghĩa trong rulebook (ví dụ doanh thu không gồm phí ship, tuần bắt đầu Chủ nhật) trên DN train,
  đáp án tính lại bằng toolkit có tham số → đo M3 / T4 (mô hình có theo rulebook mới không).
- `pub_noacc`: câu public bỏ dấu (`src/perturb.py`) → quyết định S5.
- Đánh giá truy hồi offline (CPU, không LLM): tỉ lệ top-4 ví dụ có cùng họ hàm với chương trình đúng, trên train (trừ
  `tr_ho`) và `pub_k0` → so n-gram ký tự hiện tại với S1.

## 8. Ma trận ưu tiên (trong 7 ngày tới)

| Mức | Phương án | Lý do |
|---|---|---|
| **P0 (làm ngay, rẻ, nhắm lỗi lớn nhất)** | §2.1 ngữ pháp DSL; T1, T3, T7, T8 (dạng hàm + tài liệu gọn); postcheck I4; tăng cường nhiễu H2 | ~35 % mẫu hỏng hiện nay là lỗi định dạng / API / thực thể |
| **P1 (đo được lợi rõ)** | S1 embedding truy hồi; S3 bộ kiểm 0.8B + I2; H7 gộp nhánh + H8 nhiều LoRA; H4 quỹ đạo tool; T2, T5, T6, T9; I3 tự phê bình có tool; I5 viết lại câu (A/B); C DuckDB (thử trên `pub_noind`) | nhắm nhóm "bất đồng" (118) và "đồng ý mà sai" (63), và tổng quát hoá ngành mới |
| **P2 (nếu còn thời gian)** | B JSON IR (A/B); H6 chưng cất on-policy; H10 DSPy; S4 PhoBERT; S5 khôi phục dấu; S6 draft / EAGLE; I6 tách câu | lợi chưa chắc hoặc chi phí cao |
| **Không làm trước 16/11** | H9 RL; D đa tác tử cho mọi câu; E khung agent | rủi ro thời gian / đóng gói |

## 9. Cách thử từng phương án (thống nhất)

Mỗi phương án là một cờ trong luồng / cấu hình huấn luyện; đo trên `quick400` (K = 1 rồi K = 4) + bộ bổ sung liên quan (§7);
giữ khi: PA `quick400` tăng ≥ 0.5 điểm hoặc thời gian giảm ≥ 20 % mà PA không giảm, và phần `tr_ho` / k=0 không giảm; ghi
bảng vào `EXPERIMENT_LOG.md`. Phương án dùng mô hình phụ phải ghi tổng tham số (≤ 15B) và nguồn trọng số đã kiểm (như §1
của `docs/08`).

## Nguồn

Citation: Semantic Scholar Graph API (08/11/2026); stars: GitHub search API (08/11/2026); vLLM structured outputs:
https://docs.vllm.ai/en/v0.32.0/features/structured_outputs.html ; mô hình: HF API (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B,
BGE-M3, Qwen3.5-0.8B, PhoBERT, BARTpho) và Kaggle API; tokenizer Qwen3.5 / Qwen3.6 so sánh trực tiếp `tokenizer.json`.
arXiv: SWE-agent 2405.15793; CodeAct 2402.01030; Toolformer 2302.04761; Gorilla 2305.15334; ToolLLM 2307.16789; PICARD
2109.05093; Grammar-Constrained Decoding 2305.13971; SynCode 2403.01632; XGrammar 2411.15100; Outlines 2307.09702; IRNet
1905.08205; NatSQL 2109.05153; Qwen3 Embedding 2506.05176; M3-Embedding 2402.03216; Training Verifiers 2110.14168; LEVER
2302.08468; Let's Verify Step by Step 2305.20050; Model soups 2203.05482; TIES 2306.01708; DARE 2311.03099; GKD 2306.13649;
MiniLLM 2306.08543; EAGLE-3 2503.01840; Medusa 2401.10774; Scaling test-time compute 2408.03314; CRITIC 2305.11738;
Self-Verification 2212.09561; Chain-of-Verification 2309.11495; Rephrase and Respond 2311.04205; Plan-and-Solve 2305.04091;
Least-to-Most 2205.10625; Conformal Risk Control 2208.02814; Conformal Language Modeling 2306.10193; Lost in the Middle
2307.03172; DSPy 2310.03714; TextGrad 2406.07496; Spider-Syn 2106.01065; Dr.Spider 2301.08881; OmniSQL 2503.02240.
Repos: SWE-agent/SWE-agent, xingyaoww/code-act, huggingface/smolagents, ShishirPatil/gorilla, OpenBMB/ToolBench,
mlc-ai/xgrammar, dottxt-ai/outlines, guidance-ai/guidance, cube-js/cube, dbt-labs/metricflow, Canner/WrenAI,
QwenLM/Qwen3-Embedding, FlagOpen/FlagEmbedding, huggingface/sentence-transformers, arcee-ai/mergekit, SafeAILab/EAGLE,
FasterDecoding/Medusa, duckdb/duckdb, langchain-ai/langgraph, openai/openai-agents-python, stanfordnlp/dspy, zou-group/textgrad,
verl-project/verl, huggingface/trl, VinAIResearch/PhoBERT, VinAIResearch/BARTpho.
