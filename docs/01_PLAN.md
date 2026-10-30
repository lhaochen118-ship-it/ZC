# 01 — Kế hoạch giải DataMate từ A đến Z

*Bản 2026-10-30, viết trước khi xem dữ liệu; giữ làm tài liệu tham khảo (tổng quan paper, kiến trúc). Trạng thái thực tế và việc tiếp theo: `handoff/03_HANDOFF.md`. Khác biệt đã biết: rulebook đổi theo split nên [B] "rulebook compiler" được thay bằng bước compile bảng chuẩn do LLM viết lúc chạy; planner DSL (E4) chưa làm.*

## 1. Mục tiêu và ràng buộc quyết định thiết kế

| Ràng buộc | Hệ quả thiết kế |
|---|---|
| PA: +1 / 0 / −1 | Trả lời khi P(đúng) > 0.5; cần **ước lượng xác suất đúng có hiệu chuẩn** cho mỗi đáp án |
| Khớp chính xác, không điểm từng phần | Đáp án cuối do **code thực thi** tính ra, không để LLM tự viết số; định dạng (làm tròn, `yyyy-mm`, thứ tự list) xử lý tất định |
| ≤ 15B tham số tổng, offline | Một LLM ~8–14B (int4/fp8) làm lõi; không thêm mô hình phụ trừ khi tổng vẫn ≤ 15B |
| 6 giờ trên 2 × RTX 5090; hoà điểm xét thời gian | Đường nhanh tất định cho câu "dễ", chỉ gọi nhiều mẫu LLM cho câu khó; batch bằng vLLM |
| Tất định (≤ 0.03) | Temperature 0 cho đường chính, seed cố định cho mẫu phụ, ghi log đầy đủ |
| Rulebook + lịch âm–dương dùng chung | **Biên dịch rulebook thành thư viện hàm** (định nghĩa chỉ số, mốc thời gian) thay vì bắt LLM đọc lại mỗi câu |
| Cấm logic theo id | Mọi luật đều tổng quát theo schema / ngữ nghĩa, không theo `business_id` |

## 2. Tham khảo (chỉ đọc paper, không đọc code repo)

Đã quét alphaXiv ngày 30/10 (ưu tiên 2026). Các hướng liên quan trực tiếp:

**Agent hỏi–đáp trên bảng / spreadsheet**

- FlexSQL (2605.02815): agent khám phá schema và dữ liệu linh hoạt, xen kẽ thực thi, thay pipeline cố định.
- ProSPy (2606.05836): **profile dữ liệu trước**, rồi agent SQL + Python. Khớp với dữ liệu "bẩn" của DataMate.
- SheetCompass (2608.14452), TabClaw (2606.10316), multi-agent multi-format spreadsheet (2604.12282): đọc Excel thật
  (nhiều sheet, header lệch, ô gộp) bằng đồ thị quan hệ / nhiều định dạng biểu diễn.
- TabScope (2609.03395): chọn phạm vi bảng theo câu hỏi khi bảng lớn.
- FiCo (2610.03958), VIGIL (2610.04287): find-then-compute, vòng cải tiến có cổng kiểm định.
- MATA (2602.09642): multi-agent TableQA với kiểm tra độ tin cậy.

**Câu không trả lời được và abstain**

- PremiseSQL (2610.22158), LatentRefusal (2601.10398), Never the Number (2608.13926), PRACTIQ (2410.11076),
  Agentic Abstention (2606.28733), memory cho câu không trả lời được (2608.27924).
- Selective prediction cho text-to-SQL (2607.06799): tín hiệu nào dự báo đúng/sai (độ đồng thuận thực thi, lỗi, …).
- Conformal abstention cho text-to-SQL (2609.25938); trajectory-derived confidence (2610.04156).

**Huấn luyện mô hình nhỏ bằng phản hồi thực thi**

- ReToolSQL (2608.27796), SERL-SQL (2608.00485), BIRD-RL (2610.29426), SQL-Zero (2609.04697), FINER-SQL (2605.03465),
  LIMIT (2609.24186), DualSQL (2609.18135): SFT/GRPO với reward thực thi giúp mô hình 7–14B tiệm cận mô hình lớn.
- Chọn ứng viên: SIRIUS-SQL (2606.01246, đa ứng viên neo vào thực thi), ORM verifier (2606.30851), listwise selection
  bằng memory (2609.00834).

Kinh điển cần nhớ: ReAct, self-consistency (Wang et al. 2022), DIN-SQL / CHESS / CHASE-SQL (schema linking + chọn
ứng viên), Spider 2.0.

**Mô hình ứng viên (phải xác minh ngày public < 2026-10-01 và giấy phép trước khi dùng):** họ Qwen3/Qwen3.x 8B–14B
(instruct và coder), Gemma 4 12B, Phi-4 14B, các bản fine-tune text-to-SQL 7–14B (OmniSQL, Arctic-Text2SQL-R1). Chọn
bằng thí nghiệm E1 (§5), không chọn theo cảm tính.

## 3. Kiến trúc đề xuất: "Compile – Plan – Execute – Vote – Decide"

```
data pkg ──► [A] Ingest & profile (tất định, cache) ──► DuckDB + schema card + entity index
rulebook ──► [B] Rulebook compiler (một lần/split) ──► metrics.py, time.py (âm lịch), defaults
question ──► [C] Planner LLM: NL → plan JSON (DSL)  ──► [D] Executor tất định ──► đáp án + bằng chứng
                         └─ không biểu diễn được ──► [C'] Code agent (SQL/pandas, ReAct, có sửa lỗi)
            [E] K đường giải đa dạng ──► nhóm đáp án theo luật khớp của BTC ──► phân bố p(đáp án)
            [F] Bộ quyết định hiệu chuẩn: answer / unanswerable / abstain (ngưỡng 0.5 trên p đã hiệu chuẩn)
            [G] Formatter tất định (type, làm tròn, yyyy-mm, thứ tự, tie theo mã) ──► append submission.jsonl
```

**[A] Ingest & profile.** Nạp mọi CSV / sheet Excel / bảng SQLite vào DuckDB. Tự dò dòng header, bỏ dòng tổng / ghi chú,
xử lý ô gộp, số kiểu Việt (`1.234.567`, `1,5tr`), ngày nhiều định dạng. Tạo **schema card** ngắn: bảng, cột, kiểu, khoá,
vài giá trị mẫu, giá trị rời rạc của cột ít giá trị. Tạo **entity index**: mọi cách viết tên sản phẩm/khách/chi nhánh
→ mã, gom bằng chuẩn hoá + fuzzy. Kết quả cache theo doanh nghiệp (đăng thành dataset public, xem `02_KAGGLE_OPS.md`).

**[B] Rulebook compiler (đóng góp chính).** Rulebook dùng chung cho cả split, nên chuyển nó **một lần** thành thư viện
hàm có test: định nghĩa doanh thu, đơn hợp lệ, giá vốn, lợi nhuận gộp, công nợ quá hạn, tồn kho, tiêu chí xếp hạng
mặc định, cách tính "tuần" (thứ 2 → chủ nhật?), "cùng kỳ", ngày lễ âm lịch qua bảng âm–dương. LLM không cần "nhớ" luật;
nó chỉ chọn hàm. Đây là luật tổng quát theo ngữ nghĩa, không theo id, nên hợp lệ (Rule 4). **(GĐ: rulebook đủ cụ thể
để mã hoá.)**

**[C] Planner → DSL.** LLM dịch câu hỏi thành plan JSON nhỏ: `{metric, entity_type, filters, time_range(expr),
group_by, agg, order, k, answer_type}`. Lợi ích: (1) đầu ra ngắn → nhanh; (2) executor tất định → tái lập được;
(3) kiểm tra được (thực thể có trong entity index? khoảng thời gian có dữ liệu?). Câu không biểu diễn được bằng DSL
thì chuyển sang **[C'] code agent** sinh SQL/Python trong sandbox, có vòng sửa lỗi.

**[E] Self-consistency có cấu trúc.** K đường giải đa dạng: DSL (greedy), DSL (mẫu nhiệt độ thấp, seed cố định), SQL
agent, pandas agent. Nhóm đáp án bằng **chính luật khớp của BTC** (`src/score.py`), nên hai đáp án "tương đương khi
chấm" được gộp. Chỉ chạy đủ K khi đường greedy có tín hiệu bất định (đường nhanh cho câu dễ → tiết kiệm thời gian).

**[F] Quyết định hiệu chuẩn.** Đặc trưng: tỷ lệ đồng thuận, số lỗi thực thi, plan có qua kiểm tra không, entity có
khớp chính xác không, loại câu, type đáp án, kết quả rỗng/NULL. Hồi quy logistic / isotonic học trên train (hoặc dữ
liệu tự tổng hợp) → p. Trả lời nếu p > 0.5 (cộng biên an toàn chọn bằng cross-validation để tối đa PA kỳ vọng).

**Phát hiện `unanswerable`.** Tín hiệu tất định: thực thể/cột/khái niệm không có trong schema card; khoảng thời gian nằm
ngoài phạm vi dữ liệu; câu hỏi cần thông tin không được ghi (ví dụ lý do khách không mua). Planner có nhãn
`unanswerable(reason)`; bộ quyết định chỉ chấp nhận khi có tín hiệu tất định đi kèm hoặc đồng thuận cao.

**[G] Formatter.** Ép kiểu, làm tròn 2 chữ số, `percent` giữ dấu, ngày `yyyy-mm-dd` / `yyyy-mm`, list theo thứ tự
plan, hoà xếp theo mã tăng dần, chỉ đơn hợp lệ. Trả tên theo cách viết có trong dữ liệu (hoặc mã, cũng được chấp nhận).

## 4. Fine-tune (khi baseline đã ổn)

- **Tự tổng hợp dữ liệu** (Rule 6 cho phép): sinh câu hỏi từ template × doanh nghiệp train, có biến thể khẩu ngữ, viết
  tắt, không dấu; đáp án do executor tất định tính → nhãn sạch, số lượng lớn.
- **SFT** planner (NL → DSL) bằng LoRA trên mô hình lõi; sau đó **GRPO với reward = s_i của PA** (+1/0/−1 trực tiếp,
  có hành động abstain), theo tinh thần ReToolSQL/BIRD-RL. Reward trùng metric của BTC là điểm mới đáng thử.
- Train trên Kaggle L4×4 (acc đã có grant) hoặc T4×2 với QLoRA.

## 5. Vòng lặp thí nghiệm

Mỗi vòng: **đề xuất → xây → chạy trên Kaggle → chấm bằng `src/score.py` → phân tích lỗi theo nhóm → giữ / sửa / bỏ.**
Ghi mọi vòng vào `EXPERIMENT_LOG.md` (giả thuyết, cấu hình, PA theo type/loại câu, thời gian, quyết định).

| # | Thí nghiệm | Câu hỏi cần trả lời | Đo |
|---|---|---|---|
| E0 | Đọc `data.txt` + cấu trúc zip; sửa `score.py` theo định dạng đáp án thật | Split, định dạng, phân bố type/loại câu | — |
| E1 | Baseline: code agent ReAct + DuckDB, 3–4 mô hình ứng viên, greedy, không abstain | Mô hình nào tốt nhất trong 15B; trần không abstain | PA, accuracy, thời gian/câu |
| E2 | + Ingest/profile + schema card + entity index | Bao nhiêu lỗi do đọc dữ liệu bẩn | PA theo loại lỗi |
| E3 | + Rulebook compiler + time resolver âm lịch | Lỗi nghiệp vụ/thời gian còn lại | PA nhóm thời gian/metric |
| E4 | Planner DSL + executor; code agent làm fallback | DSL phủ bao nhiêu % câu; nhanh hơn bao nhiêu | coverage, PA, s/câu |
| E5 | Self-consistency K đường + bộ quyết định hiệu chuẩn | Đường cong PA theo ngưỡng; độ hiệu chuẩn | PA, ECE, % abstain |
| E6 | Unanswerable detector | Precision/recall của nhãn `unanswerable` | PA nhóm unanswerable |
| E7 | SFT → GRPO reward PA | Tăng bao nhiêu so với prompt-only | PA trên dữ liệu chưa thấy |
| E8 | Tối ưu tốc độ: vLLM batch, prefix cache, đường nhanh, int4/fp8 | Tổng thời gian trên cả test | giờ / toàn bộ test |
| E9 | Docker RTX 5090 (CUDA ≥ 12.8), chạy 2 lần kiểm tất định | |ΔPA| giữa 2 lần ≤ 0.03? | ΔPA, thời gian |

**Kiểm định:** tách train theo **doanh nghiệp** (không theo câu) thành dev/holdout, vì private test có doanh nghiệp mới.
Mọi lựa chọn sau khi nhìn dữ liệu phải kiểm lại trên holdout. Public LB (5 lần/ngày, user nộp tay trên portal) chỉ để
đối chiếu, không dùng để chọn siêu tham số chi tiết.

## 6. Lịch dự kiến (còn 17 ngày tới hạn Docker 16/11)

| Ngày | Việc |
|---|---|
| 30–31/10 | E0, chuẩn bị Kaggle (dataset dữ liệu, wheels vLLM, trọng số mô hình), E1 |
| 1–4/11 | E2–E4 |
| 5–8/11 | E5–E6, nộp public lần đầu bản có abstain |
| 8–12/11 | E7 (nếu còn lợi), E8 |
| 12–15/11 | E9: Docker, kiểm tất định và thời gian; README solution khai báo mô hình |
| 16/11 | Nộp Docker (để dư 1 lần nộp sửa) |
| 20–23/11 | Chạy private test, nộp |

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Không có máy RTX 5090 để thử | Kiểm trên Kaggle (T4/L4); Docker dùng base image BTC; tránh kernel tuỳ biến; thử vLLM với CUDA 12.8 sớm |
| Lệch tất định giữa 2 lần chạy | Greedy, seed cố định, batch size cố định, tắt nguồn ngẫu nhiên; voting giảm phương sai |
| Private test khác phân phối | Holdout theo doanh nghiệp; dữ liệu tổng hợp đa dạng; abstain bảo thủ |
| Quá 6 giờ | Đo s/câu sớm; đường nhanh; ghi đáp án tăng dần |
| Mô hình vi phạm luật (ngày, giấy phép, 15B) | Bảng kiểm trong README solution |
