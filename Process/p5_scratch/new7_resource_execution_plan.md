# KẾ HOẠCH TRIỂN KHAI & PHÂN BỔ KAGGLE — 7 PHƯƠNG PHÁP, 14 BIẾN THỂ
**Phần bổ sung cho ZIP:** `new7_variants_research_bundle.zip` · **Ngày:** 15/11/2026 · **Trạng thái:** Lập lịch, chưa đẩy các kernel N1–N7.
**Tham chiếu:** Repository `lhaochen118-ship-it/ZC`, nhánh `claude/datamate-v3-0dny90`; tài liệu gốc `new7_variants_assessment.md` và `new7_cpu_feasibility_results.json`.
> **Nguyên tắc không đổi:** `configs/flow_p5_zs_strict.json` (P5 + Qwen3.5-9B pretrained tự sinh luật) là baseline strict zero-shot, PA D1 mean **0.754** (`b00s0=0.753`, `b00s1=0.755`). `configs/flow_p5_final.json` đạt **0.952 D1** khi fine-tuned chỉ là reference. Không viết tay quy tắc nghiệp vụ, không dùng đáp án public/private test để tạo luật/học/ngưỡng, tổng parameters inference <15B.

## 1. Mục tiêu, dữ liệu đã có và giới hạn của kế hoạch
Tích hợp từng biến thể vào P5 mà **không thay đổi backbone và các thành phần không liên quan**, chạy so cặp với B00-S trên cùng seeds/questions. Mọi thay đổi về chương trình, đồ thị schema, unit certificate, parity hoặc instrumentation đều phải để **Qwen tự suy luận, tự sinh luật** cho DN/schema mới. Chỉ giữ checks toán học và an toàn tổng quát.
**Kết quả thực đã xác minh trong source:** B00-S trung bình 0.754 trên 1.000 câu D1. Mỗi seed chạy khoảng **110–113 phút**, 4×L4; `gpu_util` trung bình **48.4–49.8%**, peak VRAM **21.8–21.9 GB/GPU** theo `Process/scores_p5zs.csv`. Đây là **thời gian toàn pipeline**, chưa đủ phân rã chính xác setup/inference/CPU. 12/12 CPU smoke tests trong ZIP là trên dữ liệu tổng hợp, **không phải** score P5.
**Tài nguyên hiện có:** repo đã có `automation/fanout.py`, `automation/launch_many.py`, `automation/quota.py`, `automation/probe_l4.py`, `automation/run_kernel.py`, `Process/tools/p5/p5_report.py`. Tài liệu `READING_GUIDE.md` đề cập **19 tài khoản** đã từng được cấp; con số account **đang hợp lệ và quota của từng acc chưa thể xác minh** trong môi trường này. Kế hoạch dùng ký hiệu A01…A19, **không đưa usernames/token vào báo cáo**. Không có Kaggle CLI hoặc secrets ở phiên soạn thảo; chưa push jobs.
**Ràng buộc:** tối đa 2 GPU commit sessions/acc, mỗi session 4×L4 24GB (96GB **không gộp**); tối đa 5 CPU sessions/acc; giới hạn session 12 giờ; GPU L4 PCIe không NVLink; nếu không có L4 thử 2×T4 với cấu hình phù hợp. Đo quota thực qua `python3 automation/quota.py`, không suy từ giới hạn theo tuần. Chính sách Kaggle và mức cho phép đồng thời phải xác minh lại khi chạy.

## 2. Chuyển số đo B00 thành ngân sách có căn cứ
| Đại lượng | Công thức / quan sát | Ý nghĩa |
|---|---|---|
| D1 full P5 strict, 1 seed | 110–113 phút × 4 L4 = **7.33–7.53 GPU-giờ** | Số liệu lịch sử trên chính P5 |
| 200 câu screen cùng setup | Thời gian dự kiến **0.55–1.60 giờ/session** tùy thêm LLM call | Ngoại suy từ 1.000 câu, có phần overhead cố định; cần pilot |
| GPU-giờ vật lý | `n_gpu × session_wall_hours` | **Không** phải giờ quota tính tiền của Kaggle |
| Quota Kaggle | `session_wall_hours × 2.5` cho 4×L4 (hệ số bảo thủ từ `CLAUDE.md`) | Cần cộng 2 sessions của cùng acc |
| Quota có headroom | `(quota theo công thức) × 1.20`, rồi trừ các reservations hiện có | Tránh chết job khi run dài hơn dự kiến |
| Thời gian thực kết thúc wave | `max(duration của các sessions cùng wave) + queue/staging overhead` | Không bằng tổng GPU-giờ / 4 nếu không song song đủ |
| Độ dài D1 | 1.000 câu theo `Process/scores_p5zs.csv` | Với dataset khác, thay N trong ước lượng; không suy ra thời gian tuyến tính tuyệt đối |
**Công thức điều chỉnh sau pilot:** `T(n)=T_load + n × t_question + T_extra_rule + T_verify`; đo riêng trên 30–50 câu theo 3–5 DN và reuse cache ingest/model; nhân hệ số 1.15–1.30 cho variance. Trước chạy full cần `T_projected < min(11h, remaining_quota/2.5)` với ít nhất 20% dự phòng. Không cố đạt 96GB VRAM utilization; 21.9GB/GPU đã gần trần 24GB, tăng batch/KV vô ý có thể OOM.

## 3. Tổng ngân sách 14 biến thể — số GPU, số acc và số giờ
Các cột sau là **ước lượng planning envelope**, **chưa benchmark N1–N7 trên Kaggle**. “Engineering CPU” là công **triển khai/instrument và test bằng CPU** (person-hours), **không phải** thời gian 5 CPU Kaggle sessions dùng để chạy script. “GPU screen” là lượt chạy 200 câu/1 seed sau smoke; các phương án có thể replay được đánh dấu 0 GPU screening.

| Biến thể | Phụ thuộc | Kỹ thuật/CPU-h | GPU/phiên | Screen wall-h | Screen GPU-h | Full D1 nếu finalist (wall-h/seed) |
|---|---|---:|---:|---:|---:|---:|
| **N1-A** | B00; CPU preflight | 2 | 4 | 0.55–0.95 | 2.2–3.8 | 1.9–2.7 |
| **N1-B** | N1-A pass | 3 | 4 | 0.65–1.05 | 2.6–4.2 | 2–3 |
| **N2-A** | CPU preflight | 4 | 0 | 0–0 | 0–0 | 1.9–2.5 |
| **N2-B** | N2-A pass | 8 | 0 | 0–0 | 0–0 | 2.2–3.5 |
| **N3-A** | B00; candidate traces | 4 | 4 | 0.55–1 | 2.2–4 | 2.3–3.7 |
| **N3-B** | N3-A pass | 7 | 4 | 0.9–1.6 | 3.6–6.4 | 3–5 |
| **N4-A** | B00; schema profiles | 3 | 4 | 0.6–1 | 2.4–4 | 1.8–2.5 |
| **N4-B** | N4-A pass | 5 | 4 | 0.75–1.3 | 3–5.2 | 2.1–3.2 |
| **N5-A** | B00; train-only labels | 5 | 4 | 1.5–3 | 6–12 | 2.8–4.5 |
| **N5-B** | N5-A pass | 4 | 4 | 1.2–2.5 | 4.8–10 | 2.5–4.2 |
| **N6-A** | B00; traces | 3 | 0 | 0–0 | 0–0 | 1.9–2.7 |
| **N6-B** | N6-A pass | 5 | 4 | 0.75–1.3 | 3–5.2 | 2.2–3.3 |
| **N7-A** | B00; rules audit | 3 | 4 | 0.65–1.15 | 2.6–4.6 | 1.9–2.8 |
| **N7-B** | N7-A pass | 4 | 4 | 0.8–1.4 | 3.2–5.6 | 2.1–3.2 |

**Tổng 14 biến thể nếu đều qua gate vào screening:** **8.90–16.25 session-hours** L4 = **35.6–65.0 GPU-hours vật lý** = **22.25–40.63 quota-hours** theo hệ số 2.5, phân bổ qua nhiều tài khoản, cộng **~60 engineering person-hours sơ bộ** (chưa tính sửa lỗi ngoài dự tính). Đây **không** phải ngân sách phải chi: có 3 biến thể ưu tiên CPU-only ở bước screen, phần lớn P2/P3 sẽ bị gate loại trước khi chạy.
**Full evaluation có điều kiện:** 1–3 finalists, mỗi finalist **2 seeds × 1 session 4×L4/seed**, thông thường **1.9–3.5 giờ/seed** (N3/N5 có thể 3.7–5.0h). Nếu có 2 finalists thông thường: **30.4–56 GPU-hours**, thêm ~**19–35 quota-hours**; nếu 3: **45.6–84 GPU-hours**, thêm ~**28.5–52.5 quota-hours**. Phải thay những số này bằng thời gian pilot theo từng finalist. Nếu finalist là phương án replay CPU-only và không thay generation/repair, không cần chạy lại model cho mọi scoring test; cần xác minh trên complete trace.

## 4. Thiết kế thực hiện của từng biến thể và điều kiện dừng
### 4.1. N1-A — Permutation invariance (P1)
**Vị trí tích hợp:** `src/hybrid.py / src/perturb.py (mới: biến đổi *dữ liệu*)`. **Dependencies:** B00;CPU preflight.
**Quy trình:** Chạy trên bản dữ liệu sao, chỉ đối với programs có chứng nhận bất biến thứ tự. Có thể xử lý replay không Qwen trước, rồi nếu cần mới trả bằng chứng cho Qwen tự sửa.
**Cổng quyết định:** Tỉ lệ applicable, false alarms, wrong rejected/correct lost; dừng nếu <5% applicable hoặc không tăng PA.
**Nguồn lực:** ~2 engineering CPU-hours; screen 4 GPU L4, 0.55–0.95h/session = 2.2–3.8 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 1.9–2.7h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.2. N1-B — Partition-recomposition (P2)
**Vị trí tích hợp:** `src/hybrid.py + aggregate analyzer`. **Dependencies:** N1-A pass.
**Quy trình:** Chia tập rows sau filter; tính SUM/COUNT/MIN/MAX hoặc (SUM,COUNT) cho AVG, so với chương trình gốc. Phải xử lý ORDER BY, groupby và NULL đúng.
**Cổng quyết định:** Chỉ mở nếu N1-A có evidence và subset aggregate được chứng minh; dừng nếu false alarms/overhead.
**Nguồn lực:** ~3 engineering CPU-hours; screen 4 GPU L4, 0.65–1.05h/session = 2.6–4.2 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2–3h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.3. N2-A — Canonical-row lineage (P2)
**Vị trí tích hợp:** `src/dm.py + sandbox instrumentation, tuyệt đối không sửa kết quả`. **Dependencies:** CPU preflight.
**Quy trình:** Gắn IDs cho các rows canonical, truy vết input của thao tác lọc/tổng hợp; replay trên log cũ trước.
**Cổng quyết định:** Dừng nếu lineage coverage <80% ở subset được hỗ trợ hoặc đổi output P5.
**Nguồn lực:** ~4 engineering CPU-hours; screen 0 GPU L4, 0–0h/session = 0–0 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 1.9–2.5h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.4. N2-B — Raw-to-canonical lineage (P3)
**Vị trí tích hợp:** `src/ingest.py → src/automap.py → canonical C`. **Dependencies:** N2-A pass.
**Quy trình:** Carry _source_file/_sheet/_row_id qua mapping rồi phân tích JOIN fanout; instrumentation tách khỏi schema mà Qwen nhìn thấy để không gây leakage.
**Cổng quyết định:** Dừng khi IDs không bảo toàn hoặc sửa canonicalization làm thay đổi đáp án; phải pass unit tests ingestion.
**Nguồn lực:** ~8 engineering CPU-hours; screen 0 GPU L4, 0–0h/session = 0–0 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.2–3.5h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.5. N3-A — Counterexample discrimination (P1)
**Vị trí tích hợp:** `src/hybrid.py / src/judge.py + counterexample solver`. **Dependencies:** B00;candidate traces.
**Quy trình:** Chỉ bật với candidate không tương đương về biểu thức nhưng đồng kết quả trên dữ liệu thật; tìm fixture nhỏ phân biệt, Qwen từ đầu vào quyết định ứng viên.
**Cổng quyết định:** Chứng minh phân biệt không đồng nghĩa chọn đúng; giới hạn solver 1–3 giây và 1 call bổ sung/câu.
**Nguồn lực:** ~4 engineering CPU-hours; screen 4 GPU L4, 0.55–1h/session = 2.2–4 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.3–3.7h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.6. N3-B — AST hole synthesis (P3)
**Vị trí tích hợp:** `AST transformer trong sandbox, sửa 1 biểu thức/lượt`. **Dependencies:** N3-A pass.
**Quy trình:** Đặc tả do Qwen tự sinh; search constrained hole choices (predicate/join key/aggregation) và kiểm bằng counterexample, không nhìn test labels.
**Cổng quyết định:** Dừng nếu spec Qwen không đủ mạnh, sửa mà tăng regression hoặc vượt runtime 6h.
**Nguồn lực:** ~7 engineering CPU-hours; screen 4 GPU L4, 0.9–1.6h/session = 3.6–6.4 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 3–5h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.7. N4-A — Structural join candidate graph (P0)
**Vị trí tích hợp:** `src/ingest.py + src/retrieval.py (schema graph supplement)`. **Dependencies:** B00;schema profiles.
**Quy trình:** Xây graph edges từ PK/FK, uniqueness, key overlap, null ratio, value profile; đưa 3–5 cạnh có lý do vào prompt; Qwen vẫn chọn JOIN.
**Cổng quyết định:** Đo câu cần JOIN, graph recall/precision, sai kỳ/entity; dừng nếu canonical C đã đủ và prompt noise.
**Nguồn lực:** ~3 engineering CPU-hours; screen 4 GPU L4, 0.6–1h/session = 2.4–4 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 1.8–2.5h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.8. N4-B — Cardinality-aware join planner (P2)
**Vị trí tích hợp:** `src/retrieval.py + src/sqlesc.py`. **Dependencies:** N4-A pass.
**Quy trình:** Tìm 2–3 join paths, ước lượng cardinality/fanout; Qwen chọn theo ý định; không hard-code bảng ngành.
**Cổng quyết định:** Chỉ mở sau N4-A thắng và có lỗi fanout thực; đo join explosion & cost.
**Nguồn lực:** ~5 engineering CPU-hours; screen 4 GPU L4, 0.75–1.3h/session = 3–5.2 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.1–3.2h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.9. N5-A — Global activation steering (P3)
**Vị trí tích hợp:** `Inference engine activation hooks + train-only contrastive examples`. **Dependencies:** B00;train-only labels.
**Quy trình:** Tính steering directions từ train-only correct/incorrect pairs; sửa hidden states trong generation, đo memory/correctness.
**Cổng quyết định:** Không phải zero-shot strict; chỉ mở nếu engine support và không phá tool/rule generation.
**Nguồn lực:** ~5 engineering CPU-hours; screen 4 GPU L4, 1.5–3h/session = 6–12 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.8–4.5h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.10. N5-B — Conditional activation steering (P3)
**Vị trí tích hợp:** `Activation hook có gate uncertainty`. **Dependencies:** N5-A pass.
**Quy trình:** Dùng vector N5-A đã kiểm; chỉ kích hoạt với vote entropy, disagreement hoặc lỗi execution; không học gate trên public.
**Cổng quyết định:** Chỉ chạy sau N5-A gain thực; phải giảm regression và compute.
**Nguồn lực:** ~4 engineering CPU-hours; screen 4 GPU L4, 1.2–2.5h/session = 4.8–10 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.5–4.2h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.11. N6-A — Pandas vs DuckDB parity (P1)
**Vị trí tích hợp:** `src/sqlesc.py + pandas deterministic replay`. **Dependencies:** B00;traces.
**Quy trình:** Chạy equivalent SELECT trên Pandas/DuckDB cho subset hỗ trợ và normalize dtype/NULL/date/rounding; offline replay không cần LLM.
**Cổng quyết định:** Dừng nếu semantics mismatch giả hoặc coverage thấp; không coi engine A đúng mặc định.
**Nguồn lực:** ~3 engineering CPU-hours; screen 0 GPU L4, 0–0h/session = 0–0 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 1.9–2.7h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.12. N6-B — Independent dual rendering (P2)
**Vị trí tích hợp:** `src/hybrid.py: cùng Qwen tự render pandas + SQL`. **Dependencies:** N6-A pass.
**Quy trình:** Một semantic plan Qwen tự viết, sinh hai programs độc lập; đối chiếu kết quả, Qwen quyết định khi lệch.
**Cổng quyết định:** Chỉ khi N6-A thấy lỗi thực; cost thêm LLM generation và duplicate execution.
**Nguồn lực:** ~5 engineering CPU-hours; screen 4 GPU L4, 0.75–1.3h/session = 3–5.2 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.2–3.3h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.13. N7-A — Qwen-generated unit certificate (P0)
**Vị trí tích hợp:** `src/units.py / P5 W1 output + neutral dimensional checker`. **Dependencies:** B00;rules audit.
**Quy trình:** Qwen tự sinh dimension certificate gắn output, toolkit xem xét nhân/chia cộng đơn vị; không hard-code bảng quy đổi ngành.
**Cổng quyết định:** Dừng nếu chứng nhận cùng lỗi tương quan với program hoặc không cải thiện PA.
**Nguồn lực:** ~3 engineering CPU-hours; screen 4 GPU L4, 0.65–1.15h/session = 2.6–4.6 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 1.9–2.8h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

### 4.14. N7-B — Dimensional AST dataflow (P2)
**Vị trí tích hợp:** `AST of generated code + dimensional typeflow`. **Dependencies:** N7-A pass.
**Quy trình:** Theo dõi thứ nguyên đại lượng trong expression tree theo metadata *do Qwen sinh*; không kích hoạt precheck unit heuristic viết tay.
**Cổng quyết định:** Chỉ mở nếu N7-A có coverage; metric dimension inconsistency precision/recall.
**Nguồn lực:** ~4 engineering CPU-hours; screen 4 GPU L4, 0.8–1.4h/session = 3.2–5.6 GPUh. Hai seed full nếu đi tiếp: 2 sessions cùng 4 L4/phiên, 2.1–3.2h mỗi session (đối với replay CPU-only có thể loại phần GPU).
**Bằng chứng đầu ra:** `config.diff.json`, full logs, `programs.jsonl`, `unit/rule_provenance.jsonl` (nếu có), `score.json`, `gpu_summary.json`, `errors.tsv`, `pairwise_deltas.csv` để so với B00-S. Không thay file baseline.

## 5. Lịch song song tối ưu: A01–A03 là account *roles*, không phải usernames
**Chiến lược:** dồn đủ **2 GPU sessions/acc** rồi mới dùng acc khác, nếu quota còn đủ. Mỗi session mặc định 4×L4 chạy **một biến thể với DP4/work stealing** (đã được repo hỗ trợ), vì chạy 1 GPU/variant khiến một variant 200 câu mất gần 4× thời gian. Chỉ chia 4 GPUs ra 4 variant workers nếu benchmarking chứng minh wall-clock tổng tốt hơn, output tách được và không có KV/OOM. Trong 200 câu đầu, ưu tiên DP4 ổn định hơn là ép full memory.

### Wave CPU-0: Audit + preflight, 0 GPU
- **A01 CPU slots 1–3:** inventory logs, input data & rules; N1-A AST precondition/replay; N4-A graph builder prototype. **A02 CPU slots 1–3:** N2-A canonical lineage feasibility, N6-A DuckDB parity tests, N7-A unit certificates schema; tối đa 5 CPU sessions/acc.
- **Kiểm thử trước GPU:** `python3 new7_cpu_feasibility.py` (đã PASS 12/12 synthetic); sau đó dùng **train/dev** P5 real data để xác nhận coverage; không dùng test labels để viết luật. Engineering window dự kiến **4–12h thực hiện phối hợp** cho nhóm đơn giản nếu có người/coding agent chạy song song, không phải cam kết 4h hoàn thành vì tổng engineering person-hour cao.
- **Gate:** Không dùng luật nghiệp vụ viết tay; Qwen tự sinh rules như `flow_p5_zs_strict.json` với `hide_api: industry`, `lrl_regen: 1`; nếu lớp triển khai thay đổi `automap`, phải so equality output trên baseline.

### Wave G1: Screening ưu tiên (chạy đồng thời 4 sessions = 2 accounts = 16 L4)
| Account | GPU slot | Biến thể | Số L4 | Wall h ước tính | Quota h cần +20% |
|---|---|---|---:|---:|---:|
| **A01** | S1 | N4-A | 4 | 0.6–1 | 1.8–3 |
| **A01** | S2 | N7-A | 4 | 0.65–1.15 | 1.95–3.45 |
| **A02** | S1 | N1-A | 4 | 0.55–0.95 | 1.65–2.85 |
| **A02** | S2 | N3-A | 4 | 0.55–1 | 1.65–3 |
**Tổng G1:** 4 sessions × 4 L4 = **16 GPUs hoạt động song song**; **9.4–16.4 GPU-hours vật lý**; **5.88–10.25 quota-hours cộng gộp** (chưa 20% reserve); wall-clock của tính toán **~0.65–1.15h** nếu queue đồng thời (chưa tính thời gian coding/audit/upload). N2-A/N6-A CPU replay song song, 0 GPU.
**Quota khuyến nghị trước push:** A01 có ít nhất **6.5h quota** cho N4-A+N7-A, A02 có ít nhất **6h quota** cho N1-A+N3-A (bao gồm buffer xấp xỉ 20%); nếu không đủ hãy chọn A03/A04 còn quota, hoặc tách 2 sessions sang acc khác; không tự động vượt quota. Nếu quota real >=? là điều phải đo chứ không giả định.

### Wave G2: Các biến thể tăng độ sâu — chỉ chạy nhánh mẹ đã qua gate
| Account | Session | Biến thể | Dependency | GPUs | Wall ước tính |
|---|---|---|---|---:|---:|
| A01 | S1 | N4-B | N4-A pass | 4 | 0.75–1.3 h |
| A01 | S2 | N7-B | N7-A pass | 4 | 0.8–1.4 h |
| A02 | S1 | N1-B | N1-A pass | 4 | 0.65–1.05 h |
| A02 | S2 | N6-B | N6-A pass | 4 | 0.75–1.3 h |
Nếu cả bốn thực sự qua gate, **16 L4**, **11.8–20.2 GPU-hours**, **7.38–12.63 quota-hours**, wall **0.8–1.4h** khi chạy song song; N2-B có thể kiểm nguồn raw→canonical ở CPU và không chiếm GPU. Không mở N6-B nếu N6-A không tìm được bất thường có ý nghĩa.

### Wave G3: Nhánh nghiên cứu sâu (gated, không chạy ồ ạt)
- **N3-B:** A01/S1, 4 L4, 0.9–1.6h nếu N3-A vượt gating. **N5-A:** A01/S2, 4 L4, 1.5–3.0h gồm activation calibration/200q, chỉ khi backend có hook được profiling. **N5-B:** A02/S1, 4 L4, 1.2–2.5h **sau khi N5-A hoàn thành và chứng minh gain** (không thể start N5-A/B cùng thời điểm).
- Nhánh này có tối đa **14.4–28.4 GPU-hours** nếu mở cả ba; wall critical path của activation steering **2.7–5.5h**, cộng setup/retry; N3-B có thể chạy song song với N5-A. N2-B vẫn là CPU investigation độc lập.

### Wave G4: Full D1, 2 seeds, 1–3 finalists
| Số finalists | Accounts tối thiểu | GPU sessions đồng thời | L4 đồng thời | Full GPU-hours ước lượng | Wall khi GPU sẵn |
|---|---:|---:|---:|---:|---:|
| 1 | **1** | 2 | 8 | **15.2–28.0** | 1.9–3.5h + queue |
| 2 | **2** | 4 | 16 | **30.4–56.0** | 1.9–3.5h + queue |
| 3 | **3** | 6 | 24 | **45.6–84.0** | 1.9–3.5h + queue |
**Yêu cầu quota mỗi account cho finalist thông thường:** `2 sessions × (1.9–3.5h) × 2.5 × 1.2` = **11.4–21.0 quota-hours/acc** nếu hai seed chạy cùng một account và không còn quota đặt chỗ. Nếu thiếu, chia các seeds qua nhiều acc có quota hoặc sử dụng chỉ 1 seed làm screen, giữ seed 2 khi bổ sung quota. Với N3/N5 phức tạp có thể cần cao hơn. Không suy ra 2×RTX5090 submission runtime từ 4×L4: phải profiling riêng.

## 6. Lịch tổng hợp theo đường găng và ngân sách ba kịch bản
| Kịch bản | Workload GPU | L4 đồng thời đỉnh | Accounts tối thiểu theo 2 sessions/acc | GPU-hours vật lý | Wall GPU nếu luôn có slots |
|---|---|---:|---:|---:|---:|
| **FAST** | G1 (4 variants) + full 2 finalists ×2 seeds | 16 | **2** | **39.8–72.4** | ~2.6–4.7h GPU + queue/staging |
| **SELECTIVE** | G1 + G2 (nếu 4 pass) + full 2 finalists | 16 | **2** | **51.6–92.6** | ~3.4–6.1h GPU + queue/staging |
| **EXHAUSTIVE* (không ưu tiên)** | Cả 11 GPU-screen variants + 3 finalists×2 seeds | 24 | **3** | **81.2–149.0** | ~6–13h GPU theo dependencies, chưa tính coding |
(*) EXHAUSTIVE chỉ dùng để đánh giá trần tài nguyên; không nên chạy N5 hoặc các biến thể hậu nghiệm nếu gate không qua. Các khoảng chưa tính 5–12 GPU-hours dự phòng smoke/debug, và ~15–25% quota/headroom. Chúng cũng chưa bao gồm tổng thời gian **engineering 60 person-hours** nếu cả 14 biến thể đều cần code. Điều kiện tài nguyên là **GPU không phải bottleneck duy nhất**.
**Quota Kaggle tương ứng:** FAST: *(39.8–72.4)/4×2.5* ≈ **24.9–45.3 quota-hours**, SELECTIVE ≈ **32.3–57.9**, EXHAUSTIVE ≈ **50.8–93.1**, phân tán trên accounts. Những con số này không phải quota 1 account. Thêm 20% headroom và kiểm trạng thái quota thật trước push.
**Tổng lịch thực tế:** CPU engineering/integration + preflight có thể là đường găng lớn nhất. Với 2 coding workers độc lập và GPU ready, FAST có thể chạy sau ~6–14h để triển khai 4 biến thể đầu (mang tính giả định, không có benchmark coding); GPU nằm sau đó ~2.6–4.7h. Với 1 worker, cần chấp nhận thời gian dài hơn. Không hứa hẹn hoàn thành trước hạn đóng băng nếu chưa thực tế xong code và preflight.

## 7. Thuật toán scheduling để tránh lãng phí account/quota
1. Lấy danh sách account ID từ cấu hình môi trường **không chứa token trong báo cáo**, kiểm từng account qua `python3 automation/quota.py` và kernel probe, xác nhận GPU L4 thật. Các account theo tài liệu từng có 19 không đồng nghĩa toàn bộ sẵn sàng hôm nay.
2. Tính `remaining_hours − reserved_hours`. Với mỗi kernel, `need = 2.5 × est_wall_h × 1.20` quota-hours. Chỉ tạo phiên mới nếu có quota đủ; với 2 phiên cùng account phải tính **tổng nhu cầu**, không check mỗi session một cách độc lập.
3. Sắp xếp theo dependency + priority + time-to-value; ghép 2 phiên lên A01 trước, hết slot chuyển A02, tiếp tục tới A19 khi cần. **Không tạo thêm acc hoạt động chỉ vì còn acc** nếu không tăng tốc hoặc có nguy cơ hết quota. Trường hợp thiếu quota, đổi account/bớt variants, không khởi chạy job chắc chắn chết.
4. Reuse datasets/code snapshots immutable; khi cùng acc chạy 2 kernels, **upload code dataset một version cố định rồi launch tuần tự** (chờ `launched` trước khi gọi plan tiếp) để tránh các kernels đè `zc-code` nhau. Chạy kernel song song trên nhiều accounts sau khi code snapshot có sẵn.
5. Nếu mục tiêu chạy một variant nặng trên 4GPU: dùng **DP4** trước nếu model 9B FP8 vừa từng GPU (P5 peak ~21.9GB); TP/FSDP chỉ nếu model/architecture không fit. N5 hidden states có thể cần **TP2 × DP2** nếu cache activations lớn. Batch size tăng từng bước theo profiler; không coi 96GB là unified VRAM. Không dùng DDP trong inference nếu không cần train.
6. Nếu không có L4: kiểm 2 T4, dtype `fp16` và **checkpoint int4 phù hợp**; phải chạy lại control tương ứng vì precision thay đổi có thể tác động PA. Không so trực tiếp các variants T4-int4 với baseline L4-FP8 như một thử nghiệm công bằng.
7. Đo `GPU utilization`, peak VRAM, tokens/s, queue time, CPU tool workers, per-shop wall, abnormal LLM calls. Nếu GPU nhàn do CPU verifier, đẩy CPU shadow checks sang CPU sessions trong cùng acc; không tăng batch tới OOM. Group by business/prefix, claim-file work stealing, deterministic seeds.

## 8. Runbook Kaggle: lệnh và cách kiểm sự cố
Dựa trên code hiện có của repo, dùng CLI thực tế sau khi có working copy và tokens đã thiết lập an toàn:
```bash
# Trước khi đẩy: không in token gốc
python3 automation/quota.py
python3 automation/preflight.py --help         # kiểm lại interface thực tế
kaggle competitions list                      # xác minh kết nối, không khẳng định đã có GPU L4

# Với plan đã viết đúng schema [{acc,name,jobs,est_h,args}, ...]
python3 automation/launch_many.py "$RUNS_DIR" "$PLAN_JSON"

# Kết quả historical / new P5 runs: khi có đúng dữ liệu, log đầy đủ
python3 Process/tools/p5/p5_report.py "$RUNS_DIR"
```
**Schema plan Kaggle thật:** mỗi entry có `acc`, `name`, `jobs`, `est_h` (giờ tường/session), `args`; tham chiếu `automation/plans_p5/wave1a.json` để điền. File `new7_resource_schedule.json` trong ZIP **chỉ là artifact scheduling chuẩn hóa, không phải file đưa thẳng cho `launch_many.py`**. Phải có notebook job `.jobs`, configs, code, dataset mount và quota snapshot thật trước launch.
**Nhật ký bắt buộc:** `Process/runs/new7/<experiment>/{config,model_revision,seed,question_hash,shop_rules_provenance,logs.jsonl,submission.jsonl,result.json,gpu_summary.json,quota_snapshot.json}`. Không ghi token, không commit file credentials; `chmod 600` với Kaggle token files trong `~/.kaggle`. Public/private ground truth chỉ dùng theo rule challenge; không phát tán dữ liệu nếu không có quyền.
**Treo / không ra output:** trong L4/CPU khoảng 5 phút (T4 khoảng 10 phút), **kiểm QUEUED vs STARTING vs RUNNING**. Log vLLM có thể ra muộn; chỉ reset khi có bằng chứng stall, không xóa vô cớ. Hủy phiên thật bằng `kaggle kernels delete -y <owner>/<slug>` theo chính sách repo; restart tối đa **4 lần** với thay đổi hợp lý hoặc chuyển acc, xem log/quota nếu vẫn lỗi. Nếu tự động hủy không được, báo người dùng thao tác thủ công. Đảm bảo cleanup không hủy phiên khác đang hữu ích.
**Session 12h:** không bắt đầu task nếu dự kiến vượt phần giờ còn lại; dùng `ZC_DEADLINE`, job checkpoint/resume từ `automation/chain_next.py`/`automation/resume.py` nếu có. Không kéo dài một kernel với chất lượng output chưa được lưu. Nếu GPU hết, chỉ đẩy CPU job rất nhẹ (<~30 phút) sang CPU sessions, GPU-heavy chờ slot; không lén chạy GPU-heavy trên CPU.

## 9. Quy chuẩn đánh giá và những gì thực sự chứng minh “phù hợp”
1. **Zero-shot strict là B00-S 0.754 trên D1**. Mọi biến thể N1/N2/N3/N4/N6/N7 dùng Qwen pretrained P5 self-generated rules được đối chiếu **cùng case IDs, decoder settings, seeds**. N5-A/B có steering direction học từ nhãn train nên báo **calibrated**, không gọi strict zero-shot.
2. Đánh giá `PA`, `correct/wrong/abstain`, `PA_pub_ind`, `PA_pub_noind`, per-case paired wins/losses, `rescued vs regressed`, `false_alarm`, `rule_generation_validity`, `tool execution`, `wall`, VRAM. Đo confidence interval bằng bootstrap paired. Không tune một mức threshold/edge score/steer alpha trên public/private test labels.
3. CPU replay chỉ đánh giá những điều kiện deterministic thật sự có đủ trace/data; nếu kết quả cần gọi Qwen sửa luật/chương trình thì phải cấp GPU và đo lại end-to-end. Không dùng kết quả smoke 12/12 synthetic để tuyên bố tăng điểm.
4. **Promotion:** bắt buộc **không đưa luật viết tay**, evidence về ΔPA trên tập mới, kiểm 2 seeds, không mất quá nhiều câu đúng, không vượt 6h inference machine theo quy định BTC, under 15B params, không phá final answer provenance. N1/N2/N6 có thể tăng abstain dù phát hiện lỗi đúng; PA là thước đo cuối cùng.
5. Điều kiện dừng khi fail: N1 applicable thấp; N2 lineage không thể bảo toàn; N3 chỉ phân biệt nhưng Qwen không chọn được; N4 canonical C đã đủ; N5 steer làm giảm chất lượng hay không hỗ trợ engine; N6 false mismatches về dtype/NULL; N7 self-certificate tương quan lỗi. Nếu không vượt, **ghi lý do và giữ P5 strict baseline**, không spam phiên bản mới.

## 10. Kiểm tra tính hợp lệ và deliverables sau chạy
**Cổng H0 trước launch:** đúng commit & hashes, active Kaggle accounts, acceptance L4 grant, quota≥need, no-token leaks, user-provided datasets được phép mount offline, baseline config `flow_p5_zs_strict`, rule origin audit.
**Cổng H1 sau screen:** đầy đủ logs + score + cost; parent variants qua gate; không dựa trên kết quả public để viết luật/ngưỡng; score phải paired với B00-S.
**Cổng H2 sau finalists:** 2 seeds, optional unseen schema/business, reproducible environment, parameter audit, full offline runtime trên target thực hoặc proxy có giải thích.
**Bảng kết luận cuối cần có:** `experiment_id`, `hypothesis`, `screen/full`, `ΔPA_B00S`, `ΔPA_tuned_reference_0.952` (chỉ tham khảo, cùng D1), `wins/losses`, `model_params`, `GPUh`, `quota_h`, `accounts_used`, `wall`, `rule_compliance`, `decision` (GO/STOP/DEFER).
**Status thực:** chưa chạy các GPU kernels N1–N7 trong phiên soạn báo cáo. Đã có số liệu baseline B00-S, các experiments P01/P02/P03 v.v. trên GitHub, và 12/12 synthetic CPU smoke tests. `new7_resource_budget.csv` + `new7_resource_schedule.json` là bảng kế hoạch có cấu trúc, có thể được agent biến thành Kaggle job plans sau khi xác minh quota.

## 11. Nguồn đối chiếu trong repo
- [B00 strict score và thời gian thực](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/Process/scores_p5zs.csv)
- [P5 zero-shot gốc, không LoRA](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/configs/flow_p5_zs_strict.json)
- [P5 tuned 0.952 reference](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/configs/flow_p5_final.json)
- [Playbook DP / GPU profile](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/docs/05_GPU_OPTIMIZATION.md)
- [Hạ tầng Kaggle 2 sessions/acc, quota/checkpoint](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/docs/02_KAGGLE_OPS.md)
- [Orchestration hiện có](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/automation/launch_many.py)
- [Công cụ check quota](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/automation/quota.py)
- [Kết quả paired score](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/Process/tools/p5/p5_report.py)
- [Bản đồ repo, historical account context](https://github.com/lhaochen118-ship-it/ZC/blob/claude/datamate-v3-0dny90/READING_GUIDE.md)

**Phụ lục file trong ZIP:** `new7_variants_assessment.pdf/.md` (14 biến thể), `new7_cpu_feasibility.py` + `new7_cpu_feasibility_results.json` (12 synthetic tests), `new7_resource_execution_plan.pdf/.md` (kế hoạch mới), `new7_resource_budget.csv`, `new7_resource_schedule.json`, `README_BUNDLE.md`.
