# PDF 1 — Scientific Findings (DataMate, đến 08/11/2026)

*Phạm vi: đóng góp, ý nghĩa, kết quả tích cực và tiêu cực, bằng chứng, điều kiện kiểm chứng. Kinh nghiệm kỹ thuật /
vận hành ở PDF 2; tiến độ, vị trí file và việc tiếp theo ở PDF 3. Số liệu gốc: `EXPERIMENT_LOG.md` entry 0–27.*

## 1. Bài toán và thước đo

- Hỏi đáp tiếng Việt trên dữ liệu bảng bẩn (CSV, Excel nhiều sheet, SQLite), **mỗi DN một cấu trúc**, kèm rulebook
  (doanh thu, đơn hợp lệ, nợ, tồn, lô, mục ngành riêng). PA = (đúng − sai) / N.
- Train: 30 DN, 3000 câu có đáp án. Public: 14 DN mới, 1000 câu; đáp án public có ở `data/raw/public_gt.jsonl` (user
  cung cấp, chỉ dùng để chấm). Private: DN và cách hỏi chưa thấy.
- **Ba thước đo, ý nghĩa khác nhau:**
- (1) Train / public gốc: đo trong mẫu (luật được viết từ chính các câu này).
- (2) Biến đổi bề mặt: câu không dấu, nhiễu chat, README xoá trắng, đổi tên cột + xáo cột (`tests/eval_split.py --perturb-*`).
- (3) **Paraphrase bằng LLM** (`src/paraphrase_llm.py`, 3 biến thể / câu, kiểm giữ số / ngày / mã / tên): đo khả năng đọc
     cách hỏi mới — gần private nhất. k=0 public = dev (được phân tích lỗi); k=1,2 public và mọi paraphrase train = held-out.

## 2. Hai kiến trúc đã xây và so sánh

| | Đường LLM (v3, đến v28) | Đường luật (rule-first, v30–v34) |
|---|---|---|
| Ánh xạ schema | LLM viết chương trình ánh xạ / DN → bảng chuẩn C; code kiểm; đồng thuận giữa các lần chạy | `automap`: README + từ điển tên cột + hồ sơ giá trị + đẳng thức tiền |
| Hiểu câu + sinh code | LLM viết K chương trình 1 dòng trên toolkit; luật `qparse` là một phiếu (qw 4) | `qnorm` làm sạch + `qparse` mẫu regex → chương trình; LLM chỉ cho câu luật bỏ lại |
| Tính toán | toolkit `dm.Shop` / `industry` (Python) | như bên trái |
| Abstain | bỏ phiếu, ngưỡng | luật không đọc được → LLM; LLM: ngưỡng phiếu 0.75 |
| Public gốc | **0.921** (v28) | **1.000** (v30) |
| Train | 0.9897 | 1.000 |

**Đóng góp dùng chung cho cả hai** (giữ nguyên trong kiến trúc tiếp theo):

1. **Chuẩn hoá theo DN, không theo câu**: chi phí ánh xạ theo số DN; chương trình trả lời chỉ ~1 dòng.
2. **Toolkit nghiệp vụ đã kiểm chứng** (`src/dm.py`, `src/industry.py`): mọi định nghĩa rulebook là hàm có test; LLM không
   tự tính. Oracle (ánh xạ tham chiếu + chương trình đúng): 2967 / 1 / 32 trên train.
3. **Kiểm tra nội dung ánh xạ theo bất biến rulebook** (`canon.secondary_checks`): doanh thu = tổng dòng, ghi nợ ⇒ còn nợ,
   biển số khớp sổ, giá vốn ≠ 0…; 0 báo động giả trên 36 ánh xạ tham chiếu train.
4. **Đồng thuận ánh xạ giữa các lần chạy** (`src/consensus.py`): gỡ "may rủi ánh xạ" (cùng cấu hình chỉ 80–89 % câu giống).

## 3. Kết quả tích cực

### 3.1 Đường LLM (train LOO, rồi public)

| Bản | Train | Public | Thay đổi chính |
|---|---|---|---|
| v1 (LLM viết pandas mỗi câu) | ≈0.45 | — | bỏ |
| v3 đầu | 0.810 | — | ánh xạ theo DN + toolkit |
| v4b → v11 → v20 | 0.955 → 0.975 → 0.991 | 0.63–0.69 (portal) | kiểm tra ánh xạ, budget forcing, grounding |
| v25 (5 lần chạy) | 0.9893 | 0.903 (chấm cục bộ) | toolkit ngành, kiểm tra mới |
| **v28** (8 lần chạy + đồng thuận) | 0.9897 | **0.921** (921 / 69 / 10) | slot chủ đề trong đồng thuận |

Phát hiện định lượng: Qwen3.5-9B > Qwen3-14B-AWQ (0.889 vs 0.751); pipeline không rào chắn nhanh 2.2×; grounding (mẫu nhắc
mặt hàng không có trong câu sai 85 %); bỏ phiếu 1 → 3 lần chạy: 0.93–0.97 → 0.98.

### 3.2 Đường luật

- Train 3000 / 3000, public 1000 / 1000 (chấm nghiêm, cả pandas 2.3.3 + calamine của image).
- Biến đổi bề mặt (luật một mình): đổi tên cột + xáo cột 1.000 / 1.000; README trắng train 1.000, public 0.993; không dấu
  0.989; nhiễu chat 0.966–0.978; toàn luồng (luật + LLM) trên nhiễu 0.989–0.996.

### 3.3 Paraphrase (held-out, toàn luồng có LLM, chấm theo đáp án câu gốc)

| Luồng | Public dev k=0 | Public held-out k=1,2 | Train held-out |
|---|---|---|---|
| Luật một mình | ≈0.51 (cả 3 biến thể) | | ≈0.67 |
| v31 luật + LLM | 0.640 | 0.559 | 0.752 |
| + ngưỡng tin cậy 0.75 | 0.660 | 0.579 | 0.768 |
| v32 (sửa chung 05/11) | 0.681 | 0.584 | 0.780 |
| **v34 + stage 1.5** (LLM viết lại câu, luật đọc lại) | 0.676 | **0.602** | ~0.801 (ước lượng) |

Bỏ các paraphrase đổi kỳ thời gian (gần như luôn đổi nghĩa): dev 0.814, held-out 0.712, train 0.829 (v32).
Stage 1.5: luật đọc câu viết lại đúng 90 % (train) / 84 % (public held-out) khi trả lời — cao hơn LLM trả lời trực tiếp
(~77 %).

## 4. Kết quả tiêu cực và hạn chế (trung thực)

1. **Điểm 1.000 của đường luật là trong mẫu.** Các mẫu `qparse` / `automap` được viết từ lỗi trên train và public. Trên
   cách hỏi mới, luật một mình chỉ ~0.67 (train) và ~0.51 (public). Đây là lý do user quyết định chuyển sang LLM-first
   (PDF 3 §2.1): luật if / else không tổng quát cho private.
2. **Train không dự báo public** ở đường LLM (0.99 vs 0.63–0.69 trên portal ở v13–v20): ngành mới, quy ước dữ liệu mới,
   từ địa phương. Sau khi có toolkit ngành + kiểm tra mới: 0.92.
3. **Điểm portal không so được giữa các đợt**: v23 = 0.87 rồi file lai X2 (v23 + 6 câu) = 0.756 → đáp án / tập chấm portal
   đã đổi. Chỉ so các file chấm cùng đợt.
4. **Paraphrase làm thước đo có nhiễu**: một phần "sai" là paraphrase đổi nghĩa (đổi kỳ, đổi đại lượng). 12 / 13 câu sai của
   luật trên câu viết lại (dev k=0) thuộc loại này. Thước đo dưới-ước lượng độ chính xác thật.
5. **Không tất định**: ánh xạ LLM khác nhau giữa các lần chạy; `VLLM_BATCH_INVARIANT=1` không chạy với Qwen3.5.
6. **Chưa đo**: đường LLM thuần (không luật) trên paraphrase held-out; thời gian trên 2×RTX 5090.

### Đã thử và không giữ

| Thử | Kết quả |
|---|---|
| LLM làm sạch câu | 70 % từ đúng (luật 99.8 %) |
| Bỏ phiếu chéo ánh xạ (vote2 / vote3) | 0.964 / 0.942 < top1 0.967 |
| FP8 KV cache, MTP speculative | không chạy / chỉ −6 % thời gian |
| Ngưỡng "câu lạ" (novelty) để chuyển câu cho LLM | câu public gốc cũng "lạ" → không tách được |
| "tháng trước" đứng một mình = tháng liền trước | 0 đúng / 3 sai (paraphrase đổi kỳ) |
| Bỏ dòng Voucher / phí ship khỏi doanh thu (v24) | portal tụt; giữ trong doanh thu đúng với đáp án BTC |

## 5. Ý nghĩa khoa học

- Với ≤ 15B tham số, **đặt ngữ nghĩa vào code kiểm chứng và giao LLM việc hẹp** (viết ánh xạ / chương trình 1 dòng) tốt hơn
  hẳn để LLM tự phân tích (0.45 → 0.99 train). Đòn bẩy lớn nhất là ánh xạ dữ liệu đúng.
- **Độ chính xác trong mẫu không phải bằng chứng tổng quát hoá**: hệ luật đạt 100 % trên mọi câu đã thấy nhưng ~0.5–0.67
  trên cách hỏi mới. Cần thước đo held-out (paraphrase, biến đổi dữ liệu) ngay từ đầu.
- **Chuẩn hoá cách hỏi bằng LLM rồi cho bộ đọc chính xác đọc lại** (stage 1.5) cho độ chính xác cao hơn LLM tự sinh
  chương trình trên cùng câu — gợi ý cho LLM-first: LLM viết lại câu + truy hồi ví dụ đã kiểm có thể tăng độ chính xác.
- Bất biến rulebook là cách tổng quát phát hiện ánh xạ sai không cần nhãn.

## 6. Điều kiện kiểm chứng

| Kết luận | Lệnh | Điều kiện |
|---|---|---|
| Train / public / biến đổi bề mặt | `S=<scratch> WORKERS=3 bash tests/eval_all.sh` (`quick` = chỉ gốc) | CPU, cache ingest; ~1–2 h đủ bộ |
| Oracle 2967 / 1 / 32 | `python3 tests/eval_lib.py data/raw/train/train <cache_train> --show 0 --clean question_cleaning/train_questions_clean.jsonl` | CPU |
| Kiểm tra ánh xạ không báo nhầm | `python3 tests/test_plausibility.py data/raw/train/train <cache_train>` → `0 flagged` | CPU |
| Luật trên paraphrase | `python3 tests/eval_split.py <split> <cache> <ref> --map auto --perturb-q para:<file>` | file paraphrase từ `<acc>/zc-para` |
| Toàn luồng trên paraphrase | `automation/jobs/v34.jobs` (Kaggle 4×L4, ~3.1 h) rồi chấm `flow/submission.jsonl` theo k | quota GPU |
| Đường LLM public 0.921 | `submissions/v28/submission.jsonl` chấm với `public_gt.jsonl` (`src/score.py`) | — |
