# PDF 1 — Scientific Findings (DataMate v3, đến 03/11/2026 14:30 UTC)

*Phạm vi: đóng góp, ý nghĩa, kết quả tích cực và tiêu cực, bằng chứng, điều kiện kiểm chứng. Kinh nghiệm kỹ thuật /
vận hành ở PDF 2; tiến độ, vị trí file và việc tiếp theo ở PDF 3. Số liệu gốc: `EXPERIMENT_LOG.md` (entry 0–13).*

## 1. Bài toán và thước đo

- Hỏi đáp tiếng Việt trên dữ liệu bảng **bẩn, mỗi doanh nghiệp (DN) một cấu trúc khác** (CSV, Excel nhiều sheet,
  SQLite), với **rulebook** định nghĩa nghiệp vụ (doanh thu, đơn hợp lệ, nợ, tồn, lô…).
- Điểm: **Penalized Accuracy** PA = (đúng − sai) / N; abstain = 0. Ràng buộc: offline, ≤ 15B tham số, ≤ 6 giờ / 2×RTX 5090.
- Train: 30 DN, 3000 câu có đáp án. Public test: 14 DN mới (dev_001–dev_014), 1000 câu, không nhãn; điểm chỉ có qua portal.
- Đánh giá nội bộ: **leave-one-business-out (LOO)** trên train. *dev* = DN có ánh xạ tham chiếu của DN khác chuyển được;
  *holdout* = DN phải để LLM tự viết ánh xạ (gần với public nhất).

## 2. Đóng góp chính (kiến trúc v3)

**Ý tưởng cốt lõi: tách "hiểu dữ liệu" khỏi "hiểu câu hỏi", đặt ngữ nghĩa nghiệp vụ vào code đã kiểm chứng.**

1. **Chuẩn hoá theo DN, không theo câu.** LLM viết *một* chương trình ánh xạ cho mỗi DN, biến bảng thô thành bảng chuẩn
   C (`orders`, `lines`, `items`, `payments`, `purchases`, `lots`…). Chi phí LLM theo số DN, không theo số câu.
2. **Toolkit nghiệp vụ `dm.Shop`** cài rulebook một lần (doanh thu, đơn hợp lệ, nợ theo tuổi, tồn theo lô, trung bình
   theo thứ…). Chương trình trả lời chỉ ~1 dòng (`revenue(month(2025, 7), item="…")`), đầu ra LLM ngắn → nhanh, ít sai.
3. **Bộ đọc câu theo luật (`qparse`)** cho các mẫu câu phổ biến, cộng mẫu LLM; bỏ phiếu có trọng số (luật qw = 4).
4. **Kiểm tra nội dung ánh xạ (soft checks)**: so bảng chuẩn với chính dữ liệu và rulebook (doanh thu đơn = tổng dòng,
   cột README gọi là "tiền hàng", tên hàng viết tắt, tên thợ gọi tắt, đơn ghi nợ không nợ gì, giờ tạo đơn chỉ có ngày…).
   Kết quả dùng để xếp hạng ứng viên và gửi lại cho LLM sửa.
5. **Chọn ánh xạ theo đồng thuận đáp án** và **dự phòng ánh xạ** (chạy chương trình trên ánh xạ hạng 2 khi hạng 1 bỏ trống).
6. **Bỏ phiếu nhiều lần chạy, biết chất lượng ánh xạ**: câu bị tách thì lấy đáp án của lần chạy có ánh xạ sạch rõ rệt.

## 3. Kết quả tích cực (có bằng chứng)

### 3.1 Tiến trình trên train (LOO, 3000 câu, một lần chạy)

| Hệ | dev | holdout | all | Ghi chú |
|---|---|---|---|---|
| v1 (LLM viết pandas mỗi câu, phương án cũ) | ≈0.47 | ≈0.44 | ≈0.45 | bỏ |
| v3 đầu (round-based, Qwen3.5-9B) | 0.897 | 0.644 | 0.810 | |
| + pipeline + think-map | 0.926 | 0.761 | 0.870 | |
| v4b: kiểm tra ánh xạ + vòng đầy đủ + câu đã làm sạch | 0.976 | 0.913 | 0.955 | |
| v11: budget forcing, grounding, sửa lỗi chung | 0.990 | 0.946 | 0.975 | |
| v19: bảng thô ánh xạ bỏ sót đưa vào prompt | 0.999 | 0.961 | 0.986 | |
| **v20: chuyển ánh xạ tham chiếu theo quy ước README** | **0.998** | **0.977** | **0.991** | 2975 đúng / 2 sai / 23 abstain |

### 3.2 Các phát hiện định lượng

- **Qwen3.5-9B > Qwen3-14B-AWQ** cùng giao thức (≈0.889 vs 0.751 trên 1499 câu): mô hình nhỏ hơn, mới hơn thắng.
- **Pipeline không rào chắn** nhanh 2.2× so với chạy theo vòng và PA cao hơn (0.848 vs 0.810): sửa lỗi ngay, không chờ.
- **Oracle luật** (ánh xạ tham chiếu + chỉ chương trình luật): 2666 / 36 / 298 → **2967 đúng / 1 sai / 32 chưa đọc**.
  Toolkit + luật gần như đúng tuyệt đối khi ánh xạ đúng → sai số còn lại nằm ở ánh xạ và ở các câu luật chưa đọc.
- **Grounding**: mẫu LLM nhắc mặt hàng không có trong câu sai 85 % (99/116) vs 7 % ở mẫu khác → loại khỏi bỏ phiếu.
- **Budget forcing** (s1, Muennighoff et al. 2025) chống vòng lặp suy luận: pha ánh xạ ngắn hơn 25 %, tỉ lệ thành công
  không đổi; 21 % lần thử đầu từng lặp "Wait, …" đến trần 18k token.
- **Bỏ phiếu nhiều lần chạy (train)**: 1 lần 0.928–0.969 → 3 lần 0.976–0.981 → 4 lần 0.980.
- **Làm sạch câu hỏi theo luật** đạt 99.79 % từ / 98.1 % câu (LOO 2232 câu); LLM làm sạch chỉ 70 % → bỏ.
- **Bỏ phiếu biết ánh xạ** (offline trên v20): gỡ 15 câu bị tách, 10 về đúng đáp án của v19 (9/9 câu dev_009).

### 3.3 Điểm public (portal, user nộp)

| Bản | PA | Đúng | Sai | Abstain | Thay đổi chính |
|---|---|---|---|---|---|
| v13 | 0.637 | 785 | 148 | 67 | |
| v15 | 0.634 | 792 | 158 | 50 | |
| v18 | 0.669 | 800 | 131 | 69 | mục ngành mới của rulebook, dòng tổng |
| **v19** | **0.687** | 801 | 114 | 85 | bảng thô bỏ sót trong prompt; unanswerable → abstain |
| v20 | 0.650 | ~767 | ~117 | 117 | (may rủi ánh xạ, xem §4.2) |

## 4. Kết quả tiêu cực và hạn chế (trung thực)

### 4.1 Train không dự báo được public

Train 0.98–0.99 nhưng public 0.63–0.69. Nguyên nhân đo được:
- public có **ngành mới** (mẹ và bé, sửa xe, điện thoại) với mục rulebook riêng mà train không có;
- **quy ước dữ liệu mới**: tên hàng viết tắt + bảng tra (dev_014), tên thợ gọi tắt (dev_012), nợ chỉ ghi trong ghi chú
  tự do (dev_008), doanh thu ở cột có tên lạ (dev_009 "Giá trị hàng"), dòng tổng (dev_009 / dev_010);
- **từ địa phương** trong câu hỏi khác với dữ liệu ("cá tràu" / "cá lóc", "đậu phụ rán" / "đậu hũ chiên", "cốc sạc" /
  "củ sạc"…) → lọc mặt hàng trượt → trả 0 ở mọi lần chạy.

### 4.2 Lỗi chung và may rủi ánh xạ chi phối điểm public

- Số câu đúng **đứng yên ~800** từ v15; điểm tăng chỉ nhờ bớt câu sai. Các phiên bản trả lời gần như giống hệt nhau
  (v18 / v19: chỉ 4 câu cùng trả lời mà khác) → **~114 câu sai là lỗi chung** của mọi lần chạy; bỏ phiếu không gỡ được.
- **Không tất định giữa các lần chạy**: cùng cấu hình chỉ 80–89 % câu giống nhau; nguyên nhân là mẫu ánh xạ LLM khác nhau
  (thành phần batch vLLM → số học khác). `VLLM_BATCH_INVARIANT=1` không chạy được với Qwen3.5.
- **v20 thua v19 (0.650 vs 0.687)** dù train tốt hơn: 3 lần chạy chọn 3 ánh xạ khác nhau cho dev_003 / dev_009 → 52 câu
  bị tách → abstain. Điểm public dao động **±0.03 chỉ vì may rủi ánh xạ**, ngang một bản sửa thật → so sánh hai bản bằng
  một lần nộp không đáng tin.

### 4.3 Những thứ đã thử và không giữ

| Thử | Kết quả | Quyết định |
|---|---|---|
| Bỏ phiếu chéo các ánh xạ (vote2 / vote3) | 0.964 / 0.942 < top1 0.967 | bỏ (giữ dự phòng fb2: 0.974) |
| Escalation (thêm mẫu khi nghi ngờ) | +0.012 toàn bộ, kiểu `table` tụt, +27 % thời gian | giữ mức 4, không tăng |
| `--map-repairs 4` không suy luận | ánh xạ thành công 64 % vs 52 % public, train 0.941 | chưa chứng minh, giữ mặc định |
| KV cache FP8, MTP speculative | FP8 KV không hỗ trợ Qwen3.5/vLLM 0.32; MTP chỉ −6 % | bỏ |
| Chọn theo đồng thuận W 100–300 | làm tệ một số lựa chọn | giữ W 30 |
| LLM làm sạch câu | 70 % từ đúng | bỏ |
| Chuyển ánh xạ theo README (v20) | holdout train +0.016, **không đổi** đáp án public nào | giữ (vô hại), không phải đòn bẩy public |

### 4.4 Ngưỡng kiểm chứng còn yếu

- Không có nhãn public → mọi nhận định "câu này sai" là suy luận từ: luật vs LLM bất đồng, ánh xạ ứng viên bất đồng,
  đáp án 0 bất thường, đối chiếu README. **Không gán nhãn tay đáp án public** (luật của user).
- Số đúng / sai public suy ra từ PA và số abstain (đúng = (PA·1000 + đã trả lời)/2), không quan sát trực tiếp.

## 5. Ý nghĩa khoa học

- Với ≤ 15B tham số, **chuyển ngữ nghĩa vào code kiểm chứng + LLM làm việc hẹp** hiệu quả hơn hẳn để LLM tự viết toàn bộ
  phân tích (0.45 → 0.99 train). Đòn bẩy lớn nhất là **ánh xạ dữ liệu đúng**, không phải đọc câu.
- **Kiểm tra nội dung ánh xạ dựa trên bất biến của rulebook** (doanh thu = tổng dòng; ghi nợ ⇒ còn nợ; giờ có thời điểm)
  là cách tổng quát phát hiện ánh xạ sai mà không cần nhãn; mỗi kiểm tra mới đều được xác nhận 0 báo động giả trên
  36 ánh xạ tham chiếu train.
- Với hệ có thành phần ngẫu nhiên ở pha "hiểu dữ liệu", **bỏ phiếu theo câu không đủ**: phải ổn định lựa chọn ánh xạ
  (gộp ứng viên giữa các lần chạy, bỏ phiếu biết chất lượng ánh xạ).

## 6. Điều kiện kiểm chứng (tái lập từng kết luận)

| Kết luận | Cách kiểm | Điều kiện |
|---|---|---|
| Oracle 2967 / 1 / 32 | `python3 tests/eval_lib.py data/raw/train/train <cache_train> --show 0 --clean question_cleaning/train_questions_clean.jsonl` | CPU, ~5 phút; cache từ `src/ingest.py` |
| Kiểm tra ánh xạ không báo nhầm | `python3 tests/test_plausibility.py data/raw/train/train <cache_train>` → `0 flagged` | CPU |
| PA train của một run | `python3 src/report.py data/raw/train/train <run>/results/train` | log của kernel |
| Bỏ phiếu / tie-break | `python3 src/vote.py out.jsonl <run1> <run2> <run3>` (+ `automation/map_penalties.py`) | log 3 lần chạy |
| Lỗi luật đọc câu trên public | `scratchpad/rule_vs_llm.py` (đã mô tả trong entry 13) | log public |
| Điểm public | chỉ user nộp portal | mỗi lần nộp là một mẫu, nhiễu ±0.03 |

Mô hình: Qwen3.5-9B FP8 (vLLM 0.32, 4×L4 Kaggle); các tham số: `--k 4 --kc 8 --repairs 2 --shots 4 --think-map
--max-tokens 1024 --qw 4 --escalate 4 --confirm 1 --think-budget 12288` (đầy đủ trong `CLAUDE.md` / PDF 3).
