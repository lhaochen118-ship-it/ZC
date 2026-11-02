# 01 — Scientific Findings (DataMate, Zalo AI Challenge 2026)

*Cập nhật 2026-10-31 04:45 UTC (hết khung chat 2). Branch `claude/peaceful-carson-0dny90`, repo `lhaochen118-ship-it/ZC`.
Tài liệu này chỉ nêu đóng góp, kết quả và bằng chứng; cách vận hành ở PDF 2, việc tiếp theo ở PDF 3.*

## 1. Bài toán và hệ quả của thước đo

- **Bài toán:** agent trả lời câu hỏi tiếng Việt đời thường trên dữ liệu bảng bẩn (CSV/Excel/SQLite) của hộ kinh doanh nhỏ,
  hoặc trả `unanswerable` / `abstain`.
- **Thước đo:** Penalized Accuracy (đúng +1, abstain 0, sai −1).
- **Hệ quả quyết định:** chỉ trả lời khi xác suất đúng ước lượng > 0.5. Ước lượng dùng **tỷ lệ phiếu** của K chương trình
  độc lập: conf = phiếu của đáp án đứng đầu / K, trả lời khi conf ≥ τ.
- **Dữ liệu:**
  - train 3000 câu / 30 DN (có đáp án); public 1000 câu / 14 DN.
  - Câu hỏi theo khuôn mẫu: doanh thu theo kỳ, % thay đổi, tỉ lệ huỷ, top-k bán chạy, khách, công nợ, kho, giá nhập…
  - Văn phong chat: viết tắt (`hqua`, `dthu`, `bnhiu`), không dấu, gõ telex sai (`tieenf`, `dduwowcj`).
- **Rulebook khác nhau giữa các split** (public thêm luật mới; private có ngành mới) → không hard-code luật; agent đọc
  rulebook lúc chạy.

## 2. Giao thức đánh giá (cố định)

- **Chia theo DN:**
  - chọn mọi cấu hình / ngưỡng trên **dev** = biz_101–120;
  - báo cáo trên **holdout** = biz_121–130.
- **So sánh cặp:** mọi so sánh giữa hai cấu hình làm trên **cùng tập DN và cùng câu hỏi**.
- **Công cụ:**
  - bộ chấm `src/score.py` theo đúng luật khớp của BTC (kiểm: chấm đáp án chuẩn với chính nó = 1.0);
  - `src/compare.py`: PA dev/holdout, τ chọn trên dev, vote lại offline;
  - `src/analyze.py`: hiệu chuẩn, loại lỗi.
- **Few-shot leave-business-out:** ví dụ truy hồi luôn lấy từ DN khác (không bao giờ cùng DN với câu đang chấm). Lúc thi mọi
  DN đều mới, nên đây là ước lượng trung thực.

## 3. Đóng góp đã kiểm chứng

### 3.1 Bộ làm sạch tổng quát (`src/ingest.py`, từ khung chat 1)

- **Header và dòng rác:** dò header theo mật độ ô chữ; bỏ dòng tiêu đề, header lặp, dòng tổng; ghép header 2 tầng.
- **Số kiểu Việt / Mỹ:** `10.000đ`, `105k`, `1tr440`, `1,134tr`, `531 500`, `(1.000)`, `12,5%`, `1 bao 25kg`. Dấu thập
  phân suy theo cột.
- **Ngày nhiều dạng:** ISO, `dd/mm/yyyy`, `1 thg 6 2025`, `Ngày 1 tháng 6 năm 2025`, serial Excel; ngày không ghi năm lấy
  năm từ tên sheet / file.
- **Mã:** giữ dạng chữ (số 0 đầu, IMEI…).
- **Gộp:** gộp các file / sheet của cùng một danh sách.

**Bằng chứng:**
- `tests/test_ingest_robust.py`: 12/12 biến thể bẩn tổng hợp (không lấy từ dữ liệu thi).
- Audit cột đọc được một phần: 78 → 0 trên 44 DN.
- Chương trình viết tay trên bảng sạch khớp tuyệt đối đáp án BTC.

### 3.2 Lớp ngữ nghĩa có kiểm thử: `src/semantic.py: build_sales` (mới)

**Ý tưởng:** tách *luật ghi chép chung* của rulebook ra code Python đã kiểm thử; LLM chỉ còn **ánh xạ cột** (việc mô hình 14B
làm tốt), thay vì tự viết lại các luật đó trong từng chương trình (việc nó làm sai một cách hệ thống).

**Luật được cài sẵn:**
- ô trống ở dòng sau của đơn kế thừa dòng đầu;
- dòng trùng hệt trong cùng đơn tính một lần; dòng trùng ở hai đơn, hoặc ở bảng không có mã đơn, tính cả hai;
- mỗi đơn đúng một trạng thái (có dòng trả → trả hàng; có dòng huỷ → huỷ);
- `valid` = không huỷ và không trả;
- doanh thu = tổng thành tiền dòng;
- khách lẻ / vãng lai / trống → không phải khách;
- quy đổi đơn vị theo bảng quy đổi (`conv=`).

**Chịu lỗi của LLM (đều tổng quát, không theo DN):**
- tự tìm bảng header của đơn khi bị quên hoặc truyền nhầm (chọn bảng có cột cần và khoá trùng mã đơn > 50 %);
- tự đổi chỗ khi truyền nhầm bảng dòng / bảng đơn;
- giữ cả cột gốc và cột chuẩn;
- báo nhãn trạng thái chưa ánh xạ.

**Bằng chứng:** cho đáp án doanh thu tháng khớp BTC trên 4 kiểu tổ chức: phẳng có mã đơn (biz_101), phẳng không mã đơn
(biz_113), header + chi tiết có giờ (biz_106, biz_124), SQLite nhiều bảng (biz_123).

### 3.3 Chuẩn hoá câu hỏi và entity linking (`src/vnnorm.py`, `agent.EntityIndex`, mới)

- **`vnnorm`:** mở rộng viết tắt chat theo nguyên từ (`hqua` → hôm qua, `th7/2025` → tháng 7/2025, `dthu`, `bnhiu`, `đc`…).
  Giải mã gõ telex sai chỉ khi kết quả là từ có trong từ vựng (rulebook + câu hỏi của split). Bản chuẩn hoá hiển thị
  **cạnh** câu gốc, câu gốc vẫn là chuẩn.
- **Entity linking:** liệt kê các giá trị chữ của DN (tên hàng / khách / nhóm…) có dạng bỏ dấu nằm trong câu hỏi bỏ dấu,
  kèm cách viết đúng và cột chứa nó (`noi lau ga la e` → 'Lẩu gà lá é' trong cột 'Hàng bán' của bảng ban_hang).
- **Khoảng ngày dữ liệu:** in khoảng ngày có dữ liệu của DN, để câu về kỳ nằm ngoài dữ liệu trả `unanswerable` (luật §7).
  **Chưa đo.**

### 3.4 Few-shot truy hồi chương trình đúng (`--shots`, `src/build_shots.py`, mới) — đòn bẩy lớn nhất

- **Xây thư viện:** từ các run trên train, giữ mỗi (câu hỏi, nhánh) một chương trình mà đáp án thực thi **đúng** theo bộ
  chấm (ngắn nhất).
- **Truy hồi:** mỗi câu mới lấy 2 ví dụ gần nhất của DN khác, theo TF-IDF từ + bigram (bỏ dấu, số → 0), theo đúng nhánh:
  - nhánh A: chương trình trên bảng chuẩn C — tên cột giống nhau ở mọi DN nên chuyển giao trực tiếp;
  - nhánh B: chương trình trên bảng thô T.
- **Thư viện v4:** 4228 chương trình, phủ 2446/3000 câu train (`src/shots/train_shots.jsonl`). Chỉ dùng train (luật 6).

## 4. Kết quả thực nghiệm

Qwen3-14B-AWQ, K = 4 mẫu (AABB: 2 trên C, 2 trên T; mẫu đầu mỗi nhánh greedy), 2 vòng sửa lỗi, tự kiểm tra khi kết quả 0 /
rỗng, 4×L4 DP.

### 4.1 Tiến trình (mỗi dòng so trên cùng DN với dòng trên)

| Cấu hình | Tập | PA dev | PA holdout | top-answer đúng |
|---|---|---|---|---|
| Path B thuần, code khung chat 1 (smoke) | 30 câu | −0.40 | | 4/4 phiếu chỉ đúng 40 % |
| + `build_sales` + compile rút gọn (orders + lines) | 2251 câu / 22 DN | +0.19 (τ 0.75) | +0.10 | 1004/2251 |
| + norm-q, khách lẻ, `conv`, auto-header (e5) | 474 câu | +0.36 | | 293/474 |
| + **shots** (e6a, cùng code, cùng kết quả compile) | 474 câu | **+0.58** | | **370/474** |
| **Cấu hình tốt nhất, toàn bộ train** (hợp e6a, e7, e7b, e8, e9) | **3000 câu** | **+0.471** | **+0.437** | **2130/3000 (71 %)** |

### 4.2 Theo type (cấu hình tốt nhất, 3000 câu)

| Type | PA |
|---|---|
| number | +0.48 |
| percent | +0.50 |
| date | +0.65 |
| string | +0.53 |
| boolean | +0.83 |
| list | +0.22 |
| table | −0.16 |
| unanswerable | **−0.29** |

### 4.3 Kết quả phụ

- **Compile + 2 nhánh tốt hơn chỉ bảng thô:** trên cùng 1502 câu, AABB dev +0.22 vs BBBB +0.17; top-answer đúng 729 vs 663.
- **Độ bền compile:**
  - Bản cũ (khung chat 1): 5/40 ứng viên.
  - Với helper và các sửa lỗi: e9 compile được 10/11 DN; e7b 5/5; e8 6/6.
  - Lỗi compile là nguồn biến thiên lớn nhất giữa các run: một DN compile hỏng mất phần lớn lợi ích của DN đó.
- **Hiệu chuẩn** (e3h, 2251 câu, độ đúng của đáp án đứng đầu theo tỷ lệ phiếu):

  | Tỷ lệ phiếu | 1/4 | 2/4 | 3/4 | 4/4 |
  |---|---|---|---|---|
  | Độ đúng | 24 % | 40 % | 62 % | 80 % |

  Với shots, độ đồng thuận và độ đúng cùng tăng; τ = 0.5 tốt nhất (holdout +0.437 vs +0.426 ở τ 0.75).
- **Thinking (Qwen3 `--think`), chỉ trên T4, không shots:**
  - Độ đúng từng mẫu: dev 77/130 vs 52/130 không think; holdout 26/60 vs 20/60.
  - Nhưng vẫn thấp hơn cấu hình có shots không think (45 % vs 51 % từng mẫu trên 60 câu holdout trùng).
  - **Cộng dồn think + shots có lợi:** e10l (L4, 316 câu, cùng câu với e7) PA dev +0.69 vs +0.59, holdout +0.90 vs +0.70, top-answer 272 vs 253; e10 (T4, 40 câu) từng mẫu 84 % vs 69 %. Chậm ≈ 2.5× (~47 s/câu/L4). e10l cũng có code / shots mới hơn e7 nên chênh lệch không hoàn toàn do thinking. Holdout nhỏ (e10b, T4, 40 câu) cho +0.00 vs +0.30, nhưng **phép thử quyết định e11 (L4, 354 câu holdout, cùng câu với e8): PA +0.514 vs +0.254** (number +0.57 vs +0.25, percent +0.50 vs +0.24, unanswerable 9/9 đúng). e11 khác e8 cả ở code (khoảng ngày dữ liệu, SafePool) và shots (v4 vs v2) → +0.26 là của cả gói; phần riêng của thinking chưa tách.

## 5. Kết quả âm / trung tính (đã loại, có lý do)

| Thử | Kết quả | Quyết định |
|---|---|---|
| Ngưỡng τ riêng theo type, chọn trên dev | holdout 0.096 vs 0.101 (τ chung) | dùng τ chung |
| Trọng số phiếu nhánh A > B (0.5–3) | dev không đổi (+0.187), holdout dao động trong nhiễu | phiếu đều |
| Chỉ bảng thô (BBBB) | kém AABB (mục 4.3) | giữ compile |
| Compile đầy đủ (items / purchases / payments) | 5/40 (khung chat 1); smoke 2 h chưa xong 30 câu | dùng `--compile-scope sales` |
| norm-q + khách lẻ + conv riêng lẻ (e4) | top-answer bằng e3h (172/316) vì biz_116 compile hỏng ở e4 | giữ (có ích khi compile được), sửa compile |

## 6. Lỗi còn lại (từ e7 / e8 / e9, theo tần suất)

1. **`unanswerable` sai** (70 câu đáp án unanswerable, chỉ 22 đúng):
   - kỳ nằm ngoài dữ liệu → trả 0;
   - thông tin không được ghi (kênh COD, ví điện tử, địa chỉ đường, "giao trễ") → trả 0;
   - ngược lại, 34 lần trả unanswerable sai, chủ yếu câu **giá nhập bình quân** (purchases không có trong C, NOTES của
     compile nói "không có dữ liệu nhập");
   - đã sửa (khoảng ngày dữ liệu, NOTES không được gọi bảng T là thiếu) nhưng **chưa đo**.
2. **Xếp hạng bán chạy sai** khi `qty_base` quy đổi sai (đơn vị m / bao / viên trộn lẫn).
3. **Kỳ tương đối:** "tuần rồi", "tuần này", ca sáng / chiều (thiếu `time=` trong compile; đã thêm gợi ý, chưa đo).
4. **Câu nghiệp vụ hiếm:**
   - đếm "mặt hàng ế";
   - tồn kho theo đơn vị lớn [chẵn, lẻ];
   - hiệu ứng tăng giá;
   - đợt giao (đã cải thiện nhờ shots).

## 7. Hạn chế và điều kiện kiểm chứng

- **Chưa có điểm public LB.** Run public `p1` (cấu hình không think) chỉ xong 764/1000 câu (một shard kẹt); compile hỏng 4/14 DN public → dữ liệu / rulebook public khó hơn train, ước lượng train không chuyển thẳng sang public.
- **Thinking:** gói code mới nhất + shots v4 + thinking có lợi rõ trên holdout (e11: +0.51 vs +0.25, 354 câu); phần riêng của thinking chưa tách. Thời gian 52–98 s / câu / L4 → sát giới hạn 6 giờ nếu private ~2000 câu.
- **Holdout chỉ 10 DN, 1024 câu.** Khác biệt < ~0.03 PA giữa hai cấu hình là trong nhiễu. Các run khác nhau dùng thư viện
  shots v1–v3 (lớn dần) và code tăng dần, nên số "toàn train" là **hợp nhiều run**, không phải một run duy nhất.
- **Bộ làm sạch và prompt được phát triển khi nhìn train + public.** Bằng chứng độc lập cho dữ liệu chưa thấy chỉ có:
  - bộ test tổng hợp;
  - giao thức leave-business-out.

  Private test có ngành mới, nên luật ngành mới phải được đọc từ rulebook.
- **Chưa đo trên phần cứng chấm:**
  - tất định (|ΔPA| ≤ 0.03);
  - thời gian trên 2×RTX 5090;
  - mô hình FP8 (L4 chỉ chạy AWQ).
