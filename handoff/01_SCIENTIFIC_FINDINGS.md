# 01 — Scientific Findings (DataMate, Zalo AI Challenge 2026)

*Bàn giao ngày 2026-10-30 (branch `datamate`, repo `lhaochen118-ship-it/ZC`). Tài liệu này nêu đóng góp, ý nghĩa, kết
quả tích cực và tiêu cực, bằng chứng và điều kiện kiểm chứng. **Chưa có điểm PA nào đo được trên dữ liệu có đáp án**
(các run có đáp án vẫn đang chạy lúc bàn giao, xem `RUNNING_KERNELS.md`).*

## 1. Bài toán và hệ quả của thước đo

- Agent trả lời câu hỏi tiếng Việt đời thường trên dữ liệu bảng bẩn (CSV/Excel/SQLite) của hộ kinh doanh nhỏ, hoặc
  trả `unanswerable` / `abstain`. Điểm là **Penalized Accuracy**: đúng +1, abstain 0, sai −1.
- **Hệ quả quyết định:** trả lời khi và chỉ khi xác suất đúng ước lượng > 0.5. Mọi thiết kế phải có ước lượng xác suất
  đúng có hiệu chuẩn; hiện dùng **tỷ lệ phiếu** của K chương trình độc lập làm ước lượng.
- Train 3,000 câu / 30 DN (có đáp án), public 1,000 / 14 DN. Train: number 63 %, percent 19 %, list 6 %, date 5 %,
  string 3 %, unanswerable 2.3 %, boolean, table < 1 %. Khoảng 40 % câu là doanh thu.
- **Rulebook khác nhau giữa các split** (public thêm định nghĩa mới), private test có ngành mới → hệ thống phải đọc
  rulebook và README lúc chạy; không thể mã hoá cứng luật của train.

## 2. Đóng góp đã kiểm chứng

### 2.1 Bộ làm sạch tổng quát (`src/ingest.py`)

Chỉ dựa trên quy ước chung, không theo tên DN / cột / file cụ thể:

- dò dòng header (mật độ ô chữ), bỏ header lặp và dòng "Tổng…/Ghi chú…";
- ghép header 2 tầng có tên con trùng;
- số kiểu Việt và kiểu Mỹ: `10.000đ`, `105k`, `1tr440`, `1,134tr`, `531 500`, `VND 1,234`, `(1.000)`, `12,5%`, `1 bao 25kg`
  (giá trị + cột đơn vị `__unit`);
- quy ước dấu thập phân suy theo **từng cột**, chỉ dùng cho giá trị mơ hồ kiểu `1.500`; cột tiền giữ quy tắc theo từng giá trị;
- ngày: `dd/mm/yyyy`, ISO, `1 thg 6 2025`, `Ngày 1 tháng 6 năm 2025`, serial Excel (cả dạng chuỗi trong cột ngày),
  `dd/mm` + năm lấy từ tên sheet / file / dòng tiêu đề;
- mã (số 0 đầu, IMEI, ≥ 12 chữ số) và địa chỉ giữ dạng chữ;
- gộp các file / sheet của cùng một danh sách.

| Bằng chứng | Kết quả |
|---|---|
| Audit "cột chỉ đọc được một phần" trên dữ liệu thật | 78 → **0** (train); public còn 2 cột IMEI, đúng là chữ |
| Bộ test tổng hợp `tests/test_ingest_robust.py`: 12 biến thể bẩn không lấy từ dữ liệu thi | **12/12** khôi phục đúng giá trị |
| So từng giá trị giữa 2 phiên bản cache trên 44 DN | phát hiện lỗi **`1,134tr` → 1,134 tỷ** (đúng: 1.134 triệu) ở 13 DN; sau sửa chỉ còn đúng các thay đổi mong muốn |
| 3 chương trình viết tay trên bảng đã làm sạch (`biz_101`) | **khớp tuyệt đối** đáp án BTC (số đơn hôm qua, doanh thu hôm nay, ngày doanh thu cao nhất tháng) |

Ý nghĩa: lớp dữ liệu đủ để ra đáp án chính xác tuyệt đối; sai số còn lại nằm ở bước sinh chương trình.

### 2.2 Bộ chấm cục bộ đúng luật BTC (`src/score.py`)

Chuẩn hoá chuỗi (NFC, lowercase, bỏ dấu câu, giữ dấu tiếng Việt), alias theo phần tử / theo dòng, list có thứ tự,
table so cột theo tên rồi theo vị trí, hàng như tập hợp, số phải là JSON number.

- Chấm đáp án chuẩn với chính nó: PA = 1.0 (3000/3000).
- 167/167 câu trả lời bằng alias / mã được chấm đúng.
- Số ghi dạng chuỗi bị chấm sai, đúng rulebook §9.

### 2.3 Kiến trúc agent (`src/agent.py`), mới kiểm chứng ở mức đường ống

1. **Compile** (một lần mỗi DN): LLM viết chương trình dựng bảng chuẩn `C` (orders / lines / items / purchases /
   payments + NOTES về thông tin không có). Kiểm tra schema; lỗi được gửi lại để sửa. Chọn ứng viên có chữ ký tháng
   đồng thuận nhất.
2. **Answer**: K chương trình mỗi câu chia hai nhánh (A dùng `C`, B dùng bảng làm sạch `T`); sửa lỗi theo traceback;
   tự kiểm tra khi kết quả 0 / rỗng.
3. **Vote** theo đúng luật khớp của BTC; abstain dưới ngưỡng τ.
4. Formatter tất định; ghi đáp án theo từng nhóm DN; ngân sách thời gian (luật 8).

Test đầu-cuối không GPU (`tests/test_agent_local.py`, FakeEngine): đúng 3/3, bắt được compile sai (trạng thái chưa
ánh xạ) và sửa, chạy đúng vòng sửa lỗi và vòng tự kiểm tra; nhánh hết thời gian cho abstain đủ dòng.

## 3. Kết quả thực nghiệm (tích cực và tiêu cực)

**E2b public test** (Qwen3-14B-AWQ, 4×L4, `--compile --k 4 --kc 4`; 746/1000 câu có log vì 1/4 shard chết; không có
đáp án nên không có PA):

| Đo | Giá trị | Ý nghĩa |
|---|---|---|
| Mẫu chạy thành công sau sửa lỗi | 2787/2984 (93 %); lần đầu 2372 (79 %) | vòng sửa lỗi có ích (+14 điểm %) |
| Tỷ lệ phiếu của đáp án đầu | 4/4: 150 câu (20 %); 3/4: 183; 2/4: 342 (46 %); 1/4: 69 | **độ nhất quán thấp** → nhiều câu có thể sai; τ = 0.5 có thể quá dễ dãi |
| Compile thành công | 5/40 ứng viên; 3/14 DN có bảng chuẩn | **tiêu cực**: bước compile quá khó với 14B ở dạng hiện tại |
| Vòng tự kiểm tra (kết quả 0/rỗng) | 246 lần | cần đo xem có cải thiện không (chưa có đáp án) |
| Thông lượng | 110–170 token/s mỗi L4 (kỳ vọng ≥ 1,500) | **tiêu cực**: prefix cache bị đẩy ra liên tục (đã sửa, chưa đo lại) |

**Lỗi chương trình phổ biến nhất (E2b):**

| Lỗi | Số lần |
|---|---|
| `.str` trên cột không phải chuỗi | 101 |
| tạo ngày không tồn tại (`2026-02-29`, "day is out of range") | ~165 |
| tự đọc file `orders.csv` dù đã được dặn | 52 |
| không gán `ANSWER` | 51 |
| sai tên cột | 33 |

Đây là mục tiêu cải tiến trực tiếp: helper ngày an toàn, nhắc `.astype(str)`, chặn đọc file.

**Hạ tầng (kết quả âm):**

- Qwen3.5-9B không khởi tạo được trên T4.
- 14B-AWQ trên một T4 không đủ KV cho prompt ~16k (DP trên T4 không dùng được).
- FlashInfer sampler JIT hỏng trên L4.
- Chi tiết ở PDF 2.

## 4. Hạn chế và điều kiện kiểm chứng

- **Chưa có PA trên train** cho bất kỳ cấu hình LLM nào. Mọi kết luận về độ chính xác đang chờ e1b / e1c / e2b
  (`RUNNING_KERNELS.md`); e1b, e1c, e2b dùng code cũ còn lỗi (xem bảng ở đó), nên số của chúng là cận dưới.
- Ngưỡng τ, tỷ lệ K giữa hai nhánh, có giữ compile hay không phải chọn trên DN dev (`biz_101`–`biz_120`) và báo cáo
  trên holdout (`biz_121`–`biz_130`); public chỉ để đối chiếu.
- Bộ làm sạch được phát triển nhìn cả train và public (audit chạy trên cả hai); bộ test tổng hợp là bằng chứng độc lập
  duy nhất cho tính tổng quát.
- Chưa chạy thử Docker trên GPU Blackwell; tất định (|ΔPA| ≤ 0.03) chưa đo.
