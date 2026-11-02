# Làm sạch bộ câu hỏi (question cleaning)

Thư mục này gom riêng phần làm sạch câu hỏi để xem độc lập với mô hình: **module**, **3000 câu train đã làm sạch**,
thống kê và script tạo lại. Chỉ dùng **luật ngôn ngữ chung** của tiếng Việt chat cộng với tên có trong dữ liệu của chính
doanh nghiệp. Không có luật nào theo doanh nghiệp hay id câu hỏi.

| File | Nội dung |
|---|---|
| `train_questions_clean.jsonl` | **3000 câu train** đã làm sạch, mỗi dòng: `id`, `business_id`, `as_of_date`, `question` (gốc), `clean`, `clean_rules`, `source`, `key`, `period`, `changes` (và `clean_llm`, `llm_check` khi đã ghép bước LLM) |
| `stats.json` | số câu đổi, số câu theo từng quy tắc, số câu nhận ra kỳ thời gian |
| `module/` | **module làm sạch**: `qnorm.py` (luật), `qnorm_llm.py` (bước LLM + kiểm tra cứng), `qnorm_phrases.txt` (cụm từ chung), `qnorm_train_questions.txt` (câu train có dấu dùng làm từ vựng). Bản sao **giống hệt** `src/`: `tests/test_qnorm.py` báo lỗi nếu khác |
| `make.py` | tạo lại tất cả: `python3 question_cleaning/make.py data/raw/train/train [--cache <ingest cache>] [--llm <questions_clean.jsonl của qnorm_llm>]` (CPU, khoảng 10 giây) |

Dùng trong code:

```python
import qnorm
nz = qnorm.Normalizer.for_split(split_dir, extra_texts=qnorm.Normalizer.entity_texts(business))
r = nz.clean(question, as_of_date)  # r["clean"], r["key"], r["period"], r["changes"]
```

Câu gốc luôn là chuẩn. Agent đưa `clean` và kỳ thời gian **bên cạnh** câu gốc cho LLM, không thay thế câu gốc. Bộ đọc luật
`qparse` so khớp trên `key` (bỏ dấu, chữ thường).

## Các bước (theo thứ tự; mỗi thay đổi ghi vào `changes` dạng `[từ, thành, quy tắc]`)

1. **Unicode**: chuẩn NFC, bỏ emoji và ký hiệu trang trí, gộp khoảng trắng.
2. **Bảo vệ tên và mã.** Hai loại từ được giữ đúng như người dùng gõ, không bước nào sau đây được sửa:
   - từ thuộc một tên trong dữ liệu (gõ cạnh từ đứng kề nó trong tên đó): `Plusssz Max`, `Sino SP D20`, `Vitamin K2`,
     `Galaxy Z`;
   - chữ dính vào số (đơn vị, mã): `2.92m`, `60x60`, `CT-11A`, `NV001`, `500ml`.

   Riêng `35.000d` được đổi thành `35.000đ`.
3. **Viết tắt chat** (chỉ thay cả từ): `dthu` → doanh thu, `bnhiu`/`bn` → bao nhiêu, `hqua` → hôm qua, `hnay` → hôm nay,
   `đc`/`dc` → được, `ko`/`hok`/`hông` → không, `trc` → trước, `vs` → với, `z` → vậy, `sp` → sản phẩm,
   `ck` → chuyển khoản, `tm` → tiền mặt…
   - Giữ nguyên `dt` vì mơ hồ (doanh thu hay điện thoại).
   - Giữ nguyên từ VIẾT HOA có trong tên của dữ liệu (`SP` trong "Sino SP D20").
4. **Lỗi gõ Telex**: `toongr` → tổng, `dduwowcj` → được, `tir leej` → tỉ lệ, `thangs` → tháng.
   - Chỉ nhận khi từ giải mã ra là **từ có thật** trong từ vựng.
   - Thử mọi vị trí dấu, vì có phụ âm đầu `gi`, `qu`: `gias` → giá, không phải gía.
   - Không giải mã từ viết hoa giữa câu (tên, nhãn hàng: `Max` không thành "Mã").
5. **Từ địa phương**, chỉ ở ngữ cảnh không mơ hồ: `tháng ni` → tháng này, `thợ mô` → thợ nào, `rứa` → vậy, `chừ` → giờ.
   - Không đổi `răng`: có thể là răng thật, như trong "Kem đánh răng".
   - Không đổi `mo` khi đứng sau từ không phải danh từ để hỏi (`giờ mở cửa`).
6. **Khôi phục dấu** cho câu gõ không dấu (dưới 15 % số từ có dấu). Câu đã gõ có dấu thì giữ nguyên, để không đổi tên riêng
   như "Huy" thành "Huỷ".
   - Thuật toán: Viterbi trên các dạng có dấu của từ vựng, mô hình bigram có làm trơn.
   - Từ vựng gồm: rulebook của split; câu train **gõ có dấu** (từ ½ số từ có dấu trở lên); cụm từ chung
     `qnorm_phrases.txt`; tên trong dữ liệu của doanh nghiệp (trọng số 2).
   - Khi chấm, luôn **bỏ câu của chính doanh nghiệp đó** ra khỏi từ vựng.
   - Câu gõ thiếu dấu bị loại khỏi từ vựng vì từng làm bẩn nó (`truoc`, `nhien`). Từ Telex và từ viết tắt cũng bị loại.
   - Từ lạ (mã, tên không có trong dữ liệu) giữ nguyên.
7. **Luật ngữ cảnh**, mỗi luật chỉ chạy ở ngữ cảnh an toàn:
   - `nhiêu` → nhiều khi đứng trước nhất / hơn / hay ít; các trường hợp khác `nhiêu` → bao nhiêu.
   - `nhieu` gõ không dấu → bao nhiêu nếu câu không có từ hỏi nào khác: "bán được nhieu tiền" hỏi số tiền. Có
     nhất / hơn / không / nào… thì vẫn là "nhiều".
   - Từ cuối câu: `hà` / `ha` → hả; `tạ` → ta; `là chi` → là gì; `có … hong` → có … không.
   - `e` đầu câu → ê (tiếng gọi), không phải "ế".
   - `lệ` → lẻ, trừ trong "tỉ lệ".
   - `j` → gì và `k` → không: chỉ khi gõ chữ thường và không đứng sau số (`50 k` là 50 nghìn, `Vitamin K` giữ nguyên).
   - `len` sau giá tiền → lên.

   Các luật sau chỉ áp dụng cho từ người dùng **gõ không dấu**:
   - `mọi loại … bao nhiêu` → mỗi;
   - `số N ngày … với` → so;
   - `, tục` → tức;
   - `nhưng` + danh từ → những;
   - `vị` + giao / lỗi / … → vì;
   - `mã khách chưa` → mà.
8. **Tháng viết tắt**: `th7/2025`, `T11/2025` → tháng 7/2025; `6-2026` → tháng 6/2026; `th trước` → tháng trước.
   - `4/26` → tháng 4/2026, chỉ khi số sau là năm 2 chữ số từ 23 đến 29.
   - `1/1` giữ nguyên.
9. **Tên theo dữ liệu (entity linking).** Áp dụng cho đoạn câu trùng với một tên trong dữ liệu khi so bỏ dấu, không phân
   biệt hoa thường và bỏ qua dấu câu. Đoạn phải trùng cả tên (từ 2 từ) hoặc từ 3 từ liên tiếp của tên.
   - Đoạn đó nhận **dấu của dữ liệu**, cả với từ dữ liệu cố ý viết không dấu: `nuoc lau san sunlight` → nước lau sàn
     sunlight (không phải "lẩu"); `men vi sinh` (không phải "mến vỉ"); `bia heineken lon` (không phải "lớn").
   - Chỉ dấu thay đổi. **Chữ hoa / thường và dấu câu giữ như người dùng gõ**, nên không còn "BÁNH CROISSANT" hay "Dhg Hộp".
   - **Từ người dùng đã gõ dấu là chuẩn.** "khách Lê Bảo Hoàng" không bị gắn vào nhãn "Khách lẻ".
   - Từ viết hoa giữa câu không gắn vào nhãn mà dữ liệu viết chữ thường.
10. **Kỳ thời gian** (`period`): kỳ đầu tiên trong câu theo rulebook §2 (`qparse.parse_period`). Kết quả gồm khoảng ngày
    ISO và biểu thức toolkit, ví dụ `{"start": "2025-10-01", "end": "2025-10-31", "code": "month(2025, 10)"}`.

## Bước LLM (`module/qnorm_llm.py`; job đang chạy trên Kaggle, kernel zc-v4-5)

Bước này dùng chính Qwen3.5-9B của giải pháp, không thêm mô hình. Mô hình thêm dấu theo ngữ cảnh cho những từ mà bigram còn
mơ hồ: mỗi / mọi, mà / mã, chở / cho, lon / lớn, tên đường…

- **Kiểm tra cứng:** bỏ dấu đầu ra phải được **đúng** chuỗi từ đầu vào, không thêm, bớt hay đổi từ, số, mã.
- Từ người dùng đã gõ dấu giữ nguyên.
- LLM không được đổi tên viết hoa giữa câu, mã, số.
- Đầu ra hỏng thì giữ kết quả của luật.

`make.py --llm` ghép kết quả theo từng từ. Mọi câu mà luật và LLM khác nhau sẽ được duyệt tay trước khi dùng.

## Kết quả trên 3000 câu train (`stats.json`)

- 1614 câu được thay đổi; 2976 / 3000 câu nhận ra kỳ thời gian.
- Số câu theo từng quy tắc: abbr 1256, accent 790, telex 231, context 152, month 35, entity 9.
  - entity giờ chỉ đếm thay đổi dấu; trước đây đếm cả đổi hoa thường.
- **Đã duyệt tay toàn bộ** các thay đổi viết tắt / Telex / địa phương / ngữ cảnh / tên trên 3000 câu. Các lỗi tìm được đều
  đã sửa bằng luật chung và có test (`tests/test_qnorm.py`, 29 trường hợp):
  - `Max` → "Mã";
  - `SP` → "SẢN PHẨM";
  - "khách Lê" → "khách lẻ";
  - "BÁNH CROISSANT" (chữ hoa áp từ dữ liệu);
  - `e` đầu câu → "ế";
  - `CT-11A` → "CT-11Ạ".
- **Độ chính xác khôi phục dấu**, đo leave-one-business-out:
  - cách đo: lấy 2232 câu người dùng gõ có dấu, bỏ hết dấu, làm sạch lại rồi so với bản làm sạch của câu gốc; coi
    huỷ = hủy và không phân biệt hoa thường;
  - kết quả: **từ 99.79 %**, từ có dấu 99.74 %, **nguyên câu đúng 98.1 %** (2189 / 2231);
  - các lần trước: 93.6 % → 96.1 % → 96.75 % → 97.3 % (cách đo cũ) → 98.1 %.
- **42 câu còn sai** đều mơ hồ thật, luật không quyết được:
  - tên đường không có trong dữ liệu: Bạch Đằng, Hai Bà Trưng, Lê Trọng Tấn;
  - cặp từ cùng mặt chữ: lon / lớn, vỉ / vị, của / cửa, hướng / hương, chậm / chấm.

  Đây là phần dành cho bước LLM. Với câu thật gõ không dấu, sai ở mức này chỉ ảnh hưởng phần đọc của LLM; bộ đọc luật dùng
  `key` không dấu nên không bị ảnh hưởng.

## Ví dụ (từ `train_questions_clean.jsonl`)

| Quy tắc | Câu gốc | Câu đã làm sạch | Kỳ (period) |
|---|---|---|---|
| abbr | quy 2/2026 cua hang co bn don hang ta | quý 2/2026 cửa hàng có bao nhiêu đơn hàng ta | 2026-04-01..2026-06-30 |
| abbr | 7 ngày gần nhất cửa hàng có bnhiu bill | 7 ngày gần nhất cửa hàng có bao nhiêu bill | 2026-07-20..2026-07-26 |
| telex | tháng 8/2025 có nhiêu khách mua tuwf 12 đơn trở leen | tháng 8/2025 có bao nhiêu khách mua từ 12 đơn trở lên | 2025-08-01..2025-08-31 |
| telex | tháng 1/2025 tir lệ đơn bij huỷ là bn % | tháng 1/2025 tỉ lệ đơn bị huỷ là bao nhiêu % | 2025-01-01..2025-01-31 |
| telex + tên | Tồn kho Plusssz Max Multivitamin viên sủi tuýp 20 viên cuối ngày hôm nay là bao nhiêu tuýp vậy? | (giữ nguyên: `Max` thuộc tên hàng) | 2026-06-12..2026-06-12 |
| abbr + tên | Số cây Ống luồn dây điện Sino SP D20 cây 2.92m bán đc trong tháng 6/2025 là bao nhiêu | Số cây Ống luồn dây điện Sino SP D20 cây 2.92m bán được trong tháng 6/2025 là bao nhiêu | 2025-06-01..2025-06-30 |
| accent | dthu thang 2/2026 so vs cung ky nam ngoai thay doi bao nhieu % | doanh thu tháng 2/2026 so với cùng kỳ năm ngoái thay đổi bao nhiêu % | 2026-02-01..2026-02-28 |
| accent + tên | coi gium ton kho bio-acimin gold men vi sinh hop 30 goi cuoi ngay hom nay la bn goi | coi giùm tồn kho bio-acimin gold men vi sinh hộp 30 gói cuối ngày hôm nay là bao nhiêu gói | 2026-05-25..2026-05-25 |
| accent + tên | so thung Chong tham Kova CT-11A tuong 20kg ban duoc trong thang 9/2025 la bn | số thùng Chống thấm Kova CT-11A tường 20kg bán được trong tháng 9/2025 là bao nhiêu | 2025-09-01..2025-09-30 |
| context | tháng 1/2026 doanh thu bình quaan mooix đơn là nhiêu | tháng 1/2026 doanh thu bình quân mỗi đơn là bao nhiêu | 2026-01-01..2026-01-31 |
| context | e gia ban mot chai Siro an ngon Ich Nhi Plus chai 100ml vao ngay 09/08/2025 la bao nhieu | ê giá bán một chai Siro ăn ngon Ích Nhi Plus chai 100ml vào ngày 09/08/2025 là bao nhiêu | 2025-08-09..2025-08-09 |
| context | check gium thang 10/2025, dthu cac don tra bang chuyen khoan va cac don tra bang tm moi loai bnhiu | check giùm tháng 10/2025, doanh thu các đơn trả bằng chuyển khoản và các đơn trả bằng tiền mặt mỗi loại bao nhiêu | 2025-10-01..2025-10-31 |
| tên (dấu của người dùng) | Cho tôi biết tháng này khách Lê Bảo Hoàng nợ thêm bao nhiêu tiền, đã trả bớt nợ đc bao nhiêu | Cho tôi biết tháng này khách Lê Bảo Hoàng nợ thêm bao nhiêu tiền, đã trả bớt nợ được bao nhiêu | 2026-08-01..2026-08-05 |
| month | so vs th truoc do, dthu thang 1/2026 tang hay giam may phan tram | so với tháng trước đó, doanh thu tháng 1/2026 tăng hay giảm mấy phần trăm | 2026-01-01..2026-01-31 |
| month | từ đầu tháng này đến hết bữa nay, doanh thu thay đổi bao nhiu % so vs cùng kỳ th trước | từ đầu tháng này đến hết bữa nay, doanh thu thay đổi bao nhiêu % so với cùng kỳ tháng trước | 2025-08-01..2025-08-17 |
