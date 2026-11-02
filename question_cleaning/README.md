# Làm sạch bộ câu hỏi (question cleaning)

Module: [`src/qnorm.py`](../src/qnorm.py) (agent dùng trực tiếp; thư mục này chỉ chứa dữ liệu đã làm sạch + script tạo lại).
Chỉ dùng **luật ngôn ngữ chung** của tiếng Việt chat, không có gì theo doanh nghiệp hay id câu hỏi.

| File | Nội dung |
|---|---|
| `train_questions_clean.jsonl` | **3000 câu train** đã làm sạch: `id`, `business_id`, `as_of_date`, `question` (gốc), `clean`, `key`, `period`, `changes` |
| `stats.json` | Số câu thay đổi, số câu theo từng quy tắc, số câu nhận ra kỳ thời gian |
| `make.py` | Tạo lại hai file trên: `python3 question_cleaning/make.py data/raw/train/train [--cache <ingest cache>]` |

## Các bước (theo thứ tự, mỗi thay đổi ghi trong `changes`)

Từ vựng của bước khôi phục dấu chỉ học từ câu **gõ có dấu** (≥ ½ số từ có dấu): câu gõ thiếu dấu từng làm bẩn từ vựng
(`truoc`, `nhien`) nên bị loại.

1. **Unicode**: NFC, bỏ emoji / ký hiệu trang trí, gộp khoảng trắng.
2. **Viết tắt chat** (cả từ, không ăn vào mã như `NV001`): `dthu` → doanh thu, `bnhiu`/`bn` → bao nhiêu, `hqua` → hôm qua,
   `hnay` → hôm nay, `đc`/`dc` → được, `ko`/`hok` → không, `trc` → trước, `vs` → với, `z` → vậy, `sp` → sản phẩm,
   `ck` → chuyển khoản, `tm` → tiền mặt… (`dt` mơ hồ — doanh thu / điện thoại — nên giữ nguyên).
3. **Lỗi gõ Telex**: `toongr` → tổng, `dduwowcj` → được, `tir leej` → tỉ lệ, `thangs` → tháng. Chỉ nhận khi từ giải mã là
   **từ có thật** trong từ vựng; thử mọi vị trí dấu (vì phụ âm `gi`, `qu`: `gias` → giá, không phải gía).
4. **Từ địa phương** chỉ ở ngữ cảnh không mơ hồ: `tháng ni` → tháng này, `thợ mô` → thợ nào, `rứa` → vậy, `chừ` → giờ.
   Không đổi `răng` (= răng, "Kem đánh răng"), không đổi `mo` đứng sau từ không phải danh từ hỏi (`giờ mở cửa`).
5. **Khôi phục dấu** cho câu gõ không dấu (câu đã có dấu thì giữ nguyên, để không đổi tên riêng như "Huy" → "Huỷ"):
   Viterbi trên các dạng có dấu của từ vựng, mô hình bigram có làm trơn. Từ vựng = rulebook của split + câu train có dấu
   (luôn **bỏ câu của chính doanh nghiệp** khi chấm) + **tên trong dữ liệu của doanh nghiệp** (hàng, nhóm, khách…, trọng số 2)
   — nhờ đó `nuoc lau san sunlight` → nước lau sàn sunlight (không phải "nước lẩu sản"). Từ lạ (mã, tên không có trong dữ
   liệu) giữ nguyên.
6. **Tháng viết tắt**: `th7/2025`, `T11/2025` → tháng 7/2025; `6-2026` → tháng 6/2026; `4/26` → tháng 4/2026 (chỉ khi số sau
   là năm 2 chữ số 23–29; `1/1` giữ nguyên); `th trước` → tháng trước.
7. **Tên theo dữ liệu (entity linking)**: đoạn câu trùng (bỏ dấu, hoa thường, dấu câu) với một tên ≥ 2 từ trong dữ liệu
   của doanh nghiệp được ghi **đúng như dữ liệu** (`cao thu hanh` → Cao Thu Hạnh, `snack lay's vi tu nhien 56g` → snack
   Lay's vị tự nhiên 56g); nhãn thường (`Ghi nợ`) giữ chữ thường của câu.
8. **Luật ngữ cảnh** (chỉ ở ngữ cảnh an toàn): `nhiêu` → nhiều trước nhất / hơn / hay ít; `hà` cuối câu → hả; `lệ` → lẻ
   (trừ tỉ lệ); `j` → gì; `k` → không (trừ sau số: `50 k`); `len` sau giá → lên; `tạ` cuối câu → ta; `là chi` → là gì.
9. **Kỳ thời gian** (`period`): kỳ đầu tiên trong câu theo rulebook §2 (`qparse.parse_period`), ra khoảng ngày ISO và biểu thức
   toolkit, ví dụ `{"start": "2025-10-01", "end": "2025-10-31", "code": "month(2025, 10)"}`.

Câu gốc luôn là chuẩn: agent đưa `clean` (và kỳ) **bên cạnh** câu gốc cho LLM, không thay thế.

## Bước LLM (đang chạy: `src/qnorm_llm.py`)

Chính Qwen3.5-9B của giải pháp (không thêm mô hình) thêm dấu theo ngữ cảnh cho các từ luật chưa chắc (mỗi / mọi, mà / mã,
chở / cho, vì / vỉ, lỗi / lời …). Kiểm tra cứng: bỏ dấu đầu ra phải ra **đúng** chuỗi từ đầu vào (không thêm / bớt / đổi từ,
số, mã), từ người dùng đã gõ dấu giữ nguyên; tên giữa câu và mã không bị LLM đổi. Job đo luôn độ chính xác của LLM so với
mô hình bigram trên 2232 câu có dấu (bỏ dấu rồi khôi phục).

## Kết quả trên 3000 câu train

- 1620 câu được thay đổi; 2976 / 3000 câu nhận ra kỳ thời gian.
- Số câu theo quy tắc: abbr 1256, accent 790, telex 235, entity 149, context 63, month 35 (`stats.json`).
- **Độ chính xác khôi phục dấu** (leave-one-business-out: lấy 2232 câu có dấu, bỏ dấu, khôi phục, so với câu gốc):
  **từ 99.66 %**, từ có dấu 99.60 %, **nguyên câu đúng 96.1 %** (lần 1: 99.3 % / 93.6 %).

## Ví dụ (ngẫu nhiên)

| Quy tắc | Câu gốc | Câu đã làm sạch | Kỳ (period) |
|---|---|---|---|
| telex | tháng 1/2025 tir lệ đơn bij huỷ là bn % | tháng 1/2025 tỉ lệ đơn bị huỷ là bao nhiêu % | 2025-01-01..2025-01-31 |
| telex | coi giumf tieenf bán Trà vải (L) tháng 3/2025 là bao nhiêu | coi giùm tiền bán Trà vải (L) tháng 3/2025 là bao nhiêu | 2025-03-01..2025-03-31 |
| telex | tháng 8/2025 xe ddax giao total mấy đợt hàng | tháng 8/2025 xe đã giao total mấy đợt hàng | 2025-08-01..2025-08-31 |
| telex | tháng 4/2026 cửa hàng bán dduwowcj tổng cộng bao nhieeu tiền | tháng 4/2026 cửa hàng bán được tổng cộng bao nhiêu tiền | 2026-04-01..2026-04-30 |
| telex | đếm soos đơn ddax huỷ trong tháng 6/2025 | đếm số đơn đã huỷ trong tháng 6/2025 | 2025-06-01..2025-06-30 |
| abbr | doanh thu thang 4/2026 hon hay kem thang trc bao nhieu % | doanh thu tháng 4/2026 hơn hay kém tháng trước bao nhiêu % | 2026-04-01..2026-04-30 |
| abbr | tháng 9/2025 các mặt hàng nhóm lẩu thu đc tổng cộng bao nhiêu tiền | tháng 9/2025 các mặt hàng nhóm lẩu thu được tổng cộng bao nhiêu tiền | 2025-09-01..2025-09-30 |
| abbr | Plusssz Max Multivitamin viên sủi tuýp 20 viên lên giá 44.000đ từ ngày 20/11/2025; so 7 ngày kể từ ngày tăng giá vs 7 ngày ngay truwowcs, lượng bán tính theo tuýp tăng hay giảm bnhiu % z | Plusssz Mã Multivitamin viên sủi tuýp 20 viên lên giá 44.000đ từ ngày 20/11/2025; so 7 ngày kể từ ngày tăng giá với 7 ngày ngay trước, lượng bán tính theo tuýp tăng hay giảm bao nhiêu % vậy | 2025-11-20..2025-11-20 |
| abbr | thang 9/2025, dthu cac don ban qua dien thoai la bao nhieu, doanh thu cac don ban qua zalo oa la bao nhieu | tháng 9/2025, doanh thu các đơn bán qua điện thoại là bao nhiêu, doanh thu các đơn bán qua zalo oa là bao nhiêu | 2025-09-01..2025-09-30 |
| abbr | ti le huy don thang 3/2025 la bao nhiu % | tỉ lệ huỷ đơn tháng 3/2025 là bao nhiêu % | 2025-03-01..2025-03-31 |
| accent | dem so mat hang co mat o ≥ 112 don khac nhau trong thang 10/2025 | đếm số mặt hàng có mặt ở ≥ 112 đơn khác nhau trong tháng 10/2025 | 2025-10-01..2025-10-31 |
| accent | nhung sp nao dang con hang co han dung tu 29/08/2026 den het 27/12/2026 | những sản phẩm nào đang còn hàng có hạn dùng từ 29/08/2026 đến hết 27/12/2026 | 2026-08-29..2026-12-27 |
| accent | e doanh thu tu dau thang nay den het hom nay hon hay kem cung ky th truoc bn % | ê doanh thu từ đầu tháng này đến hết hôm nay hơn hay kém cùng kỳ tháng trước bao nhiêu % | 2025-11-01..2025-11-27 |
| accent | doanh thu thang 4/2025 tang hay giam bao nhieu phan tram so vs thang trc nhi | doanh thu tháng 4/2025 tăng hay giảm bao nhiêu phần trăm so với tháng trước nhỉ | 2025-04-01..2025-04-30 |
| accent | dthu ngay nao cao nhat trong thang 3 nam 2026 | doanh thu ngày nào cao nhất trong tháng 3 năm 2026 | 2026-03-01..2026-03-31 |
| accent | check gium thang 10/2025, dthu cac don tra bang chuyen khoan va cac don tra bang tm moi loai bnhiu | check giùm tháng 10/2025, doanh thu các đơn trả bằng chuyển khoản và các đơn trả bằng tiền mặt mọi loại bao nhiêu | 2025-10-01..2025-10-31 |
| accent | so bo van day gas namilux ban duoc trong thang 12/2025 la bn | số bộ van dây gas namilux bán được trong tháng 12/2025 là bao nhiêu | 2025-12-01..2025-12-31 |
| month | dthu thang 8/2025 hon hay kem th truoc bao nhieu % | doanh thu tháng 8/2025 hơn hay kém tháng trước bao nhiêu % | 2025-08-01..2025-08-31 |
| month | dthu tháng 8/2025 tăng hay giảm bao nhiêu phần trăm so với th trước | doanh thu tháng 8/2025 tăng hay giảm bao nhiêu phần trăm so với tháng trước | 2025-08-01..2025-08-31 |
| month | dthu tháng 3/2026 hơn hay kém th trước bn % | doanh thu tháng 3/2026 hơn hay kém tháng trước bao nhiêu % | 2026-03-01..2026-03-31 |
