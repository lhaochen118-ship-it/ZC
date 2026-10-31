# 04 — Pipeline v2: bộ công cụ nghiệp vụ (toolkit) + chương trình ngắn

*Phiên 2026-10-31. Branch `claude/funny-euler-ceut3o`. Phiên này **không có token Kaggle** trong container, nên mọi kết
quả dưới đây đo bằng CPU, không có LLM. Phần cần GPU đã viết và test bằng FakeEngine, chờ chạy trên Kaggle.*

## 1. Vì sao đổi hướng

v1: LLM 14B tự viết chương trình pandas dài trên bảng thô cho từng câu, tự diễn giải rulebook mỗi lần. Đo trên E2b/E2c:
chỉ 20–26 % câu 4/4 phiếu đồng ý, compile 3/14 DN, lỗi hàng đầu là cộng trừ ngày, tên cột, `.str`.

v2 tách bài toán:

1. **Ngữ nghĩa rulebook viết một lần, kiểm chứng bằng đáp án train** (`src/dm.py`): kỳ thời gian (tháng này / cùng kỳ
   tháng trước, tuần T2–CN, quý, Tết theo lịch âm, N ngày gần nhất...), đơn hợp lệ, doanh thu, xếp hạng + hoà theo mã,
   công nợ (tại ngày, tuổi nợ, nợ thêm / trả nợ), nhập hàng, tồn kho lùi từ ảnh chụp cuối kỳ, lô / hạn dùng, giá theo
   ngày, hiệu ứng tăng giá, lãi gộp / ròng, giao hàng; §7 "không có dữ liệu ≠ 0" → `Unanswerable`.
2. **Chuẩn hoá mỗi DN = chương trình mapping ngắn** (đổi tên cột → `finalize`): `finalize` làm các luật §8 (ô trống kế
   thừa dòng đầu đơn, kể cả mã đơn trống; dòng trùng trong cùng đơn; số lượng âm = trả hàng; quy đổi đơn vị; tên viết khác
   nhau là một). Nguồn chương trình mapping: (a) chương trình tham chiếu của 30 DN train + bộ chuyển đổi định dạng tổng quát
   `refs/formats/` (tìm bảng theo cột, không theo tên file), (b) LLM viết khi không có tham chiếu nào khớp, với 2 tham
   chiếu gần nhất làm ví dụ.
3. **Mỗi câu hỏi = chương trình 1–3 dòng gọi toolkit**: (a) bộ đọc mẫu câu theo luật (`src/qparse.py`) sinh chương
   trình tất định, (b) K chương trình do LLM viết với few-shot là các cặp (câu hỏi train → chương trình đã kiểm chứng đúng)
   gần nhất (`src/fewshot.py`, n-gram ký tự, không cần mô hình phụ). Bỏ phiếu: chương trình luật có trọng số `--qw`
   (mặc định 2) và thắng khi hoà.

Không có logic nào theo id câu hỏi / DN. Chương trình tham chiếu là mã tổng quát theo tên cột; khi chạy trên train, chương
trình của chính DN đó bị loại (leave-one-out) để số đo trung thực.

## 2. Kết quả đã kiểm chứng (CPU, không LLM)

| Thí nghiệm | Kết quả |
|---|---|
| Thư viện + mapping tham chiếu + bộ đọc mẫu câu, 3,000 câu train | lượt 1: 1,348 đúng / 65 sai; lượt 2: 2,519 / 152; lượt 3: 2,612 / 66; **lượt 4: 2,651 đúng / 51 sai / 298 chưa đọc** → PA 0.867 nếu abstain phần còn lại |
| Bộ chuyển đổi POS tổng quát (`refs/formats/pos_invoices.py`) thay cho 6 chương trình riêng (106/108/109/119/120/128) | tái lập đúng: 604 đúng / 17 sai (riêng: 603 / 18) |
| Chuyển giao mapping sang DN chưa thấy | public: 4–5/14 DN tự ánh xạ (dev_001/002/005/006); train leave-one-out: 11/30 (trước khi thêm bảng tuỳ chọn và bộ POS) |
| Agent v2 `--no-llm` trên public (CPU 4 nhân) | 1,000 câu trong 97 s; 4/14 DN tự ánh xạ; 245/326 câu của các DN đó có đáp án từ mẫu câu |
| `tests/test_agent_v2.py` (FakeEngine) | PASS: mapping lỗi → sửa, chương trình toolkit + luật, `Unanswerable`, log từng câu, chấm điểm |
| `tests/test_ingest_robust.py` | 13/13 (thêm ca "1 kg" ≠ 1000) |

Các câu sai còn lại đã xem gần như đều do bộ đọc mẫu câu (lỗi gõ, cách hỏi lạ), không do dữ liệu. Phần chưa đọc
(~11 %) là việc của LLM.

## 3. Lỗi dữ liệu tìm ra và sửa

- **`1 kg` bị đọc thành 1,000** (chữ `k` hiểu là nghìn) trong `ingest.parse_num` → mọi hàng bán theo kg ở biz_102/103/114/117,
  dev_001/004... sai số lượng. Sửa: hậu tố nhân chỉ khi đứng riêng (`29k`, `1tr205` vẫn đúng); thêm test.
- Nhãn trạng thái: `Không giao được` = huỷ, `hàng trả về`, `đang gửi`, nhãn tiếng Anh (`completed`, `cancelled`,
  `returned`, `shipping`); ô trạng thái trống ở dòng sau kế thừa dòng đầu đơn (trước đây mặc định "hoàn thành").
- Mã đơn trống ở dòng tiếp theo (biz_103/121/122: 60 % dòng) phải kế thừa đơn phía trên.
- Tên hàng / tên khách viết khác nhau (`GAS SAIGON PETRO 12KG` / `Gas  Saigon Petro 12kg`, `Dinh 5 phan` / `Đinh 5 phân`)
  phải gộp; khách lẻ viết `kl`; kênh `zalo_oa` (dấu `_`).
- Biển số xe `61C-750.95` đang bị đọc thành số 61 + đơn vị (ingest); chương trình tham chiếu lấy từ `RAW`. Chưa sửa ingest.

## 4. Cách chạy

Đánh giá tất định trên train (CPU, ~6 phút):

```bash
python3 tests/eval_lib.py data/raw/train/train <cache> --show 8 --dump <scratch>/fewshot.jsonl
```

Agent v2 (mặc định; `--v1` để chạy pipeline cũ):

```bash
python3 src/agent.py <split> <out> --model Qwen/Qwen3-14B-AWQ --k 4 --kc 2 --dp 4 --cache /tmp/zc_cache --max-len 24576 --max-tokens 1024
```

- `--no-llm`: chỉ mapping tham chiếu + chương trình luật (không nạp mô hình).
- `--qw 2`: trọng số phiếu của chương trình luật; `--shots 4`: số ví dụ few-shot; `--map-llm`: luôn sinh thêm mapping bằng LLM.

Kaggle (khi có token), thử nhỏ trước:

```bash
python3 automation/fanout.py <runs>/v2a zc-v2a "run_kaggle.py train OUT --model Qwen/Qwen3-14B-AWQ --k 4 --kc 2 --dp 4 --cache /tmp/zc_cache --max-len 24576 --max-tokens 1024 --subset 8" 1 --accounts nguyenhoa99 --l4 --ks nhatlong1103/zc-assets
```

## 5. Việc tiếp theo (theo thứ tự)

1. **Kaggle, train đầy đủ v2** (dev `biz_101`–`biz_120`, holdout `biz_121`–`biz_130`): đo PA, tỉ lệ mapping LLM qua
   check, đồng thuận LLM vs luật; chọn `--qw`, `--tau`, K trên dev. Chương trình mapping của chính DN bị loại tự động.
2. **Public test v2** → `submission.jsonl` để user nộp portal (đo trên DN chưa thấy; 9–10/14 DN cần mapping LLM).
3. Viết thêm bộ chuyển đổi định dạng tổng quát cho các họ còn lại (sổ chi tiết bán hàng, DonHang_ChiTiet, quan hệ
   don_hang/chi_tiet_don/thanh_toan, CSDL IDCuaHang) như `pos_invoices.py`.
4. Toolkit cho luật public mới (khách trung thành / sộp / hay đổi ý, giờ cao điểm, nhà cung cấp chính, trung bình theo
   thứ, mẹ & bé, sửa xe, điện thoại): phần lớn đã có hàm (`customer_sales(by=...)`, `peak_hour`, `top_supplier(by="receipts")`,
   `weekday_avg`), cần mẫu câu + kiểm chứng.
5. Fine-tune (E7) trên các cặp (câu hỏi → chương trình toolkit) đã kiểm chứng: ~2,680 cặp trong `src/refs/answer_examples.jsonl`.
