# 03 — Handoff (DataMate)

## 0. Cập nhật khung chat 2 (2026-10-30 17:00 → 23:00 UTC, branch `claude/peaceful-carson-0dny90`)

**Kết quả chính (train, có đáp án; giao thức dev biz_101–120 / holdout biz_121–130, chi tiết `EXPERIMENT_LOG.md` entry 3–4):**

| Cấu hình | PA |
|---|---|
| Path B thuần (code khung chat 1, smoke 30 câu) | −0.40 |
| + `build_sales` helper + compile rút gọn (`--helpers --compile --compile-scope sales`) | dev +0.19 / holdout +0.10 (2251 câu, τ 0.75) |
| + chuẩn hoá câu hỏi, khách lẻ, bảng quy đổi, tự tìm header, `valid_*`, 4 vòng sửa compile | dev +0.36 (474 câu) |
| **+ few-shot truy hồi (`--shots`, chương trình đúng của DN khác) + entity linking (`--link`)** | **dev +0.58…+0.59; holdout +0.39 (708 câu, 6 DN)** |

**Cấu hình tốt nhất hiện tại (dùng cho public test `p1`):**

```bash
agent.py <split> <out> --model Qwen/Qwen3-14B-AWQ --dp <n_gpu> --max-len 24576 --chunk 2 --helpers --norm-q --link \
  --compile --compile-scope sales --k 4 --kc 4 --shots shots/train_shots.jsonl --n-shots 2      # τ = 0.5 (mặc định)
```

**Việc tiếp theo (thứ tự):**
1. `--think` (Qwen3 thinking): trên T4 độ đúng từng mẫu 77/130 vs 52/130 không think → đo trên L4 với cấu hình trên (E10 đang chạy
   nhỏ trên T4) và đo thời gian; nếu tốt, ước lượng 6 giờ trên 2×5090.
2. Mở rộng thư viện shots: chạy cấu hình tốt nhất trên toàn bộ train (E9 đang chạy 11 DN còn lại) → `build_shots.py` gộp mọi log.
3. Lỗi còn lại (e7/e8): xếp hạng theo `qty_base` sai khi quy đổi đơn vị, "tuần rồi" / ca sáng–chiều, giá nhập bình quân
   (purchases không có trong C → nhiều unanswerable sai), đếm "mặt hàng ế".
4. Chuyển sang Qwen3-14B-FP8 cho RTX 5090 (đã có dataset `nhatlong1103/zc-m-qwen3-14b-fp8`) và đo tất định (|ΔPA| ≤ 0.03).
5. Docker: cập nhật `docker/predict.sh` với cấu hình trên (thêm `src/shots/`, `semantic.py`, `vnnorm.py`).

**Quota (30/10 23:00 UTC):** hết tuần: thnliu, phuongnehe, linhmai24, synchan5704, yuan1103, mucnokieu, nguynduylm, yeshenhi,
huyenco, loinguyentran; còn: lelynhan (L4, đang chạy p1 + e9), mxuanquy / anderson1202 (T4). Cần user ngắt tay
`mxuanquy/zc-e1b-1`, `anderson1202/zc-e1c-1` (code cũ, vô ích, công cụ xoá bị chặn).

---

## (Bản bàn giao của khung chat 1)

*Bàn giao 2026-10-30 16:45 UTC. Repo `lhaochen118-ship-it/ZC`, **branch `datamate`** (là bản sao của `hmm-navigation-2`
trong `hmm-develop`, giữ lịch sử; nội dung HMM đã bỏ, khôi phục được bằng `git checkout de4360e -- <path>`). Không tạo PR.*

## 1. Bối cảnh

- **Cuộc thi:** Zalo AI Challenge 2026, track DataMate (đề đầy đủ: `docs/00_TASK.md`).
- **Mục tiêu:** tối đa Penalized Accuracy trên private test, chạy offline ≤ 6 giờ trên 2 × RTX 5090, ≤ 15 B tham số, tất định (|ΔPA| ≤ 0.03).
- **Hạn:**
  - nộp Docker **16/11 23:59 (GMT+7)**;
  - private test 20–23/11.
- **Public LB:** tối đa 5 lần nộp/ngày, do user nộp tay trên portal.

### User chốt

- Tự động hoá tối đa, không hỏi lại quyết định nghiên cứu thông thường.
- **Chỉ tạo file nộp thử khi kết quả tốt.**
- Bộ làm sạch phải tổng quát, không vá cứng theo case của dataset.
- Dùng mọi acc Kaggle; mỗi acc dùng đủ 2 phiên GPU; L4 trước, T4 sau.
- Kẹt hàng đợi 5 phút thì huỷ và đẩy lại (≤ 4 lần).
- Định kỳ dọn tiến trình treo hoặc thừa.
- Lưu kết quả dùng lại được thành dataset/output public của `nhatlong1103`.
- Theo dõi bằng watcher nền thay vì tự kiểm tra.
- Tìm thông tin chỉ đọc văn bản/paper, không đọc code repo của người khác.

## 2. Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Đề, dữ liệu, bộ chấm | Xong. Dữ liệu: Kaggle `nhatlong1103/zc-2026` (public) → `data/raw/` |
| Bộ làm sạch | Xong, đã kiểm chứng (audit 0 cột lỗi, 12/12 biến thể tổng hợp, chương trình viết tay khớp tuyệt đối) |
| Agent v1 | Compile + 2 nhánh + sửa lỗi + tự kiểm tra + vote + ngân sách thời gian + DP. Test không GPU: đạt |
| Hạ tầng Kaggle | fanout / make_kernel / run_kernel / watch_done; assets offline public; L4 đã xác nhận |
| Đo độ chính xác | **Chưa có PA nào.** 4 kernel còn chạy (`handoff/RUNNING_KERNELS.md`) |
| Fine-tune | `harvest.py`, `train_sft.py` viết xong, chưa chạy |
| Docker | File xong (`docker/`, hướng dẫn `docker/SETUP_DOCKER.md`); user sẽ build trên máy mình; chưa có Docker guideline của BTC |

## 3. Việc tiếp theo (theo thứ tự)

### Bước 0 — Khởi động (khoảng 10 phút)

1. Xin user token của các acc: `nguyenhoa99`, `mxuanquy`, `anderson1202`, `nhatlong1103` (và `yuan1103` khi quota reset).
   - Lưu vào `~/.kaggle/tok_<acc>` (chmod 600) và kiểm tra bằng `kaggle competitions list`.
2. Cài gói:
   ```bash
   pip install kaggle markdown xhtml2pdf tokenizers python-calamine
   ```
3. Tải dữ liệu:
   ```bash
   kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip
   ```
4. Chạy test không GPU, cả hai phải PASS:
   ```bash
   python3 tests/test_ingest_robust.py
   ```
   ```bash
   python3 tests/test_agent_local.py data/raw/train/train <scratch>
   ```
5. Kéo kết quả các kernel còn chạy, rồi phân tích từng run train bằng `src/analyze.py` (xem `RUNNING_KERNELS.md`):
   ```bash
   python3 automation/pull_kernels.py <scratch>/pulled
   ```

### Bước 1 — Khắc phục hết lỗi đã mắc (user yêu cầu)

Từ `01_SCIENTIFIC_FINDINGS.md` §3 và `02_ENGINEERING_OPS_NOTES.md` §5:

1. **Lỗi chương trình phổ biến:**
   - Thêm helper ngày an toàn vào namespace: cộng/trừ tháng có kẹp ngày cuối tháng, cùng kỳ tháng trước theo rulebook, khoảng tuần. Hiện ~165 lỗi `day is out of range`.
   - Chặn đọc file: thay `open` / `pd.read_*` trong sandbox bằng hàm báo "bảng đã có trong T". Hiện 52 lỗi.
   - Nhắc dùng `.astype(str)` trước `.str` (101 lỗi).
   - Kiểm tên cột bằng `match_names` (33 lỗi).
   - Nếu chương trình không gán `ANSWER` (51 lỗi), gợi ý trong vòng sửa.
2. **Compile chỉ 5/40 thành công:**
   - So A/B trên train: có compile so với không compile (`--paths BBBB`).
   - Nếu compile không giúp thì bỏ, hoặc rút gọn schema chỉ còn `orders` + `lines`.
3. **Độ đồng thuận thấp** (46 % câu chỉ 2/4 phiếu):
   - Chọn τ trên DN dev bằng `analyze.py`.
   - Thử "leo thang": chỉ câu đồng thuận ≤ 0.5 mới sinh thêm mẫu hoặc bật `--think`.
4. **Thông lượng:** kiểm tra bản sửa sinh theo nhóm prefix và chép model cục bộ. Mục tiêu ≥ 1,000 token/s mỗi L4; in tok/s từ log shard.
5. **Mọi cấu hình mới:** chạy thử nhỏ trên đúng loại GPU trước (`--subset 8`, 1 phiên), rồi mới chạy full.

### Bước 2 — Chạy lại trên GPU đã cấp

| Phiên | Việc |
|---|---|
| `nguyenhoa99` (2 × 4 L4) | toàn bộ 3,000 câu train, chia 2 kernel theo DN (dev / holdout). Cho PA theo type, chọn τ, và dữ liệu cho `harvest.py` |
| `mxuanquy`, `anderson1202` (T4, `--tp 2`) | các biến thể A/B nhỏ (300 câu) |
| Sau đó | public test chỉ khi PA holdout tốt; user quyết định nộp |

### Bước 3 — Sau đó

- **E7:** `harvest.py` → `train_sft.py` trên L4 (bf16 gắn bằng `--ms qwen-lm/qwen-3/transformers/14b/1`, wheel `bitsandbytes` từ `nhatlong1103/zc-train-wheels`) → `agent --lora`.
- So Qwen3-14B-FP8 với AWQ trên L4 (bản nộp dự kiến dùng FP8 trên RTX 5090).
- Ước lượng thời gian private test (cỡ ~1000–2000 câu) trên 2 × 5090.
- Docker: chờ guideline của BTC; nhận kết quả chạy thử từ user (`docker/SETUP_DOCKER.md`).
