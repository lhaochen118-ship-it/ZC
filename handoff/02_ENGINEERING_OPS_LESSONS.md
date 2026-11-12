# PDF 2 — Engineering & Operations Lessons (DataMate, đến 11/11/2026)

*Phạm vi: kinh nghiệm lập trình, cấu hình môi trường, quản lý Kaggle, tổ chức thử nghiệm, lỗi đã gặp và cách sửa,
bài học thành công / thất bại. Kết quả khoa học ở PDF 1; trạng thái và việc tiếp theo ở PDF 3; cấu hình máy ở
`environment_summary.pdf`.*

## 1. Lập trình

### 1.1 Thiết kế đã chứng minh hiệu quả

- **Ngữ nghĩa nằm trong code đã kiểm chứng** (`src/dm.py`, `src/industry.py`); LLM chỉ viết chương trình rất ngắn.
  `CANON_DOC` mô tả bảng chuẩn / hàm cho prompt.
- **Mỗi sửa lỗi = một quy tắc tổng quát + một test**; không logic theo id DN / câu.
- **Lưu đầu ra từng câu** (chương trình, đáp án, phiếu, ánh xạ ứng viên) → chấm lại / bỏ phiếu lại offline trong vài giây
  (`src/replay.py`, `src/vote.py`, `automation/map_penalties.py`, `src/consensus.py`).
- **Luồng nộp ghi kết quả dần**: `src/merge_sub.py` gộp theo thứ tự ưu tiên, ghi nguyên tử (tmp + rename) mỗi 20 s →
  hết giờ vẫn giữ mọi đáp án đã có.
- **Chế độ smoke** (`ZC_SMOKE=1`, `--fake-responder`, `--max-pending`): cùng đường code chạy trên CPU với mô hình giả.

### 1.2 Lỗi lập trình đã gặp

| Lỗi | Hậu quả | Sửa / bài học |
|---|---|---|
| `return "tpl", code if P else None` (thứ tự ưu tiên toán tử) | trả `("tpl", None)` → lỗi khi chạy | `qparse.parse` loại kết quả có code None; viết `return (…) if P else None` |
| Regex không dấu ("dem" khớp "đếm" / "đem về", "sàn" khớp "sản phẩm", "hoa hồng" bị đọc là "không") | luật bắn nhầm | neo regex, kiểm từ có dấu trên câu gốc |
| Từ đồng nghĩa thay trước bộ chặn | bộ chặn "thu nhập" mất tác dụng | bộ chặn đọc cả câu gốc |
| Tách từ dính quá tay ("soda" → "so da"; "hậu giang" → "châu giang") | sai tên | chỉ tách khi ≥ 2 âm tiết hợp lệ, không tách tên riêng |
| `df[c].dtype == object` bỏ sót cột `string` của pandas 3 | kiểm tra không chạy | `pd.api.types.is_string_dtype` |
| Kiểm tra ném ngoại lệ (dòng tiêu đề lẫn vào ngày) | kiểm tra **im lặng** | đọc mềm `errors="coerce"` |
| Hàm trùng tên trong class | bản sau ghi đè | grep trước khi thêm |
| Test giả định đường ánh xạ LLM khi `fmt:automap` được ưu tiên | test hỏng | `ZC_HIDE_REFS` |
| Sửa file bằng script Python với chuỗi ba dấu nháy lồng nhau | SyntaxError | script riêng hoặc Edit; kiểm `ast.parse` sau sửa |

### 1.3 Mô hình / vLLM

- Thinking không greedy (temperature 0.6, top-k 20); budget forcing (trần suy nghĩ 12288 token).
- FP8 trọng số (`--quant fp8`) chạy tốt; FP8 KV cache và `VLLM_BATCH_INVARIANT=1` không chạy với Qwen3.5 / vLLM 0.32;
  MTP chỉ −6 %.
- Nạp mô hình hai lần trong một luồng (stage 1.5 rồi stage 2, hai tiến trình riêng) không vấn đề bộ nhớ: tiến trình kết
  thúc giải phóng VRAM.
- Thời gian chủ yếu là **sinh token**, không phải ánh xạ: ưu tiên rút ngắn đầu ra và giảm số câu cho mô hình.

## 2. Môi trường container (Claude Code cloud)

- 4 nhân, 15 GB RAM, không GPU / Docker daemon; ~24 GB đĩa trống (hạn mức riêng, `df` gây hiểu lầm).
- **Container có thể khởi động lại giữa phiên** (gặp 3 lần 05–07/11): tiến trình nền mất, file trong scratch và
  `~/.kaggle` thường còn. Sau khởi động lại: kiểm token, gắn lại watcher (`run_kernel.py --attach`).
- **Tác vụ nền có trần 2 giờ** (`timeout` 7200000 ms): `fanout.py` + theo dõi kernel 3 giờ bị dừng → kernel Kaggle vẫn chạy,
  gắn lại bằng `python3 automation/run_kernel.py <work>/<slug> <work>/<slug>_out 480 5 43200 --attach`.
- OOM (exit 137) khi chạy train với 4 worker + bảng sao chép: dùng 2 worker cho train (`RW=2`).
- Không `pkill -f` / `ps | grep` theo mẫu có trong chính lệnh (tự giết shell). Liệt kê PID, xem, rồi `kill <PID>`.
- Chờ điều kiện: vòng `until …; do sleep 20; done` trong lệnh có `timeout`; không `sleep` dài trần.
- PDF: `pip install markdown xhtml2pdf` (mất khi đổi container) rồi `python3 automation/md2pdf.py <in.md> <out.pdf>`.

## 3. Kaggle

### 3.1 Vận hành chuẩn

- `python3 automation/fanout.py <runs/vNN> zc-vNN <jobs> <n_kernel> --accounts <acc> --l4 --ds <acc>/<dataset>
  --ms danbth/qwen3-5-9b/transformers/default/1 --ks nhatlong1103/zc-assets`: tự preflight (cache theo hash code + jobs),
  tạo kernel, đẩy, failover khi hết quota / đủ 2 phiên, kéo kết quả về `<runs>/<slug>_out`.
- **Quota**: `kaggle quota` (CLI 2.2.5) cho số giờ thật (`automation/quota.py`); đo trên cả kernel: phiên 4×L4 tốn ~1.5–2.2 h
  quota / giờ tường (tính dồn đầu phiên; hệ số an toàn 2.5); 30 h / tuần / acc ≈ 12–15 h tường. Làm mới thứ Bảy 00:00 UTC. Phiên
  đã chạy không bị dừng khi vượt 30 h (acc 31–36 h), nhưng không được đẩy phiên mới dựa vào điều đó (quy định user).
- Trọng số từ Kaggle Model (`--ms`), `run_kaggle.py` kiểm sha256 với `src/model_hashes.json`, copy có khoá + `.complete`.
- Dataset private không chia sẻ được giữa acc → mỗi acc một bản (`<acc>/zc-para`, `<acc>/zc-pend`).
- Kernel CPU (`ZC_NO_VLLM=1`) không có `nvidia-smi` → monitor GPU phải bọc try.

### 3.2 Lỗi Kaggle đã gặp

| Lỗi | Cách xử lý |
|---|---|
| Hết quota tuần (cả 3 acc L4 tuần W45, 06/11 03:05) | chờ reset thứ Bảy; ghi lệnh đẩy lại vào `EXPERIMENT_LOG.md` |
| Giới hạn tốc độ API khi tải hàng nghìn file log | log từng câu để trong `/tmp` của kernel; tải bằng `--file-pattern` / `--page-size` |
| Kernel failover sang acc không có dataset private | xoá kernel sai, tạo dataset ở acc đó |
| API 503 đọc thành `error` | `run_kernel.py --attach` |
| Import thư viện thẳng từ `/kaggle/input` (> 900 s) | luôn copy / cài vào môi trường kernel |
| Acc dùng chung dự án khác | chỉ đụng kernel `zc-*` |

## 4. Tổ chức thử nghiệm

- **Cổng trước khi chạy**: tự review diff → `tests/run_all.sh` (ALL PASS) → `automation/preflight.py` (`PREFLIGHT OK`;
  smoke cả đường `run_kaggle.py` + script). Dồn sửa rồi preflight một lần (~10–20 phút).
- **Kiểm tổng quát hoá ngay từ đầu** (bài học lớn nhất): `tests/eval_all.sh` (gốc, đổi tên cột, README trắng, không dấu,
  nhiễu) + paraphrase (`eval_split.py --perturb-q para:`, `stress_flow.py --modes para:` trên Kaggle).
- **Quy tắc held-out**: chỉ phân tích lỗi trên paraphrase k=0 public; giữ thay đổi khi held-out (k=1,2, train) cải thiện;
  luật lấy câu khỏi LLM khi độ chính xác held-out ≥ ~85 %. Ghi cả kết quả âm (`EXPERIMENT_LOG.md`).
- **Phân tích không tốn GPU trước**: đo chính sách mới bằng đầu ra đã có (ví dụ v33: so luật trên câu viết lại với đáp án
  v32 trên cùng câu chờ) → chỉ chạy GPU để xác nhận (v34 khớp ước lượng: 0.602 vs 0.600).
- File nộp: `automation/update_final_submission.py <run_dir>` → `final_submission/` (chỉ thay khi public PA cao hơn) → push ngay.

## 5. Bài học chiến lược

**Thành công**

- Phân tích bất đồng không cần nhãn (luật vs LLM, ánh xạ hạng 1 vs 2, đáp án 0 bất thường) tìm ra lỗi thật.
- Ngân sách thời gian + gộp kết quả liên tục: không lần chạy nào mất kết quả vì hết giờ.
- Đo GPU (`gpu_monitor.csv`, `gpu_summary.json`): bận 84–94 %, 4 GPU kết thúc lệch ≤ 20 % nhờ work stealing.

**Thất bại / cần tránh**

- **Tối ưu trên tập trong mẫu**: viết luật tới 100 % public mà không có thước đo held-out sớm → hệ khớp public, yếu trên
  cách hỏi mới. User yêu cầu: luật không sinh thẳng đáp án, chỉ hỗ trợ Agent (PDF 3 §2.1).
- Tin điểm train / một lần nộp portal để chọn bản (portal nhiễu ±0.03, đáp án portal từng đổi).
- Trả lời user dài và quá lạc quan: user cần câu trả lời ngắn, trung thực về giới hạn (ví dụ "100 % là trong mẫu").

## 6. Bài học kế hoạch 2 (09–11/11)

### 6.1 Hạ tầng thêm mới (đều có test CPU)

| Thành phần | Việc | Ghi chú |
|---|---|---|
| `automation/quota.py` | quota thật mỗi acc, `check(acc, need)`; `launch_many.py` từ chối acc < 6 h không đủ giờ (quy định user) | đã chặn đúng lelynhan 0.37 h khi M4 cần 18 h |
| `make_kernel.py` runner | hạn chót 12 h − 40 phút, `ZC_DEADLINE` cho mọi job, bỏ job tuần tự không kịp (`_skipped_jobs.json`), giết job quá hạn | `tests/test_session_limit.py` (phiên giả 40 s) |
| `training/deadline.py` + `sft*.py`, `dpo.py` | checkpoint (LoRA + optimizer + scheduler + RNG) trước hạn, `--resume` / `--resume-from auto` | thử CPU trên mô hình tí hon: dừng 1/4 → tiếp 2–4 |
| `agent --hard-deadline` | bỏ trả lời câu còn lại đúng lúc hết giờ, **không** chế độ nhanh | `--budget` cũ bật k = 1 khi ước lượng đầu lượt bi quan → đổi kết quả dù còn xa hạn: không dùng cho đánh giá |
| `arms_job.py --base-adapters` | LoRA học chồng lên súp / thành viên đã gộp | M2, M4 |
| `training/dpo.py`, `full_job.py --trainer dpo` | TRL 1.15 DPO + PEFT, FSDP 4 GPU | 1 L4 không đủ cho cặp DPO 9B; `fused_lm_head` của TRL cần stub torch trên CPU (`ZC_TORCH_LOGPROB=1`) |
| `training/data/rft_from_logs.py`, `dpo_from_logs.py` | dữ liệu RFT / DPO từ log; dựng lại prompt đúng như lúc suy luận | eval dùng **rulebook public** (tập cha): dựng bằng rulebook train làm gợi ý khác ở 7/200 câu |
| `src/selector.py`, `tests/train_selector.py`, `tests/pool_gens.py` | bộ chọn học được, phát lại / gộp mẫu nhiều lần chạy không GPU | đặc trưng đếm lệch khi gộp → dùng `FEATS_INV` |
| `automation/set_final_model.py` | đổi mô hình nộp một bước (dataset + `soup.json` + `src/adapter_hashes.json`) | `kaggle datasets version` thay toàn bộ file của dataset |
| `automation/update_final_submission.py` | `final_submission/` = file public PA cao nhất | chạy sau mỗi lần có public |

### 6.2 Lỗi / sự cố kế hoạch 2

| Sự cố | Sửa / bài học |
|---|---|
| Ổ đĩa phiên đầy (0 MB) khi tải output → file job rỗng, output WG thiếu | dọn scratch tái tạo được (test, adapter nhánh cũ), tải lại output (`kaggle kernels output <id> -p <dir>`); theo dõi `df` trước khi tải lớn |
| Container khởi động lại giữa đêm | token / scratch còn; `automation/resume.py <runs>` gắn lại mọi kernel chưa xong, rồi bật lại watcher |
| Tải output bị ngắt (proxy BrokenPipe) | tải lại; kiểm đủ `submission.jsonl` + `result.json` trước khi chấm |
| Acc mới chưa accept luật cuộc thi | push lỗi "accept this competition's rules" → `automation/accounts.json` `no_l4`; nhờ user accept |
| Failover sang acc không đọc được dataset private | `fanout.py`: không failover khi `--ds` là bản riêng của acc khác |
| Dataset chỉ có **một** thư mục gốc (`WG9/adapter/...`, đăng `-r zip`) | Kaggle gỡ lớp thư mục đó → file nằm ở `adapter/...`, `soup_job` báo "adapters not found" (12/11, 4 kernel mất 13 phút) | luôn thêm một thư mục phụ (`info/README.txt`) cạnh thư mục adapter; dataset ≥ 2 thư mục giữ nguyên cấu trúc |
| Tên dataset < 6 ký tự (`zc-wg`) | Kaggle từ chối: dùng ≥ 6 ký tự (`zc-wg-adapters`) |
| `kaggle datasets create` lỗi "slugs and hashlink are all null" (phuongnehe) | lỗi phía Kaggle; chuyển acc khác |
| Chấm điểm `score.py` | thứ tự đối số `score.py <refs> <preds>` (đảo → 0.976 thay vì 0.982) |
| `sleep` dài / poll trạng thái | dùng watcher nền `automation/watch_done.py --max 115` (trần 2 h → bật lại khi `WATCH_TIMEOUT`) |

### 6.3 Tổ chức thử nghiệm nhanh (user 11/11: "nhanh nhất mà không ảnh hưởng nhiều độ chính xác")

- **Một ứng viên / kernel, song song nhiều acc** (2 phiên GPU / acc) thay vì nhiều job tuần tự trong một kernel.
- **Sàng lọc** trên `pub_full,pub_ho400,tr_ho400` (~35–40 phút) → **xác nhận** đủ tập (`pub_full,pub_ho,tr_ho`, ~70 phút) 2 seed
  cho ứng viên dẫn đầu. Tập con nhiễu ±0.6 → chỉ để loại.
- Phát lại offline trước khi tốn GPU (bộ chọn, ngưỡng, kiểm tra mới: `pool_gens.py`, quét log tìm tín hiệu sạch).
- Cùng một lần gộp mô hình đánh giá nhiều cấu hình: `soup_job.py --eval-configs a,b`.
- Chi phí cố định mỗi kernel: cài vLLM ~7 phút, gộp adapter 2–10 phút.

### 6.4 Quét "tín hiệu sạch" cho một kiểm tra mới (mẫu)

```python
# cho mọi logs.jsonl ở Process/runs/*: với mỗi mẫu, nếu rule(code) bắn -> đếm theo (tập, đúng/sai) của đáp án cuối
# nhận rule khi không mẫu đúng nào bị chạm trên mọi tập (dev + held-out, held-out chỉ xem tổng)
```
(Mã đầy đủ: lịch sử phiên ở commit "Precheck len_mask also through list() / .tolist()" — `src/precheck.py::_len_of_mask`.)
