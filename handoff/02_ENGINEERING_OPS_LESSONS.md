# PDF 2 — Engineering & Operations Lessons (DataMate, đến 16/11/2026)

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
| Chuỗi checkpoint qua acc (`full_job.py`, env `ZC_SESSION_S` / `ZC_SESSION_MARGIN`) | phần 1 dừng trước hạn có checkpoint → tải output → đăng dataset (có `info/`) ở acc kế → phần 2 `--ds <acc>/<ckpt> --resume-from auto` | đã chạy thật 12–13/11 (QS10 26M token: nguyenmanh0404 checkpoint-59/119 → hai1108 bước 60–119, `resumed_from` đúng, 1.69k tok/s cả hai phần); phí ~15 phút cho tải mô hình + nạp checkpoint ở phần sau |

### 6.2 Lỗi / sự cố kế hoạch 2

| Sự cố | Sửa / bài học |
|---|---|
| Ổ đĩa phiên đầy (0 MB) khi tải output → file job rỗng, output WG thiếu | dọn scratch tái tạo được (test, adapter nhánh cũ), tải lại output (`kaggle kernels output <id> -p <dir>`); theo dõi `df` trước khi tải lớn |
| Container khởi động lại giữa đêm | token / scratch còn; `automation/resume.py <runs>` gắn lại mọi kernel chưa xong, rồi bật lại watcher |
| Tải output bị ngắt (proxy BrokenPipe) | tải lại; kiểm đủ `submission.jsonl` + `result.json` trước khi chấm |
| Acc mới chưa accept luật cuộc thi | push lỗi "accept this competition's rules" → `automation/accounts.json` `no_l4`; nhờ user accept |
| Failover sang acc không đọc được dataset private | `fanout.py`: không failover khi `--ds` là bản riêng của acc khác |
| Nhiều job song song trong một kernel (mỗi job một GPU) cùng `pip install vllm` | 3 / 4 job `lrl_gen` chết (`OSError ... site-packages`, `No module named vllm`) — 13/11 `hai1108/zc-lrl0a-1` | `run_kaggle.py`: khoá `/tmp/.zc_pip.lock` + dấu `/tmp/.zc_vllm_installed` — job đầu cài, các job sau chờ rồi dùng lại |
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

## 7. Bài học 12–13/11 (hướng thuần, kế hoạch 3)

### 7.1 Huấn luyện dài qua nhiều acc (chuỗi checkpoint)

- `training/full_job.py` + env `ZC_SESSION_S` / `ZC_SESSION_MARGIN`: dừng trước hạn có checkpoint (LoRA + optimizer + scheduler + RNG)
  → **`automation/chain_next.py <runs> <part> <NAME> <next_acc> <next_part> <resume.jobs> --train-data D --remaining-h H`**: chép
  `<NAME>/ckpt/checkpoint-N` + `info/` thành dataset ở acc kế, chờ file live, tính phiên theo quota (2.5 giờ quota / giờ tường), khởi chạy
  phần sau với `--resume-from auto`. Đã chạy: QS10 (2 phần), v3a (3 phần). Phí ~15–20 phút mỗi lần nối (cài + nạp checkpoint).
- Ước lượng token: **0.33 token / ký tự** cho bản ghi prompt full của Qwen3.5 (đo v3a: 16.7M token / 4918 bản ghi / 9834 ký tự); ước
  0.243 (từ WG) sai 40 %. 4×L4 FSDP: 1.72k tok/s (≈ 2.2 phút / bước, batch 64).
- **NCCL all-gather treo ở bước 0** (WR30 phần 1, một rank chậm, chờ mặc định 30 phút rồi huỷ): `sft.py` đặt `ddp_timeout=600`; chạy lại.
  Cùng cài đặt khác vẫn chạy tốt → lỗi ngẫu nhiên; job dài nên đặt trên một acc đủ giờ khi có thể.
- Sau khi dừng vì hạn phiên, bước "reference loss" của `sft.py` hết bộ nhớ trên một rank → các rank khác chờ NCCL 10 phút, job rc 1, mất
  `train_metrics.json` (checkpoint vẫn còn) → bỏ bước này khi `stopped_early`. Tốc độ mỗi bước tăng dần trong phiên (79 → 114 s / bước ở
  v3a phần 3) → khi tính phiên cho phần cuối, nhân thời gian còn lại ~1.3.

### 7.2 Kernel nhiều job song song

- Mỗi dòng job một GPU (`make_kernel.py`); **`pip install vllm` song song làm hỏng site-packages** (3 / 4 job chết) → `run_kaggle.py` khoá
  `/tmp/.zc_pip.lock` + dấu `/tmp/.zc_vllm_installed`.
- Sinh văn bản có think: **trần 2048 token không đủ** (199 / 199 câu quy nạp không kịp in JSON) → 6144 cho việc cần suy luận, tắt think cho
  việc đơn giản (newterm: 10 / 940 → 502 / 940). Danh sách JSON dài bị cắt ở trần → đọc từng object hoàn chỉnh (`lrl_gen.json_objects`);
  `first_json` phải lấy ngoặc ngoài cùng (trước đó lấy nhầm danh sách con).
- Dataset private không dùng chéo acc: mỗi acc có bản `<acc>/zc-para` riêng (`kaggle datasets list --mine -s zc-para`).

### 7.3 Dữ liệu và đo

- Đo nhiều nguồn trong một split: rulebook **theo DN** (`<split>/rulebooks/<bid>.md`, `HybridPipeline.rulebook_for`) — trước đó `tr_ho` chạy
  với rulebook public (thiếu 3 dòng train).
- Sinh dữ liệu: dùng `zlib.crc32`, không `hash()` (ngẫu nhiên theo tiến trình); tăng cường câu **không được sửa từ nằm trong giá trị của
  chương trình** (120 / 3096 câu phiên âm đã sửa nhầm tên hàng).
- Khớp thẻ luật: trigger ngắn (1 từ < 6 ký tự) phải khớp **có dấu** ('hãng' ≠ 'hàng'); bỏ trigger xuất hiện > 2 % câu của chính lần chạy.
- Ghi log mọi mẫu (`logs.jsonl`) cho phép: quét tín hiệu sạch cho luật mới, so lỗi theo phần chương trình (`Process/tools/cmp_pure_p.py`),
  dựng ca quy nạp — đều không cần GPU.

### 7.4 Quy trình

- Bàn giao: `automation/archive_runs.py <scratch runs>` lưu phần chuyển giao được vào `Process/runs` (log / submission nén, metrics, thẻ);
  adapter / checkpoint ở lại Kaggle (`Process/kaggle/kernels.csv`).
- Watcher phải chạy bằng `run_in_background` (lệnh `&` trong shell không báo về); container khởi động lại → `automation/resume.py <runs>`
  rồi watcher mới.

## 8. Bài học 13–14/11 (kế hoạch 4 S–W–V)

### 8.1 Lỗi đã gặp và cách sửa

| Lỗi | Hậu quả | Sửa |
|---|---|---|
| Đường dẫn cấu hình `src/refs/selector/...` không có trong kernel Kaggle (code đóng gói thành `<stage>/refs/...`); bộ nạp **lặng lẽ** quay về bỏ phiếu | V không chạy ở mọi lần đo ~8 giờ (G0…G0e, g1v3apf, g1v3aho, e0hv, 5 kernel đợt 00:05) | thử cả hai bố cục; thiếu file → dừng; test bố cục kernel (`tests/test_selector_flow.py`); log có `conf_sel` |
| W viết mã một dòng nối bằng `;` (một `#` nuốt phần sau), văn xuôi tới trần token, chép khuôn câu lệnh | 12 / 125 mẫu dùng được | sinh có ràng buộc JSON (vLLM structured outputs, schema có `plan` ≤ 700 ký tự), `code` = mảng dòng, bỏ `import toolkit` |
| W trả dict nhiều đại lượng, tỉ lệ thay vì %, lỗi chỉ ở một số tháng | hàm "chạy được" nhưng sai | kiểm trên 12 tháng + toàn kỳ, loại dict tên biến, lỗi > ¼ số kỳ → loại; vòng tự sửa với lỗi thật |
| `predict_flow.n_gpus` đếm mọi GPU bằng `nvidia-smi` và gán GPU tuyệt đối cho shard của W | chạy song song mỗi GPU một biến thể sẽ giẫm GPU nhau | `visible_gpus()` theo `CUDA_VISIBLE_DEVICES` |
| `chain_next.py` lấy độ dài phiên = quota / 2.5 = 11.95 h khi acc đủ 30 h | `launch_many` (trần 11 h) từ chối, WR30 phần 4 không chạy 1 giờ | trần 10.9 h |
| Thư mục run cũ còn `*.attach.log` có `STATUS=DONE` | watcher (ưu tiên attach log) không báo kernel mới cùng tên | đổi tên log / output cũ trước khi chạy lại cùng tên |
| `ref_loss` sau khi dừng vì hạn phiên chỉ chạy ở một số rank (FSDP) | NCCL watchdog, mất adapter v3b lần 1 | chỉ khi `world == 1` và không `stopped_early` |
| Đĩa container đầy (99 %) giữa bộ test | test dừng nửa chừng | xoá checkpoint / dataset staging cũ (`ds_*`, `runs/wr30p*`), cache pip; cache test sạch (chỉ `biz_*.pkl`) |

### 8.2 Tổ chức thử nghiệm

- **Mỗi GPU một biến thể:** `training/soup_job.py --eval-configs a.json,b.json,... --parallel-gpus 4` hợp nhất mô hình một lần rồi chạy các
  cấu hình cùng lúc, mỗi cấu hình một nhóm GPU (cache riêng, chia CPU worker). Khi còn nhiều acc rảnh, nhanh nhất là mỗi biến thể một phiên
  DP 4 trên acc riêng (~35 phút / 1000 câu); chế độ mỗi GPU một biến thể dùng khi thiếu acc / phiên.
- Đo thành phần bằng **so ghép cặp** (cùng câu, cùng S, chỉ khác thành phần) và **phát lại offline** trên log (bộ chọn, ngưỡng) — không cần GPU.
- Trần thời gian cho bước W trong container (`units_max_s`, giết cả nhóm tiến trình kèm engine vLLM) để bài nộp không mất khi W chậm.
- Trước khi đẩy: CPU `tests/run_all.sh` (26 test), smoke `eval_flow` với mô hình giả cho từng cấu hình mới.

### 8.3 Dùng GPU (user 14/11: 2 phiên × 4 L4 mỗi acc, không tiết kiệm acc / GPU)

| Việc | Cách chia | Số đo |
|---|---|---|
| Huấn luyện LoRA (r 32) | FSDP trên cả 4 L4 (`training/full_job.py`), micro-batch 4, batch hiệu dụng 64 bằng gradient accumulation, bf16; nối checkpoint qua acc khi một phiên không đủ (`chain_next.py`) | 1.74k tok/s (v4: 11M token / 50 bước ≈ 1.9 h; v3b 19.2M / 87 bước ≈ 3.2 h); WR30 phần cuối 11.1k tok/s (chuỗi ngắn) |
| Đánh giá một cấu hình | DP 4 (một bản mô hình / GPU, vLLM fp8, `gpu_util` 0.92 → KV gần hết 24 GB / GPU), hàng đợi động theo DN | `pub_full` 1000 câu, tính cả W (≈ 11–13 phút, 4 shard song song): k 8 ≈ 34 phút, k 16 ≈ 41–43 phút |
| Nhiều biến thể | mỗi biến thể một phiên DP 4 trên acc riêng khi còn nhiều acc (nhanh nhất về thời gian tường: lưới 9 biến thể public xong trong ~2 h); `soup_job --parallel-gpus 4` (mỗi GPU một biến thể, hợp nhất mô hình một lần) khi thiếu phiên | lưới 14/11: 9 phiên song song trên 8 acc |
| Quota | 4×L4 tốn ~2 h quota / giờ tường; acc 30 h ≈ 12–15 h tường / tuần; `launch_many.py` kiểm quota và trần 11 h / phiên | 14/11: 29 acc × 30 h |


## 9. Kế hoạch 5 (14/11) — lỗi đã phạm, cách tránh (BẮT BUỘC đọc trước khi đẩy kernel)

| # | Lỗi | Hậu quả | Quy tắc tránh |
|---|---|---|---|
| 1 | Hai lần đẩy (fanout) **cùng acc** liên tiếp; kernel đầu bị Kaggle giữ hàng đợi > 8 phút → tự huỷ + đẩy lại sau khi lần đẩy thứ hai đã thay dataset `<acc>/zc-code` | kernel chết "STALE CODE" (wjl1, wjp1: mất ~1.7 h) | **Mỗi acc chỉ một lần đẩy cho tới khi kernel trước báo RUNNING** (`tail <run>/<slug>.log`); cần 2 phiên cùng lúc → đẩy acc khác |
| 2 | Gửi file `.gz` trong `--train-data` | Kaggle tự giải nén dataset → tên file đổi, job "not found" (zjs, zjb) | Gửi file **không nén**; mọi tra cứu file thử cả tên có / không `.gz` (`judge.find_file`) |
| 3 | Mô hình đã fine-tune viết code (S) được giao việc khác **không ràng buộc định dạng** (W1 bind, W3 không suy nghĩ) | 78 % dòng rulebook lập luận văn xuôi tới trần token, không ra JSON (w1b); 95 % đầu ra phán xử không có `<choice>` | Việc cấu trúc của W → **JSON schema / structured output** của vLLM ngay từ đầu, hoặc dùng adapter đã huấn luyện cho đúng định dạng |
| 4 | Bản ghi huấn luyện dài hơn `--max-len` | bị bỏ qua lặng lẽ (`records ... longer than max_len skipped`) | Đo độ dài bản ghi (ký tự / 2.5 ≈ token) **trước** khi đẩy; đặt `--max-len` ≥ p99 |
| 5 | LoRA động của vLLM trên mô hình fp8 với adapter thô (chưa chuyển tên module) | nạp không lỗi nhưng **không có tác dụng** (ejb1lora: 0 % parsed); sau `merge.py vllm`: đúng như gộp (ejb1lora2: 383 / 450 parsed, PA 0.951 vs gộp 389 / 450, 0.950) | Chuyển bằng `training/merge.py vllm` trước, hoặc gộp adapter vào bản sao mô hình (mặc định `w_mode: merge`); luôn kiểm tra tỉ lệ parse sau lần chạy đầu |
| 6 | `preflight.py` không chạy được smoke của script trong `training/` | báo FAIL dù test CPU PASS (mất ~40 phút chờ) | Với job `training/*.py`: chạy `tests/run_all.sh` + smoke trực tiếp `soup_job.py ... --fake-responder --subset` trên CPU, rồi `--no-preflight` |
| 7 | Link dòng rulebook zero-shot gắn dòng **chung chung** kèm hàm gợi ý vào nhiều câu | S bị kéo sai (0.938 → 0.921) | Lọc dòng nối với > 5 % câu của split (`link.py --max-df 0.05`); dùng linker đã huấn luyện (WL1) |
| 8 | `watch_done.py` thoát sau mỗi kernel xong; Monitor hết hạn sau 30 phút | dễ bỏ sót sự kiện | Chạy watcher trong vòng lặp nền (`watch_loop.sh` → `watch2.out`) + Monitor `tail -F` persistent; container khởi động lại → `automation/resume.py` |
| 9 | Đánh giá trên **một** lần chạy S | nhiễu ±0.01 giữa lần chạy | So ghép cặp trên cùng log; xác nhận trên ≥ 2 log S (pva3k16, g1v3apf2) |

### 9.2 Kỹ thuật hiệu quả của kế hoạch 5 (dùng lại)

- **Phán xử offline trên log đã lưu**: W3 chỉ đọc mẫu S trong `logs.jsonl` → mọi biến thể / adapter / chính sách đo lại được mà không chạy lại S
  (mỗi lần ~10–15 phút / 4 biến thể song song, mỗi GPU một biến thể); chính sách chọn hoàn toàn trên CPU (`tests/replay_judge.py`, `ens2.py`).
- **Chỉ phán câu có mẫu bất đồng** (`--only-split 1`): ~10 % câu → nhanh ~10 lần và không làm hỏng câu đồng thuận.
- **Không chờ huấn luyện xong mới chạy container**: chạy đường container với adapter tốt nhất hiện có, rồi phán xử lại log của nó bằng adapter mới.
- **Adapter W học chồng trên S đã gộp** (`--base-adapters V3A --base-weights 0.70`): gộp `V3A × 0.70 + W × 1.0` = đúng mô hình đã huấn luyện;
  trong container gộp từng pha (`w_model`, `w_mode: merge`, ~1.5 phút / adapter) hoặc LoRA động sau `merge.py vllm` (`w_mode: lora`).
- **Mỗi lần chạy container lưu `solver_logs.jsonl`, `links.jsonl`, `cards_unit.jsonl`, phiếu phán xử** (`soup_job --runner predict`).
- **Dataset adapter riêng cho acc chạy container** (output kernel không mount chéo acc): `Process/tools/publish.sh <acc> <dir> <slug>`.
- **Huấn luyện**: FSDP 4×L4 1.7–2.1k tok/s, đỉnh 16 GB / GPU với mb 4, max-len 4608–6144; 7–10M token ≈ 2–3 h.


## 10. Kế hoạch v2 (14/11 17:00 → 15/11 04:00, chế độ H) — lỗi, cách sửa, kỹ thuật dùng lại

### 10.1 Lỗi đã phạm và quy tắc tránh

| # | Lỗi | Hậu quả | Quy tắc tránh |
|---|---|---|---|
| 1 | Chế độ H mới ẩn hàm trong prompt S; thẻ W1 public cũ, link, prompt phán xử vẫn nhắc tên hàm `industry.py` | 14 / 125 thẻ rò gợi ý hàm ngành | Một nguồn sự thật: `lrl.hidden_names()` đọc `ZC_HIDE_API`, `lrl.strip_hidden()` lọc ở **mọi** chỗ đưa văn bản vào prompt (thẻ, link, phán xử); `tests/test_mode_h.py` |
| 2 | Đẩy held-out thiếu dataset `<acc>/zc-para` | chỉ 206 / 1,122 câu được chấm, không lỗi | `eval_flow` **dừng** khi > 10 % id của tập không có văn bản; mọi thành phần tùy chọn thiếu → dừng hẳn, không chạy tiếp lặng lẽ |
| 3 | 17 watcher cùng poll Kaggle mỗi 30 s từ một IP | Kaggle API 429 hàng loạt, `run_kernel` hiểu nhầm trạng thái | `run_kernel.py` poll `ZC_POLL_S` (mặc định 150 s) + jitter, backoff 240–360 s khi 429; khi API hồi lại: `automation/reattach.py` |
| 4 | Qwen gốc sinh dữ liệu ở chế độ suy nghĩ | lặp vòng quá 7–8k token, 7 / 243 dòng dùng được (TG1) | Việc sinh dữ liệu: **không suy nghĩ + JSON schema** (vLLM structured output), 8 mẫu, đồng thuận ≥ 3; khi cần suy nghĩ: **budget forcing** (`lrl_gen.force_json`: suy nghĩ tới trần → đóng `</think>` → giải mã JSON có ràng buộc) |
| 5 | Hai engine vLLM trong một tiến trình (sinh + kiểm ngược) | engine thứ hai hết VRAM | Một engine dùng cho cả hai bước (`training/contrast_job.py`) |
| 6 | JSON có ràng buộc trả chuỗi có thực thể HTML (`&quot;`) | chương trình sinh ra không chạy | `html.unescape` trên mọi trường văn bản / mã đọc từ JSON có ràng buộc |
| 7 | Độ dài bản ghi ước theo ký tự | WT0B mất 305 / 543 bản ghi (max-len 4608 quá ngắn) | **Đo bằng tokenizer Qwen** (`apply_chat_template` của adapter có sẵn, `ad3/V3A/adapter`) trước khi chọn `--max-len` |
| 8 | FSDP mb 6 / mb 4 × 5120 token | OOM (phân mảnh) | mb 2 × tích luỹ 8 + `--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; đỉnh 15 GB / GPU, 1.73–1.86k tok/s (không chậm hơn mb 4 đáng kể) |
| 9 | Lệnh nền bị container khởi động lại giết | mất watcher / resume | Sau mỗi lần khởi động lại: `watch_loop.sh` + `automation/resume.py`; kiểm `ps` |
| 10 | So điểm giữa *file container* và *file phán xử* lẫn lộn | kết luận lệch 0.003–0.013 | Bảng điểm luôn ghi cả hai (`Process/scores_v2.csv` cột `config_or_dir`); `<config>/submission.jsonl` = container |
| 11 | `merge_sub.read` bỏ dòng abstain → file container lấy đáp án S khi phán xử đã quyết định bỏ | container thấp hơn phán xử 0.003–0.013 ở mọi lần chạy | Gộp cuối phải coi file phán xử là quyết định cho mọi câu nó phủ (kể cả abstain) — **chưa sửa**, việc đầu tiên PDF 3 §5 |
| 12 | `tests/run_all.sh` chạy với `timeout 900` | bị cắt ở `test_plausibility` (exit 143), tưởng lỗi | Bộ test CPU mất ~30–40 phút khi máy bận: `timeout 2700`, ghi ra file log, không pipe qua `tail` |
| 14 | Chèn câu `if` mới vào **giữa** chuỗi `if / elif / else` của `lrl_gen.main` (I4 / I5) | `--mode unit` rơi vào `else: sys.exit('unknown mode unit')` khi cờ mới = 0 → 7 kernel (04:43–05:15) chạy không có hàm cửa hàng; smoke có báo `unknown mode unit` nhưng bị đọc nhầm là lỗi mô hình giả | Thêm bước xử lý **sau** chuỗi phân nhánh; smoke phải chạy cả cấu hình mặc định (cờ mới = 0) và mọi dòng lỗi trong smoke phải truy tới gốc trước khi đẩy; huỷ kernel hỏng bằng `kaggle kernels delete -y` và đẩy lại tên mới |
| 15 | Pha song song (shard / GPU) chỉ ghi 3 dòng cuối log mỗi shard và chạy tiếp khi một shard hỏng | shard 2 / 4 bộ viết hàm thoát rc 1 ở 4 lần chạy I4 + I5 → mất hàm 1–2 cửa hàng, điểm 0.840 bị đọc như kết quả của ý tưởng; 3 dòng cuối là dòng tắt vLLM nên mất traceback | Shard rc ≠ 0 → ghi traceback (đã sửa trong `prepare_units`); khi so biến thể, kiểm `units: rc=[…]` và số hàm trước khi kết luận; đánh dấu lần chạy có sự cố là "không kết luận được" |
| 16 | Watcher nền (`watch_loop.sh`) chết khi worker của phiên khởi động lại; tải output bị `Connection reset` giữa chừng mà vẫn ghi DONE | 3 h không thu kết quả (08:00–11:00); phquvx thiếu `submission.jsonl` → grid bỏ qua | Sau mỗi lần khởi động lại: `ps` kiểm watcher, chạy lại; với mỗi DONE kiểm có `submission.jsonl`, thiếu thì `kaggle kernels output` lại (thử 3 lần) |
| 13 | Tối ưu theo số đo trên phân phối nguồn (dòng held-out DN train) | WT-b +0.08 offline nhưng hại trên public | Mọi thành phần mới phải có ít nhất một phép đo **đầu-cuối trên phân phối đích** (public chế độ H) trước khi xây tiếp lên nó |

### 10.2 Kỹ thuật hiệu quả (dùng lại)

- **Dùng lại hàm / link của một lần chạy** (`units_file`, `links_file` trong config, tìm bằng `predict_flow.find_input`): so biến thể S trên
  **cùng** bộ hàm (ghép cặp sạch) và bớt ~15–20 phút / lần chạy. Bản nộp vẫn sinh lại. Ví dụ `configs/flow_h_reuse.json` + `configs/reuse/`.
- **Tập đánh giá tự mang câu** (`tests/splits/<set>.txt` + `<set>.questions.jsonl`, id có `~`): chạy S trên câu do mô hình sinh (kiểm ngược
  dữ liệu UQ: `uqchk_a/b`) bằng đúng đường `eval_flow` — không cần code mới.
- **Phân tích log trên CPU trước khi đẩy GPU**: tỉ lệ đúng theo biến thể / vòng (`var_acc2.py`), theo việc gọi hàm (`unit_use*.py`,
  `unit_final.py`), gộp log (`pool.py`) — mỗi phân tích ~1 phút, đã loại 3 hướng (k lớn, gộp lần chạy, WT) trước khi tốn thêm GPU.
- **Mỗi biến thể public = một phiên DP 4** (mỗi GPU một vLLM fp8, `gpu_util` 0.92, hàng đợi động theo DN): ~2.0–2.4 h tường, ~4.5–5.5 h quota;
  huấn luyện = một phiên FSDP 4 GPU; hai biến thể huấn luyện song song trên 2 phiên của cùng acc.
- **Đẩy nối tiếp có điều kiện** bằng script nền: `when_ready.sh` (chờ dataset ready rồi `launch_many`), `after_running.sh` (chờ kernel trước
  RUNNING để không đẩy trùng acc) — `Process/tools/v2/`.
- **Dataset adapter Docker có sha256**: `src/adapter_hashes_w.json` + `docker/fetch_assets.sh`; tải lại và kiểm khớp trước khi coi là xong.

### 10.3 Thời gian / quota đo được (4×L4)

| Việc | Giờ tường | Ghi chú |
|---|---|---|
| Đường container public 1000 câu, chế độ H, k16 | 1.8–2.2 h | units ~12–20 phút, link ~8, S ~47 (k24: ~82), phán xử ~28 |
| Sinh UQ 907 hàm × 2 câu (4 shard, không suy nghĩ) | 19 phút + cài ~5 | |
| Huấn luyện FSDP 13M token (mb 2, max-len 5120) | ~2.3 h | quota ~5.5 h |
| Quota thực | 2.0–2.4 h quota / giờ tường | kế hoạch dùng hệ số 2.5 |

## 11. Kế hoạch P5 zero-shot + ghép (15/11 08:00 → 16/11 02:30) — lỗi, cách sửa, quy tắc tránh

*Rà soát code 16/11 02:00 (37 tệp, ~1.900 dòng thay đổi của P5) + đối chiếu log mọi lần chạy (`units: rc`, số hàm, link, phán xử).*

### 11.1 Lỗi ảnh hưởng điểm (đã sửa; đã chạy lại hoặc đã có lần chạy thay thế)

| # | Lỗi | Hậu quả | Cách sửa / quy tắc tránh |
|---|---|---|---|
| 1 | Template Qwen3.5 **tự suy nghĩ khi không truyền `enable_thinking`**; `agent.render` chỉ truyền khi đường dẫn mô hình chứa `qwen3` → các lần 0.952 (thư mục merged, không có `qwen3`) suy nghĩ, còn bản ghép dùng tên khác thì không | C0 0.921 thay vì 0.952; mọi lần chạy MoE (`Qwen3.6-...` chứa `qwen3`) tắt suy nghĩ | Công tắc `think_kw` (legacy / explicit / default, env `ZC_THINK_KW`); chạy lại C0L 0.942, C1L 0.943, C2L, E1L, E2L. **Quy tắc: mọi config ghi rõ `think_kw`; hành vi không được phụ thuộc tên thư mục** |
| 2 | `lrl_gen` (W1 bind / alias / units) dùng mặc định vLLM (256 seq, util 0.9) → engine không khởi động với MoE 15.5 GB trên L4 | **mọi lần chạy E88 trước 15/11 19:20** (P02, P02s1, P03, P02x, P02xk, P02T, C3, C3T, C4, E1, E1L, E3, E5, E6) chạy **0 hàm, không thẻ bind**; log chỉ ghi `units: rc=[1,1,1,1]` | `max_num_seqs` 128, util 0.92, `max_batched_tokens` 16384. Chạy lại có ghép cặp: E3→E3FT z +1.85, C3T→C3FT z +0.57, E1L→E7 z +3.54, nhưng **P02T→P02xFT z −2.72** (hàm do MoE chưa tinh chỉnh viết làm hại). **Quy tắc: bảng điểm phải có cột `units rc / số hàm`; một bản sửa khôi phục một pha cũng là một biến thể, phải đo ghép cặp, không mặc định là tốt hơn** |
| 3 | Mô hình W (link / phán xử) đọc thẳng từ mount `/kaggle/input` bởi 4 tiến trình | nạp 27 phút, pha link bị giết ở trần → E5 = 0.0 | `predict_flow.resolve_model`: chép một lần sang `/tmp/models/<tên>` (khoá `fcntl`, dấu `.complete`) |
| 4 | Trần phán xử 1800 s cắt phán xử của mô hình zero-shot (chậm hơn bản tinh chỉnh) | S01 bị trần: thiếu +0.023 | `judge_max_s` 5400 cho zero-shot; luôn kiểm dòng `judge: killed at the cap` |

### 11.2 Lỗi vận hành / hạ tầng

| # | Lỗi | Hậu quả | Quy tắc tránh |
|---|---|---|---|
| 5 | `run_kernel` tải toàn bộ output (18 GB trọng số stage) | đầy đĩa container | `ZC_OUT_PATTERN` chỉ tải kết quả; trọng số ở lại Kaggle (mount bằng `--ks`) |
| 6 | Output kernel (`--ks`) là **riêng tư theo acc**; failover của `fanout` chuyển E1/E2 sang acc khác | kernel chạy không có mô hình | `fanout` chặn failover khi `--ks` trỏ output riêng tư (danh sách `PUBLIC_KS`) |
| 7 | Hai `fanout` liên tiếp trên cùng acc | STALE CODE (dataset code bị ghi đè giữa chừng) | Đẩy tuần tự từng acc; chờ dataset ready |
| 8 | Probe H01 lấy logits toàn từ vựng | OOM | `logits_to_keep=1`, lô theo ngân sách token |
| 9 | `--judge-args --tau,...` (giá trị bắt đầu bằng `--`) | argparse báo `expected one argument` (bắt được trước khi đẩy, 16/11) | Viết `--judge-args=--tau,...` |
| 10 | Acc có quota nhưng không dùng được: `mxuanquy` chưa accept rules `gemma-4-developer-agent` (không có L4), `yeshenhi` chỉ T4 (FP8 MoE 15 GB không vừa) | E4m bỏ trắng toàn bộ | Probe acc 1 dòng trước khi giao việc GPU; `accounts.json` → `no_l4` |
| 11 | Quota thấp: `launch_many` từ chối E2L, E4FT (cần 6.0 / 8.8 h) | 2 biến thể không chạy, log ghi `not launched` nhưng dòng cuối vẫn `all launched` | Sau mỗi lần đẩy grep `not launched`; mô hình stage chỉ có ở acc đã stage → stage sớm trên ≥ 2 acc nhiều quota |
| 12 | Worker phiên khởi động lại (13:20, 17:15–19:05, 01:57) | watcher / script nối tiếp chết | `ps` sau mỗi lần khởi động lại; `resume.py`, `p5_watch_loop.sh` |
| 13 | `tests/run_all.sh` với timeout 900 s (lặp lại lỗi §10 #12) | bị cắt | timeout ≥ 2700 s, chạy nền, ghi file |
| 14 | I02 (grammar ràng buộc theo hàm cửa hàng) | 337 phút, CPU tạo mask là nút cổ chai, 0.675 | Đo tốc độ 1 DN trước khi chạy toàn tập |
| 15 | N6-A: SQL trả về `None` được tính như đáp án; `CertificateResult` có `.valid`, không phải `.ok` | chấm sai / ngoại lệ | Lọc đáp án `None`; test đơn vị cho mọi API mới trước khi chấm |

### 11.3 Lỗi tiềm ẩn phát hiện khi rà soát 16/11 (chưa ảnh hưởng điểm)

- `training/merge_fp8.py` bỏ qua `use_rslora` (hệ số α/√r) và không từ chối DoRA / `modules_to_save` → đã thêm; mọi adapter P5 (MS1, MS2, ML1, MJ1, MJ2) là LoRA thường nên kết quả không đổi.
- `new7_score.py` dòng 94: biến `sq` gán rồi bị ghi đè (mã chết, không ảnh hưởng).
- Vote của `new7_score` (θ = 0.5 trên mọi mẫu) khác vote của container → điểm N7 / N3 / N6 chỉ so ghép cặp với chính mốc của nó, không so với điểm container.

### 11.4 Kết quả âm cần nhớ (không lặp lại)

- REAP-64 cộng đồng: 0.429 (sụp); tỉa theo saliency trên prompt của chính framework (E88 / E92) mới dùng được.
- 4B làm phán xử (V01 / V02 / C5): luôn kém 9B.
- Gộp ứng viên nhiều lần chạy rồi **bỏ phiếu** (cùng mô hình 2 seed, hoặc 9B + MoE): ≤ +0.005, thường âm; oracle tăng (946 → 959) nhưng bộ chọn là nút cổ chai.
- N3-A, N6-A, N7-A, N4-A: không cải thiện (z ≤ 0.7).

### 11.5 Nguyên tắc thí nghiệm "chỉ thay đúng thành phần" (user 16/11 05:30, BẮT BUỘC)

- Fine-tune: so `flow_p5_final` (0.952, 9B + V3A × 0.7 / WL1 / WJR1) với **cùng config** chỉ đổi mô hình lõi + LoRA tương ứng; chế độ H: so `flow_p5_final_h` (0.863). Zero-shot: so cùng framework dùng Qwen3.5-9B gốc với mô hình mới gốc (B00-S ↔ P02), không dùng 0.952 làm mốc zero-shot.
- Config thí nghiệm phải **sinh từ config baseline bằng code** (copy + update đúng các khoá thay đổi) và in ra danh sách khoá khác baseline trước khi đẩy; mọi điều chỉnh tương thích (định dạng LoRA, template / suy nghĩ, loader, TP / KV để vừa GPU) ghi trong `_doc`.
- Lỗi đã phạm (15/11): E1–E8 dựng trên framework zero-shot nghiêm (bộ thẻ luật khác, `lrl_regen`, trần phán xử 5400 s) → không quy được hiệu quả cho mô hình; P02T / C3T gộp 3–4 thay đổi một lần. Nhiều thay đổi cùng lúc ⇒ thêm lượt ablation riêng từng thay đổi.
- Mã mới gọi mô hình merged phải đặt `think_kw` tường minh (T02 lượt 1 hỏng vì template tự suy nghĩ).
- 16/11 05:27: 2 kernel gộp LoRA MoE (CPU) chạy qua `fanout.py` **không đặt `ZC_OUT_PATTERN`** → tải về 2 × 15 GB trọng số, đĩa container đầy, `launch_many` của e7rw7 hỏng khi đóng gói code (`shutil.Error ... No space left`). Sửa: xoá trọng số đã tải, attach lại các kernel gộp còn lại với `ZC_OUT_PATTERN='.*\.(log|json)$'`, đẩy lại e7rw7. **Quy tắc: mọi kernel sinh trọng số (stage / merge / train) phải đẩy với `ZC_OUT_PATTERN`; sau mỗi đợt kiểm `df -h`.**
- 16/11 fts3: LoRA trên tham số expert hợp nhất (PEFT `target_parameters` → `ParamWrapper`) **bắt buộc `lora_dropout = 0`**; test CPU dùng mặc định nên lọt, mất 18 phút GPU + 1 lần đẩy. Quy tắc: test CPU phải dựng `LoraConfig` **đúng như mã huấn luyện** (cùng `target_modules`, dropout, r, alpha), không dùng cấu hình rút gọn.
