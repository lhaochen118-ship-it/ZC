# CLAUDE.md — bộ nhớ dự án (Claude Code tự đọc file này mỗi phiên)

Zalo AI Challenge 2026, track DataMate. Branch làm việc **`claude/datamate-v3-0dny90`** (repo `lhaochen118-ship-it/ZC`).
**Khung chat mới: đọc `READING_GUIDE.md` trước** (trạng thái sống `handoff/STATUS.md`; PDF 1 / 2 trong `handoff/`; log, dữ liệu huấn luyện,
sổ kernel ở `Process/` — `Process/README.md`). **Trạng thái 14/11 15:30: kế hoạch 5 (`docs/18_PLAN5_TWO_LORA.md`) xong, ĐANG DỪNG chờ kế hoạch mới (docx) của user. Cấu hình cuối `configs/flow_p5_final.json` (S v3a × 0.70 + W1 bind + W2 linker WL1 + W3 phán xử WJR1; public offline TB 0.952, container WJB1 0.943); Docker cho cấu hình này chưa làm (hạn 16/11 04:00 UTC). Lỗi đã phạm + quy tắc tránh: PDF 2 §9.**Lịch sử quyết định (đã thay thế; chi tiết trong PDF 1 / 2 và lịch sử git):** 08/11 hybrid = mô hình sinh code + luật tổng quát hỗ trợ
(cấm hard code đáp án) và kế hoạch 1 (WP0–WP12, chạy tự động); 09/11 mục tiêu 100 % public bằng cải tiến tổng quát, kế hoạch 2 (hướng R
luật tổng quát / M mô hình); 12/11 chốt P (public 0.994) + Docker; 13/11 kế hoạch 3 (luật do mô hình học). Vẫn áp dụng: **cấm hard code
đáp án** (luật / chương trình sinh thẳng đáp án không qua mô hình); phân tích lỗi chỉ trên public gốc / train, không xem từng câu held-out;
không train trên public / private test.

**`final_submission/submission.jsonl` (user 10/11):** file nộp public có PA cao nhất đến nay (`python3 automation/update_final_submission.py
<run_dir>` chỉ thay khi PA cao hơn, ghi `score.json`); file public của hệ nộp mới (S–W) ở `final_submission/swv/`.

**Quyết định user 13/11 tối (bắt buộc):** (1) **ngừng phát triển luật viết tay** (hints / qparse / qnorm / cờ, precheck viết tay
của P) — khi vận hành thật có nhiều tình huống mới; luật nếu có **phải do mô hình tự viết**; (2) **bản nộp cuối là hệ mới không luật viết
tay** (S–W–V), kể cả khi điểm thấp hơn P; (3) WR30 chạy xong chỉ làm dự phòng. Kế hoạch 4 (xong 14/11): **`docs/17_PLAN4_SWV.md`** (S mô hình
giải, W mô hình viết luật thực thi được lúc suy luận, V bộ chọn học được; gói V1, W1, S1–S3, W2, G1–G2, R1, F1–F2, D1). Đo ngành mới
bằng `pub_noindH` (= `pub_noind` + `--hide-api industry`). Sau khi xong kế hoạch: dừng, đọc lại findings, đề xuất hướng mới, lên kế
hoạch chi tiết (chia luồng, batch, thời gian) xong rồi mới chạy (user 13/11).

**Quyết định user 14/11 02:30 (mới nhất, bắt buộc):** (1) hết thời gian, đã có đội đạt 96 % → **không chạy held-out nữa**, chỉ tối ưu
**public** hướng tới 1.0 (bỏ R1 / G2 và mọi đánh giá held-out mới); (2) chạy xong kế hoạch hiện tại (`docs/17`: G1 public → chọn S → V2 → F1 →
F2 → D1) thì **tạm dừng**, cập nhật **PDF 1 Scientific Findings** (đóng góp, ý nghĩa khoa học, kết quả tích cực, hạn chế / kết quả âm, bằng
chứng, điều kiện kiểm chứng) và **PDF 2 Engineering & Operations Lessons** (lập trình, môi trường, quản lý Kaggle, tổ chức thử nghiệm, lỗi
gặp, cách sửa, bài học), rồi rà soát **xoá** tệp / bản nháp / log lỗi thời chắc chắn không cần (giữ mã, cấu hình, kết quả, tài liệu quan
trọng); user sẽ đổi mô hình (Fable 5.2) để nghĩ chiến lược mới tới public = 1; (3) **GPU:** mỗi acc 2 phiên commit × 4 L4 (24 GB / GPU, 96 GB /
phiên; không có L4 → 2 T4): dùng hết cả 4 GPU mỗi phiên — FSDP 4 GPU cho huấn luyện (batch lớn nhất không OOM, gradient accumulation, mixed
precision), hoặc **mỗi GPU một biến thể thử nghiệm** (4 biến thể / phiên, 8 / acc); không tiết kiệm acc / GPU, chia việc cho ra kết quả nhanh
nhất (nối checkpoint qua acc); (4) token mới cho nhatlong1103, quynhle07, quynhtran35, nguyenhoa99, yuan1103, synchan5704 (đã lưu `~/.kaggle`).

## Nguyên tắc user (đã chốt, không hỏi lại)

- Tự động hoá tối đa; chỉ hỏi việc chỉ user làm được (token, nộp portal, accept rules Kaggle trên web).
- Mục tiêu: PA cao nhất trên private test **và nhanh nhất có thể**; tổng tham số mô hình ≤ 15B, offline, ≤ 6 giờ trên
  2×RTX 5090. Tổng quát, không vá theo DN / câu của tập test. Chỉ tạo file nộp khi kết quả tốt.
- Token Kaggle chỉ ở `~/.kaggle/tok_<username>` (600); container mất khi sang khung chat mới → xin lại user.
- **Quota GPU Kaggle (user, 10/11 — bắt buộc):** acc có số giờ GPU còn lại **dưới 6 h** → trước khi giao task mới cho nó phải kiểm tra
  xem còn **đủ giờ chạy task đó** không, rồi mới chạy. Số giờ thật: `python3 automation/quota.py` (= `kaggle quota`; làm mới hằng tuần,
  W46: 14/11 00:00 UTC). Phiên 4×L4 tốn ~1.5–2.2 h quota mỗi giờ chạy tường (đo 10/11; tính hệ số 2.5). `launch_many.py` (mục plan có `est_h` = giờ tường ước tính) và
  failover của `fanout.py` tự kiểm tra và từ chối acc không đủ giờ.
- **Tốc độ trước, không tiết kiệm acc / GPU (user 12/11 — bắt buộc):** thiết kế luồng và chia việc sao cho ra kết quả nhanh nhất: mỗi nhánh huấn luyện một acc, FSDP trên cả 4 GPU (`training/full_job.py`, ~1.75k tok/s, 14.7M token ≈ 2.4 h), nối checkpoint qua nhiều acc khi một acc không đủ giờ (tải output → đăng dataset ở acc kế → `--resume-from`), chia tập đánh giá ra nhiều phiên / acc chạy song song, dùng 2 phiên GPU mỗi acc. **Trước khi chạy** kiểm số giờ GPU còn lại (`automation/quota.py`; đo 12/11: 4×L4 ≈ 2.0–2.2 h quota / giờ tường) có đủ chạy xong task không; không đủ → tối ưu bằng các cách trên để giảm giờ chạy; vẫn không đủ → tìm giải pháp khác; không có giải pháp thì **không chạy**.
- **Giới hạn 12 h mỗi phiên Kaggle (user, 10/11 — bắt buộc):** phiên commit bị dừng ở 12 h và mất output → tác vụ dài phải chia thành nhiều
  phiên hoặc lưu checkpoint. Đã cài: runner của kernel (`make_kernel.py`) đặt hạn chót = 12 h − 40 phút, truyền `ZC_DEADLINE` cho mọi job,
  không bắt đầu job tuần tự không kịp (ghi `results/_skipped_jobs.json` → chạy ở kernel mới), dừng job quá hạn để output đã xong được lưu;
  huấn luyện (`sft.py`, `sft_unsloth.py`, `dpo.py` qua `arms_job.py` / `full_job.py`) lưu checkpoint (LoRA + optimizer + scheduler + RNG)
  vào output rồi dừng trước hạn, nhánh ghi `partial`; kernel sau mount output đó (`--ks`) với `--resume-from auto` để chạy tiếp; đánh giá
  (`eval_flow.py`) nhận `--budget` theo hạn chót. Lập kế hoạch: `launch_many.py` với `est_h` > 11 h → tách các dòng job độc lập ra 2
  kernel (`"independent": true`), chạy chuỗi checkpoint (`"checkpointed": true`; mỗi phần ≤ min(11 h, quota còn lại / 2.5)) hoặc từ chối.
  Test: `tests/test_session_limit.py`. Lưu ý: 4×L4 tốn ~1.5–2.2 h quota mỗi giờ → acc 30 h chạy được ~12–15 h tường mỗi tuần.
- **Dataset ngoài (user, 08/11):** dataset public là dùng được (kể cả giấy phép nghiên cứu / giáo dục), không cần hỏi;
  chỉ dùng dataset uy tín (hội nghị / tạp chí có hạng hoặc nhiều citation); không bao giờ dùng public / private test của
  cuộc thi để train. Bảng thống kê: `docs/07_DATASETS.md`.
- Theo dõi kernel bằng watcher nền, không poll; tìm tài liệu: đọc văn bản / paper, không đọc code repo người khác.

## Playbook tốc độ GPU (user yêu cầu ghi nhớ, 2026-11-02) — chi tiết: `docs/05_GPU_OPTIMIZATION.md`

1. **Đo trước khi tối ưu.** Mọi kernel chạy qua `src/run_kaggle.py` → `gpu_monitor.csv` + `gpu_summary.json`
   (util trung bình, tỉ lệ mẫu bận, VRAM đỉnh). Không kết luận "đã tận dụng GPU" khi chưa có số này.
2. **4×L4 = 4 GPU 24 GB độc lập, không NVLink** (không phải 1 GPU 96 GB). Mô hình vừa 1 GPU → **data parallel**
   (`--dp 4`, mỗi GPU một bản); chỉ dùng TP khi không vừa. Lượng tử hoá (AWQ / FP8 `--quant fp8`) để dư VRAM cho KV.
3. **VRAM cho KV cache:** `--gpu-util 0.92`; đọc dung lượng KV thật từ vLLM (dòng `[engine] ... KV cache N tokens`);
   prefix budget = ½ KV (tự động).
4. **Prefix cache:** phần chung (rulebook + README + card) đặt **trước**, phần riêng mỗi request (`# EXAMPLES`,
   `# QUESTION`, `# TASK`) đặt **sau**; gom request theo DN; không trộn nhiều DN trong một đợt.
5. **Không có rào chắn giữa các vòng:** `--pipeline` (`src/pipeline.py`) — request vào vLLM từng cái
   (`add_request`/`step`), chương trình chạy CPU song song khi GPU vẫn sinh, sửa lỗi đưa vào hàng đợi ngay.
6. **Cân bằng tải động:** các tiến trình GPU tự nhận DN tiếp theo (claim file nguyên tử, DN nhiều câu trước); không chia
   tĩnh.
7. **Đầu ra ngắn là đòn bẩy lớn nhất:** việc của LLM phải hẹp (chương trình toolkit ~100 token, ánh xạ ~1.5k token);
   ngữ nghĩa nằm trong code đã kiểm chứng (`src/dm.py`).
8. Thử thêm khi cần: `--max-num-seqs`, `--max-batched-tokens`, `--spec-mtp N` (MTP của Qwen3.5); chỉ giữ khi đo nhanh hơn
   mà PA không giảm.
9. **Kaggle:** dồn mọi luồng lên một acc (2 phiên GPU + 5 phiên CPU) rồi mới sang acc khác; L4 cần acc đã accept rules
   `gemma-4-developer-agent`; probe acc bằng kernel 1 dòng; QUEUED > 8 phút (L4) → huỷ, đẩy lại (tối đa 5 lần).
10. **Bẫy công cụ:** watcher `automation/watch_done.py` chạy `run_in_background` với `timeout` 7200000 ms (mặc định
    30 phút sẽ bị kill); `A && B && nohup C &` đưa cả chuỗi vào subshell (mất biến) → dùng `(…) &`.
    `run_kernel.py` chỉ đọc trạng thái từ dòng `has status "KernelWorkerStatus.X"`: lỗi API 503 từng bị đọc thành `error`;
    sửa xong thì attach lại bằng `--attach`. Không import thư viện thẳng từ mount `/kaggle/input` (rất chậm). Thay đổi ở
    phần chỉ chạy trên Kaggle (môi trường, cài đặt) → chạy thử nhỏ trước.

## Cổng kiểm tra trước khi chạy (user chốt 2026-11-02: "code xong phải kiểm tra lại rồi mới được chạy")

Mục tiêu: không còn run chết giữa chừng hay chạy cấu hình chưa tối ưu. Trước **mọi** lần đẩy kernel:

1. **Tự review diff**: đọc lại từng hàm vừa sửa, tìm lỗi logic / công thức (deadline, ngân sách thời gian, đếm, chỉ số,
   vòng lặp, đường dẫn), trường hợp biên (rỗng, None, hết giờ, worker chết, nhiều tiến trình), và chỗ chưa tối ưu
   (GPU chờ CPU, tải lệch, đọc lại dữ liệu, bộ nhớ). Ghi các lỗi tìm được vào commit message.
2. **`automation/preflight.py "<jobs>"`** phải in `PREFLIGHT OK`: flag hợp lệ theo `agent.build_parser()`, file kernel
   sinh offline đúng (`OUT/...` nằm trong `/kaggle/working/results`), `tests/run_all.sh` PASS, chạy thử end-to-end đúng
   lệnh đó trên CPU với mô hình giả (`--fake-responder`, `--subset`, cùng `--dp` / `--pipeline` / `--budget`).
   `fanout.py` tự chạy preflight và **từ chối khởi chạy** nếu chưa qua (cache theo hash code + jobs; `--no-preflight`
   chỉ dùng cho kernel không chạy agent, ví dụ tải mô hình).
3. Phần không thử được trên CPU (vLLM, GPU): đối chiếu API với lần chạy GPU gần nhất đã thành công; thay đổi lớn ở
   phần này → chạy nhỏ (`--subset`) một lần trước khi chạy toàn bộ.

Test CPU (`tests/run_all.sh`, chi tiết `tests/README.md`): `test_ingest_robust`, `test_engine_batching`, `test_sandbox`,
`test_agent_v2`, `test_pipeline` (gồm chạy CLI dưới `__main__`, `--dp 2` với claim, ngân sách thời gian), và
`eval_lib.py` (oracle, chạy với `--clean question_cleaning/train_questions_clean.jsonl`: phải giữ 2967 đúng / 1 sai / 32 chưa đọc).
