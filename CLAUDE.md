# CLAUDE.md — bộ nhớ dự án (Claude Code tự đọc file này mỗi phiên)

Zalo AI Challenge 2026, track DataMate. Branch làm việc **`claude/datamate-v3-0dny90`** (repo `lhaochen118-ship-it/ZC`).
**Khung chat mới: đọc `READING_GUIDE.md` trước** (bàn giao: `handoff/STATUS.md`, `handoff/03_HANDOFF.pdf`,
`handoff/01_SCIENTIFIC_FINDINGS.pdf`, `handoff/02_ENGINEERING_OPS_LESSONS.pdf`, `handoff/environment_summary.pdf`).

**Quyết định mới nhất (user, 08/11): hybrid framework = AI Agent sinh code + luật tổng quát hỗ trợ.** Cấm hard code
**answer**: luật sinh thẳng code Python / đáp án 100 % mà không qua mô hình. Được hard code **logic**: quy tắc tổng quát
nhận diện ý định câu hỏi, đọc data, viết schema (`ingest`, `automap`; Agent viết thêm mô tả schema khi cần), toolkit
nghiệp vụ (`dm`, `industry`), làm sạch câu, ví dụ few-shot, kiểm tra bằng code — để Agent sinh code nhanh / chính xác hơn.
Code trả lời của **mọi** câu do mô hình sinh; chương trình luật (`qparse`) chỉ là gợi ý / kiểm chéo. Chi tiết:
`handoff/03_HANDOFF.md` §2.1; **thiết kế + tổng quan tài liệu + kế hoạch thí nghiệm: `docs/06_HYBRID_FRAMEWORK.md`**
(luồng 8 bước, giao thức xử lý câu sai, điểm tin cậy + sinh lại, precheck, tool, mô hình phụ, chưng cất);
**kế hoạch chính + đặc tả triển khai + lộ trình 08–16/11: `docs/08_MASTER_PLAN.md`** — user đã duyệt; **việc của khung chat
kế tiếp: triển khai code đúng theo tài liệu này (gói việc §8 theo thứ tự WP0 → WP12, đặc tả §5, quy tắc quyết định §10)**;
**phương án đào sâu (tool cho Agent, sinh có ràng buộc ngữ pháp, mô hình phụ, chưng cất, chọn / tin cậy): `docs/09_OPTIONS_DEEP_DIVE.md`**. Luồng nộp hiện tại (`src/predict_flow.py`, stage 1 luật trả lời thẳng → `canon_llm.py` →
`agent.py`) phải được thay. Đo bằng held-out: paraphrase public k=1,2 và paraphrase train
(`src/paraphrase_llm.py`, `eval_split.py --perturb-q para:`, `stress_flow.py --modes para:`) và 6 DN train giữ ngoài
huấn luyện; được phân tích lỗi trên public gốc, k=0 public và train (không xem từng câu held-out). Mốc so sánh: v34 held-out public 0.602; v28 (LLM) public gốc 0.921.

**CHẾ ĐỘ CHẠY TỰ ĐỘNG (user, 08/11 — bắt buộc):** Claude Code chạy tự động cho đến khi thực thi xong **toàn bộ** kế hoạch
`docs/08_MASTER_PLAN.md` (WP0 → WP12). **Không được dừng lại** trước khi xong; không hỏi ý kiến user (chỉ xin những thứ chỉ user
làm được: token, acc mới, nộp portal). **Hết hạn mức sử dụng của Claude (phiên 5 giờ)** giữa chừng → tạm dừng; khi hạn mức reset
thì tự động chạy tiếp từ chỗ dở. (Hết quota GPU Kaggle → làm phần code / CPU, chạy tiếp phần GPU khi quota reset hoặc có acc mới.)
Tiếp tục đúng chỗ dở nhờ bảng "Tiến độ WP" trong `handoff/STATUS.md` (cập nhật sau mỗi bước; không làm lại bước đã xong; kernel
đang chạy thì attach). Ngay đầu khung chat mới: tạo routine tự tiếp tục mỗi giờ (`create_trigger`, tiếp tục vào chính phiên này)
với lời nhắc "đọc handoff/STATUS.md, làm tiếp bước chưa xong đầu tiên của docs/08; không dừng đến khi xong toàn bộ kế hoạch" —
routine là cơ chế đánh thức lại sau khi hạn mức Claude reset; chỉ xoá routine khi toàn bộ kế hoạch đã xong.

## Nguyên tắc user (đã chốt, không hỏi lại)

- Tự động hoá tối đa; chỉ hỏi việc chỉ user làm được (token, nộp portal, accept rules Kaggle trên web).
- Mục tiêu: PA cao nhất trên private test **và nhanh nhất có thể**; tổng tham số mô hình ≤ 15B, offline, ≤ 6 giờ trên
  2×RTX 5090. Tổng quát, không vá theo DN / câu của tập test. Chỉ tạo file nộp khi kết quả tốt.
- Token Kaggle chỉ ở `~/.kaggle/tok_<username>` (600); container mất khi sang khung chat mới → xin lại user.
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
