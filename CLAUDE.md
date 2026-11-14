# CLAUDE.md — bộ nhớ dự án (Claude Code tự đọc file này mỗi phiên)

Zalo AI Challenge 2026, track DataMate. Branch làm việc **`claude/datamate-v3-0dny90`** (repo `lhaochen118-ship-it/ZC`).
**Khung chat mới: đọc `READING_GUIDE.md` trước** (trạng thái sống `handoff/STATUS.md`; bản lưu không nằm trong code — log, dữ liệu huấn luyện, sổ kernel — ở `Process/`, xem `Process/README.md`) (bàn giao: `handoff/STATUS.md`, `handoff/03_HANDOFF.pdf`,
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

**NHIỆM VỤ SAU KẾ HOẠCH 1 (user, 09/11 — bắt buộc, đọc nguyên văn `docs/13_TASK_AFTER_PLAN1.md`):** xong WP0 → WP12 thì chấm public gốc
với `data/raw/public_gt.jsonl` (= `score 1.jsonl`, chỉ để chấm) và liệt kê mọi câu sai; **điều kiện dừng cuối cùng = 100 % public gốc**
đạt bằng cải tiến tổng quát mà **không giảm held-out** (pub_ho, paraphrase train, tr_ho; cải tiến tăng public mà giảm held-out → loại; báo
cả hai số ở mọi mốc). Chưa đạt → vòng Phân tích lỗi (`docs/10_ERROR_ANALYSIS.md`, taxonomy 7 nhóm) → Tài liệu ngoài
(`docs/11_LITERATURE.md`) → Kế hoạch 2 (`docs/12_PLAN2.md`, gói WP-P2-xx, lưới thí nghiệm song song) → Triển khai → Đo, lặp lại
("Vòng n" trong `docs/12_PLAN2.md`); bảng "Tiến độ P2" trong `handoff/STATUS.md`. **Kế hoạch 1 đã chạy hết; việc hiện tại = kế hoạch 2
`docs/12_PLAN2.md` (gói WP-P2-01…10, quy tắc giữ / bỏ §4)**; **user 10/11: hai hướng — R (thêm luật tổng quát, trước mắt) và M (§6:
cải tiến mô hình từ bản không luật 0.939 — mô hình phụ, fine-tune lại, dataset ngoài, chưng cất, RL — chạy song song đến public = 1.0,
không thêm luật)**; công cụ: `tests/list_wrong.py`, `tests/hint_audit.py`, `tests/replay_vote.py`. Kaggle: mỗi acc tối đa 2 phiên GPU + 5 phiên CPU,
dùng hết slot một acc rồi mới sang acc khác; việc không cần GPU đưa lên phiên CPU.

**Quyết định user 12/11:** không cần chờ public 100 % mới dừng. Triển khai **hết kế hoạch hiện tại** (`handoff/STATUS.md` §4: P seed 2 + đường container → chốt cấu hình nộp; WG9–12 sau quota 14/11 → sàng lọc / xác nhận; chốt bản nộp + Docker trước 15/11), rồi **dừng lại cùng user phân tích và xây dựng kế hoạch mới** phát triển mô hình hướng tới score = 1, sau đó mới chạy thử nghiệm tiếp.

**`final_submission/submission.jsonl` (user 10/11):** luôn là file nộp public có PA cao nhất đến nay; sau **mỗi** lần chạy GPU có public gốc,
chạy `python3 automation/update_final_submission.py <run_dir>` (chỉ thay khi PA cao hơn; ghi `final_submission/score.json`) rồi commit + push.

**Quyết định user 13/11 (bắt buộc):** duyệt **kế hoạch 3** `docs/16_PLAN3_LEARNED_RULES.md` (kiểm tra: `docs/15_AUDIT.md`) — "thực hiện hết":
mô hình thuần + **lớp luật do mô hình tự học** (quy nạp từ lỗi train, biên dịch rulebook lúc suy luận, bí danh giá trị, ViLexNorm; luật chỉ học từ train,
nhận bằng A/B trên DN khác), dữ liệu v3, gói WP3-00 → WP3-10 đến 16/11, rồi **H2** sau 16/11 (RL thưởng theo PA, mô hình chuẩn hoá câu). **Đổi bản nộp**
(§5.4) khi hệ mới có held-out gộp ≥ P và public ≥ 0.985. **WP3-04:** huấn luyện lại P trên đủ 30 DN, giữ nếu public ≥ 0.993. Tiến độ: bảng "Tiến độ P3"
trong `handoff/STATUS.md`. (13/11 08:30: bàn giao sang khung chat mới; routine cũ đã tắt — phiên mới tạo routine theo `READING_GUIDE.md` §0.) Vẫn áp dụng: kiểm quota trước khi chạy (không đủ → tối ưu: FSDP 4 GPU, chuỗi checkpoint qua acc, chia shard; vẫn không đủ →
giải pháp khác; không có thì không chạy), tốc độ trước, không tiết kiệm acc / GPU.

**Quyết định user 13/11 tối (mới nhất, bắt buộc):** (1) **ngừng phát triển luật viết tay** (hints / qparse / qnorm / cờ, precheck viết tay
của P) — khi vận hành thật có nhiều tình huống mới; luật nếu có **phải do mô hình tự viết**; (2) **bản nộp cuối là hệ mới không luật viết
tay** (S–W–V), kể cả khi điểm thấp hơn P; (3) WR30 chạy xong chỉ làm dự phòng. Kế hoạch hiện tại: **`docs/17_PLAN4_SWV.md`** (S mô hình
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
