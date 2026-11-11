# 03 — Quy trình làm việc

*Cập nhật 11/11/2026 (§5 vòng lặp kế hoạch 2: public gốc + held-out, sàng lọc → xác nhận; §1b quota / 12 h; §7 bàn giao qua `Process/`). Giữ tinh thần quy trình gốc (`docs/RESEARCH_WORKFLOW.pdf`), đổi đích từ "paper" sang "điểm PA
tối đa trên private test, trong 6 giờ, tái lập được".*

## 1. Nguyên tắc chung

- **Tự động hoá tối đa.** Tự đề xuất → chạy → kiểm chứng → quyết định; không hỏi lại những quyết định đã có trong tài
  liệu. Chỉ hỏi user việc chỉ user làm được (token, nộp portal, accept rules, xoá kernel bị chặn, quyền public dữ liệu).
- **Mỗi vòng lặp:** Đề xuất → Phản biện → Xây → Chạy trên Kaggle → Chấm (`src/score.py`) → Phân tích lỗi theo nhóm →
  Giữ / sửa / bỏ → ghi `EXPERIMENT_LOG.md`.
- **Một thí nghiệm quyết định mỗi lần**, nhưng khi đã chắc thì chạy song song tối đa trên mọi GPU.
- **Đo lỗi trước khi sửa:** luôn lưu đầu ra từng câu (plan, code, kết quả thực thi, đáp án, p) để phân tích offline;
  tìm vị trí và điều kiện lỗi rồi mới đề xuất bản sửa.
- **Chọn siêu tham số bằng dữ liệu tách riêng** (holdout theo doanh nghiệp), một cấu hình chung; mọi biến thể chọn sau
  khi nhìn dữ liệu phải được ghi rõ và kiểm lại trên holdout.
- **Báo cáo kết quả âm trung thực**; ý tưởng không cải thiện thì loại và ghi lý do.
- **Kiểm dữ liệu đầu vào trước khi kết luận** (header Excel, định dạng số/ngày, mã hoá tiếng Việt).
- **Không vi phạm luật BTC:** không dùng test để train, không logic theo id, tổng tham số ≤ 15B, offline.

## 1a. Cổng kiểm tra trước khi chạy (user chốt 2026-11-02)

Code xong → tự review diff (logic, công thức, biên, tối ưu) → `automation/preflight.py "<jobs>"` in `PREFLIGHT OK`
(flag, file kernel, test CPU, chạy thử end-to-end trên CPU với mô hình giả) → mới được đẩy kernel. `fanout.py` tự kiểm
và từ chối khởi chạy nếu chưa qua. Chi tiết: `CLAUDE.md`.

## 1b. Nguyên tắc dùng Kaggle (ZAIC.docx §4, cập nhật 2026-11-02)

- Dùng **tất cả** acc được cấp. Token chỉ ở `~/.kaggle/tok_<username>` (quyền 600), không đưa vào code, prompt, GitHub
  hay báo cáo; kiểm tra kết nối bằng `kaggle competitions list`. Khung chat mới phải xin lại token.
- Acc và quota mới nhất: `handoff/ENVIRONMENT_SUMMARY.md` §2 và `python3 automation/quota.py` (24 acc, 11/11). **Quota 30 giờ GPU /
  tuần / acc** (phiên 4×L4 ≈ 1.5–2.2 h quota / giờ tường), làm mới thứ Bảy 00:00 UTC. **Acc < 6 h: kiểm đủ giờ trước khi giao task**
  (tự động trong `launch_many.py`). **Phiên ≤ 12 h**: runner tự đặt hạn chót, huấn luyện tự lưu checkpoint (`CLAUDE.md`).
  Acc dùng chung với dự án khác: không đụng kernel không phải `zc-*`.
- **Dồn luồng lên một acc**: mỗi acc tối đa 2 phiên GPU + 5 phiên CPU cùng lúc; hết slot mới sang acc khác.
  Mỗi phiên GPU = 4×L4 (24 GB/GPU, 96 GB phân tán, không phải một GPU 96 GB); không có L4 → 2×T4.
- **Kẹt hàng đợi:** không có đầu ra sau ~8 phút (L4 / CPU) hoặc ~10 phút (T4) → kiểm trạng thái; L4 kẹt hàng đợi → ngắt,
  tạo phiên mới, tối đa 5 lần; vẫn hỏng → xem lỗi code / logic, sửa rồi chạy lại; không phải lỗi code → chuyển 2×T4.
  Vẫn hỏng → xem log, quota, cấu hình.
- Tận dụng hết GPU trong phiên: inference `--dp <N>` (N = số GPU) (mỗi GPU một bản mô hình, chia DN cân bằng theo số câu,
  prefix caching theo DN); mô hình không vừa 1 GPU thì `--tp`. Fine-tune: DDP mọi GPU, bf16 (L4) / fp16 (T4),
  gradient accumulation, gradient checkpointing chống OOM.
- Hết GPU: việc rất nhẹ chạy CPU, việc nặng đợi GPU. Định kỳ dọn tiến trình treo (kernel Kaggle và tiến trình nền cục bộ);
  không ngắt được thì báo user.
- Cần thêm acc (và Kaggle không chặn API) → báo user.

## 2. Tối ưu thời gian chạy

- Cache mọi thứ tất định (ingest, profile, entity index) và đăng thành dataset public để acc nào cũng mount.
- Lưu đầu ra LLM theo câu → thay đổi ở phần quyết định / formatter / voting chấm lại **offline trong vài giây**, không
  chạy lại LLM.
- Chia câu hỏi theo doanh nghiệp ra nhiều kernel / GPU; trong kernel dùng vLLM batch + prefix caching.
- Thử nhỏ (vài doanh nghiệp) trước khi chạy toàn bộ.

## 3. Tối ưu token cho Claude

- **Không chủ động kiểm tra kernel.** Sau khi khởi chạy, bật một watcher nền:

  ```bash
  python3 automation/watch_done.py <runs> --max 115
  ```

  chạy bằng Bash `run_in_background`. Nó thoát (→ Claude được báo ngay) khi **bất kỳ** kernel nào xong, in tên kernel và
  trạng thái; xử lý xong thì bật lại. Hết 115 phút thì in `WATCH_TIMEOUT` để bật lại (tác vụ nền có trần ~2 giờ).
- **Tìm thông tin:** chỉ đọc văn bản / paper (alphaXiv, web), **không đọc code repo của người khác**.
- **Dữ liệu lớn:** chỉ đọc cấu trúc và vài dòng (`head`, `python -c` in shape/cột), không in cả file.
- Log dài: chỉ đọc tóm tắt (`score.py` theo nhóm, `--errors N`).

## 4. Git

- Branch làm việc: `claude/datamate-v3-0dny90` (repo `lhaochen118-ship-it/ZC`). Không tạo PR nếu user không yêu cầu. Chỉ tạo file nộp thử khi kết quả tốt (user chốt).
- Commit kèm trailer theo hướng dẫn của phiên; không ghi model ID vào commit.
- Kết quả mới để file mới (`results/<run>/…`), không ghi đè kết quả cũ.
- Không commit token, dữ liệu sinh ra lớn, cache (`.gitignore`).

## 5. Vòng lặp hiện tại (kế hoạch 2, từ 09/11: đến public gốc 100 %, held-out không giảm)

Hybrid (user 08/11): mô hình viết chương trình trả lời mọi câu; luật tổng quát chỉ làm ngữ cảnh / kiểm tra. Vòng lặp
(`docs/13_TASK_AFTER_PLAN1.md`, `docs/12_PLAN2.md`):

1. **Thước đo cố định**: public gốc (`pub_full`, chấm với `data/raw/public_gt.jsonl`), held-out `pub_ho` (paraphrase public) +
   `tr_ho` (6 DN train giữ ngoài) → báo cả hai ở mọi mốc; tập con sàng lọc `pub_ho400`, `tr_ho400`.
2. **Phân tích lỗi trên public gốc / train** (`tests/list_wrong.py`, taxonomy 7 nhóm `docs/10`); held-out chỉ xem tổng.
3. **Sửa bằng cơ chế chung** (gợi ý, làm sạch, kiểm tra có tín hiệu sạch trên mọi log, toolkit) hoặc **phía mô hình** (dữ liệu
   fine-tune, thành viên, bộ chọn) + test; không mẫu theo câu / DN.
4. **Ước lượng offline** trên log có sẵn khi được (`tests/pool_gens.py`, `replay_vote.py`, quét tín hiệu sạch).
5. **Sàng lọc** (một ứng viên / kernel, song song nhiều acc, tập con, ~40 phút) → **xác nhận** đủ tập 2 seed → giữ / bỏ theo §4
   `docs/12` (public ↑ ≥ 0.3, held-out gộp không ↓ > 0.3).
6. Sau mỗi lần chạy có public: `automation/update_final_submission.py <run>`; đổi mô hình nộp: `automation/set_final_model.py
   --publish` + chạy đường container; ghi `handoff/STATUS.md` (+ `Process/scores.csv` khi bàn giao), commit + push.

## 6. Nhiễu của điểm public

- Luồng hybrid hiện tại: một lần chạy dao động public **±0.4 điểm**, held-out gộp ±0.3; chất lượng còn phụ thuộc seed huấn luyện
  (cùng dữ liệu: 0.991 vs 0.983) → mọi quyết định ≥ 2 seed; tập con 400 câu nhiễu ±0.6 (chỉ để loại).
- Ưu tiên các sửa giảm phương sai (ổn định ánh xạ, bỏ phiếu biết ánh xạ) trước các sửa tăng trung bình nhỏ.

## 7. Chuyển khung chat

- Trước khi đóng: commit + push mọi thứ; lưu vào `Process/` những gì không nằm trong git (log nén của lần chạy then chốt, dữ liệu huấn
  luyện nén, plan khởi chạy, sổ kernel, quota, bản vá worktree; `Process/README.md`); cập nhật `handoff/STATUS.md`, ba PDF bàn giao,
  `handoff/environment_summary.pdf`, `docs/RESEARCH_WORKFLOW.pdf`, `READING_GUIDE.md`; tắt routine tự chạy của phiên cũ.
- Khung mới: đọc `READING_GUIDE.md` → `CLAUDE.md` → `handoff/STATUS.md`; xin token; tạo routine mỗi giờ; tải output kernel đang chạy
  (`Process/kaggle/kernels.csv`).
