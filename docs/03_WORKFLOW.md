# 03 — Quy trình làm việc

*Cập nhật 08/11/2026 (§5 thay bằng vòng lặp LLM-first + held-out; §1b acc mới). Giữ tinh thần quy trình gốc (`docs/RESEARCH_WORKFLOW.pdf`), đổi đích từ "paper" sang "điểm PA
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
- Acc và quota mới nhất: `handoff/ENVIRONMENT_SUMMARY.md` §2 (08/11: L4 quynhtran35, yuan1103, phatle1578; T4
  yeshenhi, mxuanquy). **Quota 30 giờ GPU / tuần / acc, tính theo GPU** (phiên 4×L4 = 4 h quota / giờ), reset thứ Bảy
  00:00 UTC. Acc dùng chung với dự án khác: không đụng kernel không phải `zc-*`.
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
  python3 automation/watch_done.py <runs> --max 100
  ```

  chạy bằng Bash `run_in_background`. Nó thoát (→ Claude được báo ngay) khi **bất kỳ** kernel nào xong, in tên kernel và
  trạng thái; xử lý xong thì bật lại. Hết 100 phút thì in `WATCH_TIMEOUT` để bật lại (tác vụ nền có trần ~2 giờ).
- **Tìm thông tin:** chỉ đọc văn bản / paper (alphaXiv, web), **không đọc code repo của người khác**.
- **Dữ liệu lớn:** chỉ đọc cấu trúc và vài dòng (`head`, `python -c` in shape/cột), không in cả file.
- Log dài: chỉ đọc tóm tắt (`score.py` theo nhóm, `--errors N`).

## 4. Git

- Branch làm việc: `claude/datamate-v3-0dny90` (repo `lhaochen118-ship-it/ZC`). Không tạo PR nếu user không yêu cầu. Chỉ tạo file nộp thử khi kết quả tốt (user chốt).
- Commit kèm trailer theo hướng dẫn của phiên; không ghi model ID vào commit.
- Kết quả mới để file mới (`results/<run>/…`), không ghi đè kết quả cũ.
- Không commit token, dữ liệu sinh ra lớn, cache (`.gitignore`).

## 5. Vòng lặp hiện tại (từ 08/11: LLM-first, đo bằng held-out)

User chốt (08/11): hybrid — cấm luật sinh thẳng code / đáp án không qua mô hình; được dùng luật tổng quát (ý định, schema,
làm sạch câu, toolkit, kiểm tra) làm ngữ cảnh cho Agent; LLM hiểu câu, phân tích schema, sinh code; code
tính toán và kiểm tra; abstain theo logic + độ tin cậy (`handoff/03_HANDOFF.md` §2). Vòng lặp:

1. **Thước đo cố định trước khi sửa**: public gốc (`public_gt.jsonl`), train, paraphrase public k=0 (dev) / k=1,2
   (held-out), paraphrase train (held-out), biến đổi dữ liệu (đổi tên cột, README trắng). Chấm bằng `src/score.py`.
2. **Phân tích lỗi chỉ trên dev** (train + paraphrase k=0 public), theo nhóm (ánh xạ, đọc câu, toolkit, abstain).
3. **Sửa bằng cơ chế chung** (prompt, ví dụ truy hồi, kiểm tra, toolkit đã kiểm) + test; không mẫu theo câu / DN.
4. Ước lượng offline trên đầu ra đã có khi được; preflight; chạy Kaggle; **giữ khi held-out cải thiện** và thời gian
   ước lượng trên 2×5090 ≤ 6 h; ghi `EXPERIMENT_LOG.md` (kể cả kết quả âm).
5. File nộp: `automation/make_submission.py` → `submissions/vNN/` + README → push → báo user.

## 6. Nhiễu của điểm public

- Các lần chạy cùng cấu hình chỉ giống nhau 80–89 % câu (mẫu ánh xạ LLM không tái lập). Điểm public dao động **±0.03**
  chỉ vì may rủi ánh xạ → một lần nộp không đủ để kết luận một sửa đổi tốt / xấu nếu chênh lệch < 0.03.
- Ưu tiên các sửa giảm phương sai (ổn định ánh xạ, bỏ phiếu biết ánh xạ) trước các sửa tăng trung bình nhỏ.

## 7. Chuyển khung chat

- Trước khi đóng: commit + push mọi thứ; chép job / script khỏi scratch vào repo (`automation/jobs/`); cập nhật
  `handoff/STATUS.md`, `EXPERIMENT_LOG.md`, ba PDF bàn giao (`handoff/01_*.pdf`, `02_*.pdf`, `03_HANDOFF.pdf`), `handoff/environment_summary.pdf`;
  tắt routine tự chạy.
- Khung mới: đọc `READING_GUIDE.md` → `CLAUDE.md` → `handoff/STATUS.md`; xin token; tải lại output kernel đang chạy.
