# 03 — Quy trình làm việc

*Giữ tinh thần quy trình gốc (`docs/RESEARCH_WORKFLOW_original.pdf`, `git show de4360e:handoff/04_RESEARCH_WORKFLOW.md`), đổi đích từ "paper" sang "điểm PA
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

## 1b. Nguyên tắc dùng Kaggle (ZAIC.docx §4, cập nhật 2026-11-02)

- Dùng **tất cả** acc được cấp. Token chỉ ở `~/.kaggle/tok_<username>` (quyền 600), không đưa vào code, prompt, GitHub
  hay báo cáo; kiểm tra kết nối bằng `kaggle competitions list`. Khung chat mới phải xin lại token.
- Acc khung chat 3 (username Kaggle): `thnliu`, `linhmai24`, `lelynhan`, `mxuanquy`, `duylnng`, `caotienminh1`,
  `nguyenmanh0404`, `hai1108`, `nguynduylm`, `phuongnehe`. Probe 02/11: **4×L4** ở `lelynhan`, `linhmai24`, `thnliu`, `duylnng`, `caotienminh1`, `nguynduylm`, `phuongnehe` (2 acc user gửi thêm 02/11);
  `mxuanquy` chỉ T4; `nguyenmanh0404`, `hai1108` chưa accept rules `gemma-4-developer-agent` (chỉ T4 đến khi user accept
  trên web). Các acc này **dùng chung với dự án khác** (kernel `e27*`, `e32*`, `nclt-*`…): không
  đụng vào kernel không phải `zc-*`. Không có token `nhatlong1103` → dataset mới đăng public trên một acc ở trên.
- **Dồn luồng lên một acc**: mỗi acc tối đa 2 phiên GPU + 5 phiên CPU cùng lúc; hết slot mới sang acc khác.
  Mỗi phiên GPU = 4×L4 (24 GB/GPU, 96 GB phân tán, không phải một GPU 96 GB); không có L4 → 2×T4.
- **Kẹt hàng đợi:** không có đầu ra sau ~8 phút (L4 / CPU) hoặc ~10 phút (T4) → kiểm trạng thái; L4 kẹt hàng đợi → ngắt,
  tạo phiên mới, tối đa 5 lần; vẫn hỏng → xem lỗi code / logic, sửa rồi chạy lại; không phải lỗi code → chuyển 2×T4.
  Vẫn hỏng → xem log, quota, cấu hình.
- Tận dụng hết GPU trong phiên: inference `--dp <số GPU>` (mỗi GPU một bản mô hình, chia DN cân bằng theo số câu,
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
