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

- Branch làm việc: `datamate` (repo `lhaochen118-ship-it/ZC`). Không tạo PR nếu user không yêu cầu.
- Commit kèm trailer theo hướng dẫn của phiên; không ghi model ID vào commit.
- Kết quả mới để file mới (`results/<run>/…`), không ghi đè kết quả cũ.
- Không commit token, dữ liệu sinh ra lớn, cache (`.gitignore`).
