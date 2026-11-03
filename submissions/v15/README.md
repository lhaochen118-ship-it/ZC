# File nộp public test — hệ v15

Code: commit `7c28f66`. So với v13, bản này có thêm:
- chọn ánh xạ theo đồng thuận đáp án giữa các ứng viên;
- fallback sang ánh xạ kế tiếp khi ánh xạ được chọn không trả lời được;
- bộ đọc luật phủ thêm nhiều cách hỏi.

`submission.jsonl` là kết quả bỏ phiếu của 3 lần chạy (seed 0 / 1 / 2) bằng `src/vote.py`. Đã kiểm định dạng bằng
`automation/make_submission.py`: 1000 id, 0 lỗi, 50 câu bỏ qua (abstain).

Ước lượng trên train (một lần chạy): PA 0.982 (dev 0.999, holdout 0.949).

Bản này **chưa** có cách xử lý ngành mới của rulebook public. Phần đó nằm ở v18 / v19 (xem `submissions/v18`, `submissions/v19` khi xong).
