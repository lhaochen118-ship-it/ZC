# final_submission — bản nộp có điểm public cao nhất hiện tại

- `submission.jsonl`: 1000 dòng (đúng thứ tự `questions.jsonl` của public test), sinh bởi luồng hybrid (mô hình viết chương trình cho mọi câu); không sửa tay.
- `score.json`: điểm chấm offline bằng luật của BTC với `data/raw/public_gt.jsonl` (PA, đúng / sai / bỏ) và lần chạy nguồn.
- Tự động cập nhật (user 10/11): sau mỗi lần chạy, `python3 automation/update_final_submission.py <run_dir>` — chỉ thay khi PA public **cao hơn hẳn**.
- File này để nộp thử lên portal; bản nộp chính thức là Docker (`docker/`, `configs/flow_final.json`), BTC chạy lại trên private test.
