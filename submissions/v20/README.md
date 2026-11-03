# File nộp public test — hệ v20

Code: commit `5dba631` = v19 + chuyển ánh xạ tham chiếu theo quy ước README của DN mới (`--ref-adapt 3`, de097f9) +
trung bình mỗi ngày thứ X theo rulebook public (`weekday_avg`).

`submission.jsonl` = bỏ phiếu 3 lần chạy (seed 0 / 1 / 2), cùng cách `docker/predict.sh` làm trên máy chấm. Đã kiểm định
dạng: 1000 id, 0 lỗi, **117 câu abstain**, 22 câu unanswerable.

- Train (một lần chạy): **PA 0.991** (2975 đúng / 2 sai / 23 abstain; dev 0.998, **holdout 0.977**, v19: 0.961) — cao nhất.
- Public so với v19 (`submissions/v19`, portal 0.687): 918 câu giống hệt; 52 câu v20 abstain mà v19 có trả lời (dev_003: 18,
  dev_009: 9 — hai DN mà các lần chạy chọn ánh xạ khác nhau); 20 câu v20 trả lời mà v19 abstain; 10 câu đổi đáp án.
- Phần chuyển ánh xạ theo README không đổi đáp án nào ở các DN public dùng ánh xạ tham chiếu (dev_005 giống hệt, dev_002
  gần như giống hệt) → **dự đoán v20 không cao hơn v19 trên portal**. Bản nên thử tiếp là **v21** (sửa đúng các lỗi ánh xạ
  của dev_003 / dev_009 / dev_012 / dev_014 + luật đọc câu), đang chạy, xong khoảng 15:00 UTC.
