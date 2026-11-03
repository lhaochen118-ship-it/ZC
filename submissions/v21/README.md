# File nộp public test — hệ v21

Code: commit `8dca4a2` = v20 + luật đọc câu tổng quát + các kiểm tra ánh xạ mới: doanh thu đơn ≠ tổng tiền các dòng,
order_time không có giờ, tên hàng viết tắt có bảng tra, tên thợ gọi tắt, cột doanh thu theo mô tả README. Thêm dòng
máy `…GB` và quy một người nhiều cách viết tên về một.

`submission.jsonl` = bỏ phiếu 3 lần chạy (seed 0 / 1 / 2) có tie-break theo ánh xạ (`map_penalties.json`, commit 7fa70f2).
Đã kiểm định dạng: 1000 id, 0 lỗi, **70 câu abstain**, 42 câu unanswerable.

- Train (một lần chạy, `huyenco/zc-v21-1`): PA **0.984** (2965 đúng / 13 sai / 22 abstain; dev 0.991, holdout 0.971).
- Ba lần chạy public: 856 câu nhất trí / 72 câu đa số / 72 câu tách. v19 là 820 / 95 / 85, tức ánh xạ ổn định hơn.
- So với v19 (`submissions/v19`, portal 0.687): 927 câu giống hệt; 37 câu v21 trả lời mà v19 abstain (dev_014: 19);
  22 câu v21 abstain mà v19 trả lời (dev_012: 7); 14 câu đổi đáp án. dev_003 và dev_009 giống hệt v19 (v20 mất điểm ở
  hai DN này).
