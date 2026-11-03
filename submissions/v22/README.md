# File nộp public test — hệ v22

Code: commit `20cfec7` = v21 + từ địa phương cho tên hàng ('cá tràu' = 'cá lóc', 'đậu phụ rán' = 'đậu hũ chiên',
'cốc sạc' = 'củ sạc', 'găng tay tất chân' = 'bao tay bao chân'…) + size / số theo cột danh mục + cảnh báo đơn ghi nợ mà
không nợ gì + `money_in_text`.

`submission.jsonl` = bỏ phiếu 3 lần chạy (seed 0 / 1 / 2) có tie-break theo ánh xạ (`map_penalties.json`, commit 7fa70f2).
Đã kiểm định dạng: 1000 id, 0 lỗi, **76 câu abstain**, 33 câu unanswerable.

- Train (một lần chạy, `thnliu/zc-v22-1`): PA **0.988** (2969 đúng / 5 sai / 26 abstain; dev 0.998, holdout 0.969).
- Ba lần chạy public: 856 câu nhất trí / 68 câu đa số / 76 câu tách.
- Số đáp án bằng 0 (thường là lọc mặt hàng sai): v19 có 19 câu, v21 11 câu, **v22 3 câu**.
- So với v19 (portal 0.687): 929 câu giống hệt; 27 câu thêm trả lời; 18 câu thêm abstain; 26 câu đổi đáp án (dev_012: 9,
  dev_008: 5, dev_013: 5). dev_003 và dev_009 giống hệt v19.
- So với v21: 937 câu giống hệt; 22 câu thêm trả lời; 28 câu thêm abstain; 13 câu đổi đáp án.
