# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-03 13:45 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
(kể cả sau khi hết / reset token): đọc `CLAUDE.md` → file này → làm mục "Việc tiếp theo" theo thứ tự. Routine
`trig_01U26KaZP68Dmg8JL2RGcbVS` chạy mỗi giờ đến **17:00 UTC 03/11**; sau mốc đó tắt routine và báo kết quả.*

## Điểm portal (user nộp) — EXPERIMENT_LOG entry 13

| File nộp | PA | Đúng | Sai | Abstain |
|---|---|---|---|---|
| `submissions/v13/submission.jsonl` | 0.637 | 785 | 148 | 67 |
| `submissions/v15/submission.jsonl` | 0.634 | 792 | 158 | 50 |
| `submissions/v18/submission.jsonl` | 0.669 | 800 | 131 | 69 |
| **`submissions/v19/submission.jsonl`** | **0.687** | 801 | 114 | 85 |

- Số câu đúng đứng yên ~800 từ v15; điểm tăng nhờ **bớt câu sai**. Các phiên bản trả lời gần như giống hệt nhau
  (v18 / v19: chỉ 4 câu cùng trả lời mà khác nhau) → ~114 câu sai là lỗi **chung của mọi lần chạy**: bỏ phiếu thêm
  lần chạy không gỡ được; phải sửa nguyên nhân (luật đọc câu, ánh xạ, quy ước README).
- Train (LOO) không còn dự báo được public: v19 train 0.986.

## Đang chạy (13:45 UTC)

- **v20** xong: train **0.991** (holdout 0.977); public → `submissions/v20/` (117 abstain; dự đoán ≈ v19 trên portal: 52 câu
  thêm abstain ở dev_003 / dev_009 do các lần chạy chọn ánh xạ khác nhau). Đã push + gửi user.
- **v21** (8dca4a2: luật đọc câu + kiểm tra ánh xạ doanh thu / giờ / tên viết tắt / tên thợ / cột doanh thu theo README +
  dòng máy …GB): huyenco / mucnokieu, khởi chạy 12:52, xong ~15:00.
- **v22** (a6f5800 = v21 + từ địa phương cho tên hàng 'cá tràu' = 'cá lóc', 'đậu phụ rán' = 'đậu hũ chiên', 'cốc sạc' =
  'củ sạc'… + size theo cột danh mục): preflight 13:41, chạy trên caotienminh1 / duylnng, xong ~16:00.
- Đĩa: đã xoá 22 GB thư mục tạm cũ (user cho phép); run_all.sh / preflight.py nay tự dọn.

## Việc tiếp theo (thứ tự)

1. v21 / v22 xong → chấm train, `submissions/v21|v22/submission.jsonl` (bỏ phiếu 3 lần chạy public) + README, **push ngay**,
   gửi user; so với v19 (portal 0.687).
2. Lỗi chung còn biết mà chưa sửa: nợ chỉ ghi trong ghi chú tự do (dev_008: 'còn thiếu …', 'còn lại ghi sổ' → mọi lần chạy
   trả 0); định nghĩa ngành điện thoại (dòng máy không gồm phụ kiện); câu trả lời bằng 0 là tín hiệu tốt để tìm lỗi lọc.
3. 17:00 UTC: tắt routine `trig_01U26KaZP68Dmg8JL2RGcbVS`, báo user.
