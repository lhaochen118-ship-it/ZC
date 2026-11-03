# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-03 12:05 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
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

## Đang chạy

- **v20** (code 5dba631 = ref-adapt de097f9 + weekday_avg): train + public ×3 trên caotienminh1 / duylnng, khởi chạy
  11:35. Watcher nền theo dõi `runs/active_runs.txt`.
- **v21** (code d0007a9 + tài liệu): sửa luật đọc câu tổng quát (tiếng lóng '10 củ', '3tr7', 'o đờ', 'bq', 'đh',
  'cất hàng'; top N; tăng trưởng 'so với A thì B'; nhiều tháng; so sánh; mặt hàng / tên không liên kết được; nợ theo
  hạn trả, tuổi nợ quá N ngày) + kiểm tra ánh xạ (doanh thu đơn ≠ tổng thành tiền dòng; order_time không có giờ).
  Preflight rồi chạy trên huyenco / mucnokieu.

## Việc tiếp theo (thứ tự)

1. v20 xong → chấm train, tạo `submissions/v20/submission.jsonl` (bỏ phiếu 3 lần chạy public) + README, **push ngay**,
   gửi user.
2. v21 xong → như trên (`submissions/v21/`); so sánh với v19 / v20 (đồng thuận, abstain, câu đổi đáp án).
3. Tiếp tục tìm lỗi chung không cần nhãn: luật vs LLM bất đồng, ánh xạ ứng viên bất đồng, README của DN dùng ánh xạ
   tham chiếu (dev_001 / 002 / 005 / 006), DN ngành mới (dev_011–014).
4. 17:00 UTC: tắt routine, báo user.
