# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-03 04:45 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
(kể cả sau khi hết / reset token): đọc `CLAUDE.md` → file này → làm mục "Việc tiếp theo" theo thứ tự. Khung 12 giờ
của user kết thúc 03:30 UTC 03/11; sau mốc đó tắt routine `trig_01U26KaZP68Dmg8JL2RGcbVS` và báo kết quả.*

## Kết quả (train 3000 câu, LOO; EXPERIMENT_LOG entry 8–9)

| Hệ | dev | holdout | all |
|---|---|---|---|
| v6-1 (một lần chạy, tốt nhất) | 0.974 | 0.961 | 0.969 |
| v9-1 | 0.989 | 0.809 | 0.928 |
| **bỏ phiếu 4 lần chạy v6–v9** (`src/vote.py`) | | | **0.980** (2960 / 19 / 21) |
| bỏ phiếu 3 lần chạy (mọi bộ ba) | | | 0.976 – 0.981 |

- Oracle luật trên ánh xạ tham chiếu: **2791 đúng / 1 sai / 208 chưa đọc** (trước 2754 / 3 / 243).
- File nộp public đã có: `submissions/public_vote4_v7-v9.jsonl` (bỏ phiếu v7-2, v8-2, v9-2, v9-3; 60 abstain; đã kiểm
  định dạng). Sẽ thay bằng bản bỏ phiếu các lần chạy mới nhất (v11) khi xong.

## Code mới từ 00:00 (đã commit, chưa có kết quả GPU)

- fefaec6: bỏ phiếu nhiều lần chạy — `src/vote.py`, `--seed-base`, `docker/predict.sh` chạy tối đa 3 lần trong ngân sách.
- 76b676e: luật `unit_qty` ("bán được bao nhiêu ly"), `kg` không còn bị đọc thành "không", lọc mặt hàng theo mọi mã / cách
  viết, `frame()` nhận hàm, sổ bán không mã đơn, gợi ý KeyError, tên bảng làm biến; StepLLM: thinking không greedy
  (temp 0.6, top-k 20) + budget forcing (`--think-budget 12288`).
- 327acf2: bỏ phiếu chỉ tính mẫu LLM "bám câu hỏi" (tên hàng trong chương trình phải có trong câu); cảnh báo cột bị
  `frame()` bỏ qua (chỉ ánh xạ LLM).

## Kết quả cuối khung 12 giờ (04:45 UTC 03/11) — không còn kernel nào chạy

- **v11 train (code e73a151): PA 0.975** (2950 / 25 / 25; dev 0.990, holdout 0.946), 4172 s — tốt nhất một lần chạy.
  Bỏ phiếu 6 lần chạy train v6–v11: 0.982.
- v12 (`--map-repairs 4 --map-repair-think 0`): train 0.941 (biz_123 ánh xạ sai), public subset ánh xạ thành công 64 %
  (v11s 52 %) → chưa chứng minh, giữ mặc định.
- Sau v11, chưa chạy GPU: 69c0d8d mẫu câu lô sắp hết hạn / hết hàng (oracle 2798 / 1 / 201).
- **File nộp public** (đã kiểm định dạng; user tự nộp portal):
  1. `submissions/public_v10-v11_vote5.jsonl` — bỏ phiếu 5 lần chạy (v10 ×2, v11 ×3), 34 abstain (**đề xuất nộp trước**).
  2. `submissions/public_v11_vote3.jsonl` — bỏ phiếu 3 lần chạy v11, 103 abstain (đúng thiết kế bản Docker 3 lần chạy).
- Routine `trig_01U26KaZP68Dmg8JL2RGcbVS` đã tắt (hết khung 12 giờ).

## Việc tiếp theo (thứ tự)

1. User nộp vote5 (và nếu còn lượt, vote3) lên portal → so điểm thật với ước lượng train.
2. Chạy GPU code mới nhất (69c0d8d) train + public ×3 → file nộp mới.
3. Ổn định ánh xạ DN phức tạp (biz_123 / sqlite nhiều bảng, dev_012 sổ sửa xe): đó là nguồn sai và lệch giữa các lần chạy
   lớn nhất; hướng: chọn ánh xạ theo đồng thuận đáp án giữa các ứng viên, kiểm tra nội dung bảng nhập / lô.
4. Docker: build image trên máy có GPU, đo thời gian 3 lần chạy trên 2×RTX 5090 (≤ 6 h).
