# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-03 10:15 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
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

## Khung 6 giờ thứ hai (04:55 → 11:00 UTC 03/11) — routine `trig_01U26KaZP68Dmg8JL2RGcbVS` bật, tắt sau 11:00

- **Điểm portal (user): v13 vote = 0.637** (train ước lượng ~0.97). Lỗi chủ yếu ở DN ngành MỚI của rulebook public
  (sửa xe dev_012, điện thoại dev_011 / dev_014, mẹ và bé dev_013: các lần chạy chỉ trùng 12–75 %) + dòng tổng (dev_009).
- Train: v14 0.966 (holdout 0.951), **v15 0.982 (dev 0.999)**, v16 0.970 (biz_124 xui ánh xạ).
- Code sau v15 (chưa có điểm public): ce0c688 unanswerable → abstain (ánh xạ LLM bỏ sót bảng), 234e2f8 đợt ánh xạ 2 khi
  chỉ 1 ứng viên, 084407b + a0e5c08 dòng tổng (có dấu), **a0e5c08 mục ngành mới** (luật ngành cạnh README, luật qw 1,
  2 mẫu bảng thô), 4b85565 `--kv-dtype`, **a246b0b bảng thô ánh xạ bỏ sót đưa vào prompt** (dev_014: góp / thu máy cũ /
  bảo hành từng bị trả unanswerable).
- Kết quả (10:15):
  - train v18 0.980 (15 sai), **v19 0.986** (dev 0.999, holdout 0.961) — tốt nhất;
  - v17 tốc độ: KV fp8 không chạy với Qwen3.5 / vLLM 0.32; MTP chỉ −6 % pha ánh xạ → giữ mặc định;
  - file nộp đã push: `submissions/v15/` (50 abstain), `submissions/v18/` (69 abstain, các DN bất ổn ổn định hơn nhiều);
    `submissions/v19/` khi 3 lần chạy public xong (~10:30).
- Đĩa scratch từng đầy lúc 08:24 (replay cũ ~4 GB) → đã xoá; giữ `runs/`, `cache_*`.

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
