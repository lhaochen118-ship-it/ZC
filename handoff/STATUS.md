# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-03 02:15 UTC. Branch `claude/datamate-v3-0dny90`. User cho phép tự chạy tiếp không cần hỏi
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

## Đang chạy (02:15 UTC)

- **v11** (`scratchpad/runs/v11`, launched 02:02, commit e73a151 = mọi sửa tới sheet alias): zc-v11-1 train, -2 public
  seed 0, -3 public seed 1, -4 public seed 2 (huyenco ×2, mucnokieu ×2). Dự kiến xong ~03:20–03:35.
- **v12** (preflight đang chạy; commit 9bb6191): `--map-repairs 4 --map-repair-think 0` — v12-1 train, v12-2 public
  `--subset 28` (so trực tiếp với v11s). Đẩy lên lelynhan.
- Đã xong: v10 (train 0.957; public 47 / 40 abstain; 2 lần public trùng 87.4 %), v11s (public subset: không lỗi, pha
  ánh xạ −25 %, tỉ lệ ánh xạ thành công không đổi 52 %).
- File nộp tạm: `submissions/public_vote6_v7-v10.jsonl` (bỏ phiếu 6 lần chạy public, 60 abstain, đã kiểm định dạng).
- Watcher: `automation/watch_done.py <scratchpad>/runs --max 115`; `runs/active_runs.txt` = v11 (+ v12 khi đẩy).

## Việc tiếp theo (thứ tự)

1. v11s xong và không lỗi (có đáp án, thời gian ánh xạ ngắn lại) → chạy **v11** = commit mới nhất, 4 kernel:
   `python3 automation/fanout.py $R/v11 zc-v11 "$(cat $S/v11.jobs)" 4 --accounts hai1108,huyenco --l4 --ks nhatlong1103/zc-assets,lelynhan/zc-m-qwen35-9b`
   (train, public seed 0 / 1 / 2); thêm `v11` vào `active_runs.txt`.
2. Chấm v10 train (`score.json`, `scratchpad/vote_errors.py`), so v6–v9; đo độ trùng public v10-2 / v10-3.
3. Khi v11 xong: chấm train; bỏ phiếu 3 lần chạy public v11 (`src/vote.py out.jsonl run1 run2 run3`) →
   `automation/make_submission.py <dir> data/raw/public_test/public_test public_v11_vote3` → commit file nộp.
4. Cập nhật EXPERIMENT_LOG (entry 10: v10, v11), file này; 03:30 UTC: tắt routine, báo user.
5. Sau đó: ghim pandas trong `docker/Dockerfile` theo bản Kaggle (log của run_kaggle); biz_123 / DN sqlite phức tạp
  (ánh xạ LLM hỏng 2/3 lần): thêm vòng sửa cho ánh xạ hoặc tách ánh xạ theo nhóm bảng.
