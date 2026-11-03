# STATUS — trạng thái làm việc hiện tại (cập nhật liên tục; đọc file này đầu tiên khi tiếp tục)

*Cập nhật: 2026-11-03 16:00 UTC (khung chat mới). **Đang dừng theo yêu cầu user: chờ user chấm v21 / v22 trên portal
và gửi điểm rồi mới chạy tiếp.** Token: 11 acc ở `~/.kaggle/tok_<user>` (gồm huyenco, mucnokieu; Duylan = duylnng,
HaiDinh = hai1108, NguyenManh = nguyenmanh0404).*

## Mới (khung 03/11 chiều)

| File nộp | Train (1 lần) | Public 3 lần chạy (nhất trí / đa số / tách) | Abstain | Đáp án 0 | So với v19 (giống / thêm trả lời / thêm abstain / đổi) |
|---|---|---|---|---|---|
| `submissions/v21/submission.jsonl` | 0.984 (holdout 0.971) | 856 / 72 / 72 | 70 | 11 | 927 / 37 / 22 / 14 |
| `submissions/v22/submission.jsonl` | 0.988 (holdout 0.969) | 856 / 68 / 76 | 76 | 3 | 929 / 27 / 18 / 26 |

- Cả hai: bỏ phiếu biết ánh xạ (map_penalties + vote.py); dev_003 / dev_009 giống hệt v19.
- Sẵn sàng nhưng **chưa chạy** (chờ user): `automation/jobs/v22b.jobs` (public seed 3/4/5, preflight đã OK).
- `automation/jobs/pool_study.py`: nghiên cứu chọn ánh xạ trên ứng viên gộp của nhiều lần chạy. Trên v19 public (code
  hiện tại chọn lại): 7/10 DN ánh xạ LLM cả 3 lần chạy cùng cụm đáp án; lựa chọn gộp theo xếp hạng hiện tại có lúc rơi
  vào cụm 1 ứng viên (dev_008, dev_014) → gộp thẳng chưa chắc ổn định hơn; cần xếp hạng theo kích thước cụm. Đang chạy
  bản train có nhãn (v13/v14/v18/v19/v22, `scratchpad/pool_train.log`).

---

## Điểm portal (user nộp) — EXPERIMENT_LOG entry 13

| File nộp | PA | Đúng | Sai | Abstain |
|---|---|---|---|---|
| `submissions/v13/submission.jsonl` | 0.637 | 785 | 148 | 67 |
| `submissions/v15/submission.jsonl` | 0.634 | 792 | 158 | 50 |
| `submissions/v18/submission.jsonl` | 0.669 | 800 | 131 | 69 |
| **`submissions/v19/submission.jsonl`** | **0.687** | 801 | 114 | 85 |
| `submissions/v20/submission.jsonl` | 0.650 | ~767 | ~117 | 117 |

- Số câu đúng ~800 từ v15; các bản trả lời gần như giống nhau → ~114 câu sai là **lỗi chung** (luật đọc câu, ánh xạ,
  quy ước README / từ địa phương), bỏ phiếu thêm lần chạy không gỡ được.
- v20 thua v19 vì **may rủi ánh xạ**: 3 lần chạy chọn 3 ánh xạ khác nhau (dev_009 đọc doanh thu 3 kiểu, dev_003) → bỏ
  phiếu tách → 52 câu abstain mà v19 trả lời đúng. Điểm public dao động ±0.03 chỉ vì may rủi này.
- Train không còn dự báo public (v20 train 0.991, holdout 0.977).

## Kernel đang chạy trên Kaggle (khởi chạy trong khung cũ; kết quả phải TẢI LẠI ở khung mới)

| Bản | Code | Kernel (acc/slug) | Job |
|---|---|---|---|
| v21 | 8dca4a2 | huyenco/zc-v21-1, huyenco/zc-v21-2, mucnokieu/zc-v21-3, mucnokieu/zc-v21-4 | train, pub (seed 0), pub_s1, pub_s2 |
| v22 | 20cfec7 | thnliu/zc-v22-1, thnliu/zc-v22-2, duylnng/zc-v22-3, duylnng/zc-v22-4 | train, pub (seed 0), pub_s1, pub_s2 |

- v21 khởi chạy 12:52 (xong ~15:00 UTC), v22 14:02–14:08 (xong ~16:15). Một lần chạy public mất 1.9–2.3 giờ.
- v21 = luật đọc câu tổng quát + kiểm tra ánh xạ (doanh thu đơn ≠ tổng dòng, order_time không có giờ, tên hàng viết tắt,
  tên thợ gọi tắt, cột doanh thu theo mô tả README) + dòng máy …GB + một người nhiều cách viết.
- v22 = v21 + từ địa phương cho tên hàng ('cá tràu' = 'cá lóc', 'đậu phụ rán' = 'đậu hũ chiên', 'cốc sạc' = 'củ sạc',
  'găng tay tất chân' = 'bao tay bao chân'…) + size theo cột danh mục + cảnh báo đơn ghi nợ không nợ gì + `money_in_text`.
- Acc: caotienminh1 **hết quota GPU tuần** (30 h). Còn: duylnng, huyenco, mucnokieu, thnliu (đang chạy), linhmai24,
  phuongnehe, hai1108, nguynduylm, lelynhan, mxuanquy, nguyenmanh0404 (quota chưa rõ; một bản = 4 kernel × ~2 h ≈ 8 h GPU).

## Việc tiếp theo (thứ tự)

1. **Tải kết quả v21 / v22** (mỗi kernel): `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels status <acc>/<slug>`;
   xong thì `KAGGLE_API_TOKEN=... kaggle kernels output <acc>/<slug> -p runs/vNN/<slug>_out` (thư mục chứa
   `results/{train,pub,pub_s1,pub_s2}/`).
2. Chấm train: `python3 src/report.py data/raw/train/train runs/vNN/zc-vNN-1_out/results/train`.
3. **File nộp public với bỏ phiếu biết ánh xạ** (7fa70f2):
   - ingest public: `python3 src/ingest.py data/raw/public_test/public_test <cache_pub> 4`;
   - `python3 automation/map_penalties.py <cache_pub> <run2>/results/pub <run3>/results/pub_s1 <run4>/results/pub_s2`
     (ghi `map_penalties.json`; v21 / v22 chưa ghi `chosen_penalty` trong compile log);
   - `python3 src/vote.py <tmp>/submission.jsonl <ba thư mục kết quả public>` →
     `python3 automation/make_submission.py <tmp> data/raw/public_test/public_test tmp_vNN` →
     `submissions/vNN/submission.jsonl` + README; **push ngay**, gửi user.
   - Thử offline trên v20: bỏ phiếu biết ánh xạ gỡ 15 câu tách (9 câu dev_009 về đúng đáp án v19).
4. Hướng tiếp (EXPERIMENT_LOG entry 13):
   - ánh xạ không phụ thuộc may rủi: **gộp ứng viên ánh xạ giữa các lần chạy** (mọi lần chạy chọn ánh xạ tốt nhất trong
     toàn bộ ứng viên), thêm kiểm tra nội dung bắt ánh xạ sai;
   - lỗi chung đã biết chưa sửa: nợ chỉ ghi trong ghi chú tự do (dev_008; v22 có cảnh báo + `money_in_text`), định nghĩa
     ngành điện thoại (dòng máy không gồm phụ kiện), câu trả lời bằng 0 = tín hiệu lọc sai (tìm thêm từ địa phương).
5. Docker cuối: `docker/predict.sh` chạy tối đa 3 lần + `vote.py` (đã đọc `chosen_penalty` từ compile log).
