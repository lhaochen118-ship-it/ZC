# File nộp public test — hệ v18

Code: commit `4b85565`. So với v15, bản này có thêm:
- mục luật ngành mới của rulebook (sửa xe, điện thoại, mẹ và bé): đưa lên cạnh README, giảm trọng số chương trình luật,
  thêm 2 chương trình trên bảng thô cho mỗi câu;
- bỏ các dòng tổng ("Tổng ngày …", "Cộng ngày …");
- trả abstain thay vì "unanswerable" khi ánh xạ LLM bỏ sót bảng thô;
- sinh thêm một đợt ánh xạ khi chỉ có 1 ứng viên.

`submission.jsonl` là kết quả bỏ phiếu của 3 lần chạy (seed 0 / 1 / 2). Đã kiểm định dạng: 1000 id, 0 lỗi, 69 câu abstain.

Train (một lần chạy): PA 0.980, chỉ 15 câu sai (dev 0.9985, holdout 0.944).

Độ đồng thuận giữa 3 lần chạy public tăng mạnh ở các doanh nghiệp từng bất ổn (v13 → v18):

| Doanh nghiệp | v13 | v18 |
|---|---|---|
| dev_009 | 57 % | 100 % |
| dev_008 | 62 % | 91 % |
| dev_012 | 12 % | 60 % |
| dev_013 | 75 % | 84 % |
| dev_014 | 25 % | 47 % |
