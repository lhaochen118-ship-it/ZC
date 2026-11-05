# automation/jobs/analysis — công cụ so sánh / kiểm thô (từ khung chat 04/11)

Chạy từ gốc repo. `S` = thư mục scratch có `cache_pub/` (ingest public) và `cons_*/canon/` (ánh xạ dựng lại bởi consensus).

| Script | Dùng |
|---|---|
| `cmp.py <a.jsonl> <b.jsonl>` | đếm giống / đổi / thêm trả lời / thêm abstain, theo DN |
| `diff_detail.py <a.jsonl> <b.jsonl> [dev_0XX]` | in từng câu khác nhau (câu hỏi + hai đáp án) |
| `warnprobe.py <cache_pub> <biz> <canon_dir>` | chạy `canon.secondary_checks` trên mọi ứng viên đã dựng → điểm phạt, cảnh báo nặng |
| `revprobe.py <biz> <canon_dir>` | doanh thu vài tháng + trạng thái đơn của từng ứng viên |
| `debtprobe.py <biz> <ngày> <canon_dir>` | tổng nợ / số đơn nợ của từng ứng viên |
| `phone_direct.py <data_dir> "<code>"` | khung tự tính sổ bán điện thoại từ file thô (cẩn thận: ffill STT — xem PDF 4 §3.3) |
| `dev014_direct.py <S>` | tự tính các câu điện thoại dev_014 từ bảng ingest (ffill, bỏ trùng, bỏ đơn trả) |

Đáp án tự tính chỉ để **đánh giá** (dev set `goldkey/devset.py`), không đưa vào `src/`.
