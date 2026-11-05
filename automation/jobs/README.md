# automation/jobs — job chuẩn + công cụ

- `v25.jobs`, `v28.jobs`: chuỗi job chuẩn (train + 5 public, seed khác nhau; jobs cách nhau `;`). Bản sau: copy, đổi seed / cờ.
- `prepreflight.py "<jobs>"`: chạy `automation/preflight.py`, ghi mark cache để `fanout.py` khởi chạy ngay.
- `mksub5.sh vNN [vCONS]` (env `S` = scratch có `runs/`, `cache_pub/`): bỏ phiếu 5 lần chạy public của vNN + (tuỳ chọn)
  consensus → `submissions/`.
- Phân tích log (không cần nhãn): `rule_vs_llm.py`, `agree_biz.py` (3 lần chạy), `alt_diff.py`, `scan_maps.py`, `two_maps.py`.
- `analysis/`: so sánh file nộp, chạy kiểm tra trên ứng viên ánh xạ, tự tính từ sổ thô (xem `analysis/README.md`).
