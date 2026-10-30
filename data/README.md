# data/

Đặt ở đây:

- `data.txt`: nội dung tab **Data** của portal DataMate (không đọc được trên web).
- Hai file zip dữ liệu của ban tổ chức (train / public test).

Khi xử lý: giải nén vào `data/raw/` (bị `.gitignore`), chỉ đọc cấu trúc và vài giá trị; rồi đăng bản giải nén lên Kaggle
thành dataset `nhatlong1103/zc-datamate-data` (xem `docs/02_KAGGLE_OPS.md` §4, hỏi user về quyền public trước).
