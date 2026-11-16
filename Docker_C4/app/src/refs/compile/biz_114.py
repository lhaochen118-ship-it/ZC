# flat sales book: one row per order line; base quantity already in "Quy ra ĐV lẻ"; conversions in the README
t = T["ban_hang"]
lines = frame(t, order_id="Mã GD", order_date="Ngày", status=map_status(t["trang_thai"], {"đã lấy": "hoàn thành"}), customer_name="Tên khách",
              item_name="Sản phẩm", unit="Đơn vị tính", qty="Sluong", qty_base="Quy ra ĐV lẻ", unit_price="Giá bán lẻ", line_amount="Tổng")
orders, lines = finalize(lines)
C = {"orders": orders, "lines": lines, "conversions": readme_conversions(README)}
NOTES = "Status 'đã lấy' = completed. No item / customer codes, no time of day, no purchases, stock, costs or debt."
