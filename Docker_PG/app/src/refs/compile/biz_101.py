# flat sales book: one row per order line; status in "Kết quả"; base quantity already in "Quy ra ĐV lẻ"
t = T["ban_hang"]
lines = pd.DataFrame({
    "order_id": t["Mã hóa đơn"], "order_date": t["Ngày tháng"], "status": map_status(t["Kết quả"]),
    "customer_name": t["Đối tác"], "item_name": t["Nội dung"], "unit": t["Đơn vị (tính)"], "qty": t["so_luong"],
    "qty_base": t["Quy ra ĐV lẻ"], "unit_price": t["Giá bán"], "line_amount": t["Thanh tien"]})
orders, lines = finalize(lines)
C = {"orders": orders, "lines": lines}
NOTES = "No item codes, no customer codes (customers by name), no time of day, no purchases, stock, costs or debt."
