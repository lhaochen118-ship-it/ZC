# hand-kept sales book without order ids or customers: every row is one sale record (no orders to count); the sale date
# is an Excel serial number; base quantity in "Số lượng lẻ"
t = T["SoBanHang"]
lines = frame(t, order_id=pd.Series(range(len(t)), index=t.index).astype(str), order_date=t["ngay_ban"].map(parse_date),
              status=map_status(t["Trạng thái xử lý"]), item_name="Nội dung", unit="Đơn vị (tính)", qty="Số lượng", qty_base="Số lượng lẻ",
              unit_price="don_gia", line_amount="Tổng tiền")
orders, lines = finalize(lines, dedupe=False)
C = {"orders": orders, "lines": lines, "conversions": readme_conversions(README)}
NOTES = "No order ids: each row is one record (order counts, customers and order values are not recorded). No codes, purchases, stock or debt."
