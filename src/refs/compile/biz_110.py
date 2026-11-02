# invoice list only (one row per invoice, no item lines): invoice value, paid, unpaid at the end of the data
h = T["danh_sach_hoa_don"]
orders = frame(h, order_id="Mã bill", order_date="Ngày giao dịch", customer_name="Bên mua", status=map_status(h["Status"]),
               revenue="Tien hang", total_payable="TongTien", paid_total="Da tra", debt_end="Chưa trả")
lines = frame(h, order_id="Mã bill", line_amount="Tien hang")
orders, lines = finalize(lines, orders)
C = {"orders": orders, "lines": lines}
NOTES = "No item lines (items, quantities and groups are not recorded); debt only known at the end of the data (debt_end)."
