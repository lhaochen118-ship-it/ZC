# online-shop export: one row per order line (order cells may be blank on later lines); unit conversion table
d, q = T["DonHang_ChiTiet"], T["DanhSachQuyDoi"]
lines = frame(d, order_id="Mã đơn hàng", order_date="Ngày đặt hàng", status=map_status(d["Trạng thái đơn hàng"]), customer_name="Tên khách hàng",
              item_name="Tên sản phẩm", unit="Đơn vị tính", qty="Số lượng sản phẩm", unit_price="Giá sản phẩm", line_amount="Tổng tiền")
cv = conversions_table(q, "Tên sản phẩm", "Đơn vị quy đổi", "Tỉ lệ quy đổi", "Đơn vị cơ bản")
orders, lines = finalize(lines, conversions=cv)
C = {"orders": orders, "lines": lines, "conversions": cv}
NOTES = "No item codes, no customer codes (customers by name), no time of day, no payments, purchases or stock."
