# pharmacy, three sources: sales software (BanHang_*: invoices, lines with cost at sale, payments, lots sold, lots, products,
# unit conversions, customers with tier, drug info, price history), notebook (SoTay_*: purchases with lots, expenses), stock
h, d, k, sp = T["BanHang_HoaDon"], T["BanHang_ChiTietHoaDon"], T["BanHang_KhachHang"], T["BanHang_HangHoa"]
orders = frame(h, order_id="Order ID", order_date="Ngày bán", order_time="Giờ tạo đơn", customer_id="Mã KH mua", customer_name="Tên KH",
               staff="Mã NV bán", channel="Nguồn đơn", payment_method="Phương thức TT", status=map_status(h["Trạng thái"]),
               order_discount="CK", total_payable="Tong tien", debt_end="Còn nợ")
orders["customer_group"] = orders["customer_id"].map(dict(zip(k["Customer ID"], k["Hạng"])))
items = frame(sp, item_code="ma_sp", item_name="Món", item_group="Phân loại", base_unit="Đơn vị", cost="Giá vốn hiện tại", price="Giá")
th = T["BanHang_Thuoc"]
items["active_ingredient"] = items["item_code"].map(dict(zip(th["Mã HH"], th["Hoạt chất"])))
items["prescription"] = items["item_code"].map(dict(zip(th["Mã HH"], th["Kê đơn"].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0", "có", "x")))))
lines = frame(d, order_id="Mã giao dịch", line_no="TT dòng", item_code="Ma hang", item_name="Chi tiết hàng", unit="Đ.vị", qty="Slg",
              qty_base="SL cơ bản", unit_price="Giá niêm yết", line_discount="Khuyến mãi", line_amount="Thanh tien", unit_cost="Giá vốn lúc bán")
orders, lines = finalize(lines, orders, items=items)
q = T["BanHang_PhieuThu"]
payments = frame(q, order_id="Mã bill", pay_time="Thời gian thu", amount="Số tiền thu", method="PTTT")
payments["pay_date"] = payments["pay_time"].dt.normalize()
n = T["SoTay_NhapHang"]
purchases = frame(n, receipt_id="Số phiếu", receipt_date="Ngày nhập", supplier="Nhà cung cấp", item_name="Product", unit="ĐVT nhập",
                  qty="SL nhập", unit_cost="Giá nhập", amount="Thành tiền", lot_id="Lô", expiry="HSD", note="Mô tả")
purchases["cancelled"] = False
lo = T["BanHang_LoHang"]
lots = frame(lo, lot_id="Lô", receipt_id="Mã phiếu nhập", item_code="Mã món", expiry="HSD", qty_in_base="SL nhập (ĐVT cơ bản)")
sl = T["BanHang_BanTheoLo"]
lot_sales = frame(sl, order_id="Mã hóa đơn", line_no="No.", lot_id="Lô", qty_base="SL quy ra lẻ", returned_base="SL trả (ĐVT cơ bản)")
e = T["SoTay_ChiPhi"]
expenses = frame(e, date="Ngày chi", category="Mục", amount="So tien", note="Ghi chú")
st = T["ton_kho"]
stock = frame(st, item_code="ID SP", item_name="Hàng hóa", base_unit="Đơn vị", qty_base="Ton kho")
prices = frame(T["BanHang_LichSuGia"], date="Ngày áp dụng", item_code="Mã sản phẩm", item_name="Sản phẩm", price="Giá mới", old_price="Giá cũ")
cv = conversions_table(T["BanHang_DonViQuyDoi"], "Mã sp", "Đơn vị", "Bằng")
C = {"orders": orders, "lines": lines, "items": items, "payments": payments, "purchases": purchases, "lots": lots, "lot_sales": lot_sales,
     "expenses": expenses, "stock": stock, "prices": prices, "conversions": cv}
NOTES = "Status 'Không giao được' = cancelled, 'Đổi trả' = returned. Customer tier = customer_group. Lines carry the cost at sale (per sold unit)."
