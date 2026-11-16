# POS export (invoice list + invoice lines, KiotViet-like column names), any file / sheet names. Optional parts: payments
# (one row per payment), customer list (groups), catalogue (groups, cost), unit list, stock report, purchases, lots, drugs, tables
h = tbl("Mã hóa đơn", "Thời gian", "Trạng thái", "Khách hàng")
d = tbl("Mã hóa đơn", "Mã hàng", "Số lượng", "Thành tiền")
orders = frame(h, order_id="Mã hóa đơn", order_time="Thời gian tạo", order_date="Thời gian", customer_id="Mã KH", customer_name="Khách hàng",
               channel="Kênh bán", staff="Người bán", payment_method="Phương thức thanh toán", status=map_status(h["Trạng thái"], {"đã lấy": "hoàn thành"}),
               order_discount="Giảm giá", total_payable="Tổng sau giảm giá" if "Tổng sau giảm giá" in h.columns else h["Tổng tiền hàng"] - h["Giảm giá"].fillna(0),
               paid_total="Khách đã trả", debt_end="Còn nợ")
try:
    k = tbl("Mã khách hàng", "Tên khách hàng", "Nhóm khách hàng")
    orders["customer_group"] = orders["customer_id"].map(dict(zip(k["Mã khách hàng"], k["Nhóm khách hàng"])))
except KeyError: pass
items = None
try:
    sp = tbl("Mã hàng", "Tên hàng", "Nhóm hàng(3 Cấp)")
    items = frame(sp, item_code="Mã hàng", item_group="Nhóm hàng(3 Cấp)", base_unit="ĐVT", cost="Giá vốn", price="Giá bán")
    th = tbl("Mã hàng", "Hoạt chất")
    items["active_ingredient"] = items["item_code"].map(dict(zip(th["Mã hàng"], th["Hoạt chất"])))
    items["prescription"] = items["item_code"].map(dict(zip(th["Mã hàng"], th[[c for c in th.columns if "kê đơn" in c.lower()][0]].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0", "có", "x")))))
except (KeyError, IndexError): pass
lines = frame(d, order_id="Mã hóa đơn", line_no="STT" if "STT" in d.columns else d.groupby("Mã hóa đơn").cumcount() + 1, item_code="Mã hàng",
              item_name="Tên hàng", unit="Đơn vị tính", qty="Số lượng", qty_base="Số lượng quy đổi", unit_price="Đơn giá", line_discount="Giảm giá",
              line_amount="Thành tiền", unit_cost="Giá vốn")
orders, lines = finalize(lines, orders, items=items)
if lines["item_name"].map(lambda x: key(x).startswith("phi phuc vu")).any(): lines["is_service"] = lines["item_name"].map(lambda x: key(x).startswith("phi phuc vu"))
C = {"orders": orders, "lines": lines}
if items is not None: C["items"] = items
try:
    q = tbl("Mã phiếu", "Mã hóa đơn", "Giá trị")
    C["payments"] = frame(q, order_id="Mã hóa đơn", pay_time="Thời gian", pay_date=q["Thời gian"].dt.normalize(), amount="Giá trị", method="Phương thức thanh toán")
except KeyError: pass
try: C["conversions"] = conversions_table(tbl("Mã hàng", "ĐVT", "Quy đổi"), "Mã hàng", "ĐVT", "Quy đổi")
except KeyError: C["conversions"] = readme_conversions(README)
try:
    ph, pdt = tbl("Mã nhập hàng", "Thời gian", "Mã NCC"), tbl("Mã nhập hàng", "Mã hàng", "Số lượng", "Đơn giá")
    pu = pdt.merge(ph.drop(columns=[c for c in ph.columns if c in pdt.columns and c != "Mã nhập hàng"]), on="Mã nhập hàng", how="left")
    C["purchases"] = frame(pu, receipt_id="Mã nhập hàng", receipt_date=pu["Thời gian"].dt.normalize(), supplier="Mã NCC", item_code="Mã hàng", unit="ĐVT",
                           qty="Số lượng", unit_cost="Đơn giá", amount=pu["Số lượng"] * pu["Đơn giá"], lot_id="Lô", expiry="Hạn sử dụng",
                           cancelled=pu["Trạng thái"].map(lambda x: canon_status(x) == "huỷ") if "Trạng thái" in pu.columns else False, note="Ghi chú")
except KeyError: pass
try:
    st = tbl("Mã hàng", "Tồn cuối kỳ")
    C["stock"] = frame(st, item_code="Mã hàng", item_name="Tên hàng", qty_base="Tồn cuối kỳ")
except KeyError: pass
try:
    lo, sl = tbl("Lô", "Hạn sử dụng", "Số lượng (ĐVT cơ bản)", "Mã nhập hàng"), tbl("Mã hóa đơn", "Lô", "Số lượng trả (ĐVT cơ bản)")
    C["lots"] = frame(lo, lot_id="Lô", receipt_id="Mã nhập hàng", item_code="Mã hàng", expiry="Hạn sử dụng", qty_in_base="Số lượng (ĐVT cơ bản)")
    C["lot_sales"] = frame(sl, order_id="Mã hóa đơn", line_no="STT", lot_id="Lô", qty_base="Số lượng (ĐVT cơ bản)", returned_base="Số lượng trả (ĐVT cơ bản)")
except KeyError: pass
try:
    pb = tbl("Mã hóa đơn", "Phòng/Bàn")
    C["orders"] = orders.merge(frame(pb, order_id=pb["Mã hóa đơn"].astype(str), table="Phòng/Bàn", area="Khu vực", guests="Số khách"), on="order_id", how="left")
except KeyError: pass
NOTES = "POS export. " + ("Debt only known at the end of the data (debt_end). " if "payments" not in C else "") + \
        ("No cost price on lines. " if "unit_cost" not in lines.columns else "") + ("No customer groups. " if "customer_group" not in orders.columns else "")
