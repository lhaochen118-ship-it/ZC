import json, re
def responder(conv):
    u = conv[-1]["content"]
    if "# NHIỆM VỤ\nViết" in u:
        return "<think>x</think>" + json.dumps([{"term": "Doanh thu ca sáng", "definition": "doanh thu các đơn đặt trong ca sáng của kỳ", "variant_of": None},
                                                 {"term": "Số đơn lớn", "definition": "số đơn có doanh thu từ 500000 đồng trở lên trong kỳ", "variant_of": "Số đơn"}], ensure_ascii=False)
    if "DÒNG RULEBOOK CẦN VIẾT HÀM" in u:
        return "<think>y</think>" + json.dumps({"plan": "dùng revenue", "name": "ham_thu", "signature": "ham_thu(p)", "doc": "doanh thu kỳ p",
                                                "triggers": ["doanh thu"], "code": ["def ham_thu(p):", "    return revenue(p)"]}, ensure_ascii=False)
    return "{}"
