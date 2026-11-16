def responder(conv):
    u = conv[-1]["content"]
    if "chứng chỉ" in u: return '{"question_unit": "vnd", "variables": {"r": "vnd"}, "expression": "r", "result_unit": "vnd"}'
    if '"choice"' in u: return '{"choice": 1, "reason": "x"}'
    return '{"sql": "SELECT SUM(revenue) AS number FROM orders"}'
