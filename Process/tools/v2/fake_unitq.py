import json, re
def responder(conv):
    u = conv[-1]["content"]
    m = re.search(r"# HÀM CỦA CỬA HÀNG\n([A-Za-z_0-9]+)\(", u)
    n = m.group(1) if m else "f"
    return json.dumps({"questions": [{"question": "thang 3/2025 the nao", "program": "ANSWER = {\"type\": \"number\", \"answer\": %s(month(2025, 3))}" % n}]}, ensure_ascii=False)
