import json, re
def responder(conv):
    u = conv[-1]["content"]
    if "# CHƯƠNG TRÌNH GỐC" in u:
        q = re.search(r"# CÂU GỐC[^\n]*\n(.*?)\n\n", u, re.S).group(1)
        p = re.search(r"# CHƯƠNG TRÌNH GỐC \(đúng\)\n(.*?)\n\n# NHIỆM VỤ", u, re.S).group(1)
        if "CÁCH VIẾT" in u: return json.dumps({"question": q + " z", "program": p}, ensure_ascii=False)
        return json.dumps({"question": q + " (khác)", "program": p.replace("TODAY", "TODAY - pd.Timedelta(days=1)")}, ensure_ascii=False)
    if "# CHƯƠNG TRÌNH 1" in u: return "<think>ok</think><choice>1</choice>"
    return "{}"
