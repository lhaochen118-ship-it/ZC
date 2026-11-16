import json
exec(open('/home/user/ZC/Process/tools/v2/fake_all.py').read(), globals())
_all = responder
def responder(conv):
    u = conv[-1]["content"]
    if "câu hỏi KHÁC NHAU mà chủ cửa hàng này" in u:
        return json.dumps({"questions": [
            {"question": "hom nay co bao nhieu don", "program": "ANSWER = {\"type\": \"number\", \"answer\": len(C['orders'])}"},
            {"question": "Tổng số dòng hàng?", "program": "ANSWER = {\"type\": \"number\", \"answer\": len(C['lines'])}"},
            {"question": "loi", "program": "ANSWER = {\"type\": \"number\", \"answer\": 1/0}"}]}, ensure_ascii=False)
    return _all(conv)
