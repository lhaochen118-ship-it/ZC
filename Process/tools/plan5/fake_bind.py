import json, random
def responder(conv):
    u = conv[-1]["content"]
    h = "avg_ticket(p)" if "TOOLKIT NGÀNH (Sửa xe)" in u else ("revenue(p)" if random.random() < 0.7 else "bad_fn(p)")
    return json.dumps({"triggers": ["khach quen", "xe quen", "doanh thu"], "content": "định nghĩa thử của thuật ngữ này", "code_hint": h}, ensure_ascii=False)
