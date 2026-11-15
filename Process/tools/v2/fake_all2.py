import os
_g = {}; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_all.py")).read(), _g)
_u = {}; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_unitq.py")).read(), _u)
def responder(conv):
    if "# HÀM CỦA CỬA HÀNG" in conv[-1]["content"]: return _u["responder"](conv)
    return _g["responder"](conv)
