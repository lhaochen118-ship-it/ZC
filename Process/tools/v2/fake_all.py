import json
exec(open('/home/user/ZC/Process/tools/v2/fake_solver.py').read(), globals())
_solver = responder
def responder(conv):
    u = conv[-1]["content"]
    if "# CANDIDATES" in u: return "<plan>x | y | z</plan>\n<choice>1</choice>"
    if "# RULEBOOK ROWS (số thứ tự" in u: return '{"ids": [1]}'
    return _solver(conv)
