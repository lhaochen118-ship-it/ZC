import json
exec(open('/tmp/claude-0/-home-user-ZC/53d4fe9e-6712-52f3-bd72-e042dbdf5ede/scratchpad/fake/fake_solver.py').read(), globals())
_solver = responder
def responder(conv):
    u = conv[-1]["content"]
    if "# CANDIDATES" in u: return "<plan>x | y | z</plan>\n<choice>1</choice>"
    if "# RULEBOOK ROWS (số thứ tự" in u: return '{"ids": [1]}'
    return _solver(conv)
