import json
exec(open('/tmp/claude-0/-home-user-ZC/67cec8b5-c044-570d-a1c2-4ef17349b57a/scratchpad/fake_solver.py').read(), globals())
_solver = responder
def responder(conv):
    u = conv[-1]["content"]
    if "# CANDIDATES" in u: return "<plan>x | y | z</plan>\n<choice>1</choice>"
    if "# RULEBOOK ROWS (số thứ tự" in u: return '{"ids": [1]}'
    return _solver(conv)
