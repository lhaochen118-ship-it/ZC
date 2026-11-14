import re
def responder(conv):
    u = conv[-1]["content"]
    q = u.split("# QUESTION")[1].split("\n")[1].lower()
    ids = []
    for m in re.finditer(r"^(\d+)\. ([^:]+):", u, re.M):
        if m.group(2).lower().split(" nhất")[0][-10:] in q: ids.append(int(m.group(1)))
    return '{"ids": %s}' % ids[:3]
