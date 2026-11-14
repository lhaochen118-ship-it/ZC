def responder(conv):
    u = conv[-1]["content"]
    n = u.count("\n[")  # candidates
    return "<plan>revenue | month | -</plan>\n<choice>%d</choice>" % (1 if "[2]" not in u else 2)
