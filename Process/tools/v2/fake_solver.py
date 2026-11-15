import re
def responder(conv):
    sysmsg, user = conv[0]["content"], conv[-1]["content"]
    if sysmsg.startswith("You are DataMate's data engineer"):
        return "```python\nraise ValueError('fake mapping')\n```"
    return "```python\nANSWER = {'type': 'number', 'answer': 0}\n```"
