import json, re, sys
sys.path.insert(0, "/home/user/ZC/src"); import score
ref = score.load("/home/user/ZC/data/raw/public_gt.jsonl")
P = r"(?i)(?<!\w)th?\s?\d{1,2}\s?[/-]\s?\d{2,4}|(?<![\w/])\d{1,2}/\d{2}(?![\d/])|(?i)(?<!\w)th\.?\s(này|nay|trước|truoc)"
P = re.compile(r"(?i)(?<!\w)th?\s?\d{1,2}\s?[/-]\s?\d{2,4}|(?<![\w/])\d{1,2}/\d{2}(?![\d/])|(?<!\w)th\.?\s(này|nay|trước|truoc)")
tot = {"with": [0, 0], "without": [0, 0]}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["id"] not in ref: continue
    k = "with" if P.search(r["question"]) else "without"
    v = score.judge(r.get("final"), ref[r["id"]])
    tot[k][0] += 1; tot[k][1] += (v == "correct") - (v == "wrong")
for k, (n, s) in tot.items(): print(f"{k:8s} T-notation: {n:4d} questions, PA {s / max(n, 1):.3f}")
