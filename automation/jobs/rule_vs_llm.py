import json, sys, glob, collections
runs = sys.argv[1:]
def canon(a):
    if not a: return None
    return json.dumps([a.get("type"), a.get("answer")], ensure_ascii=False, sort_keys=True)
st = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
for R in runs:
    for f in glob.glob(R + "/results/pub/shard*/logs.jsonl"):
        for l in open(f):
            r = json.loads(l); b = r["business_id"]; s = r.get("samples", {})
            rule = s.get("rule"); llm = [x for k, v in s.items() if k != "rule" and isinstance(v, list) for x in v if isinstance(x, dict) and x.get("ok") and not x.get("ungrounded")]
            st[b]["n"] += 1
            if not rule or not rule.get("ok"): st[b]["norule"] += 1; continue
            st[b]["rule"] += 1
            ra = canon(rule.get("answer")); la = [canon(x.get("answer")) for x in llm]
            if not la: st[b]["rule_only"] += 1; continue
            agree = sum(1 for a in la if a == ra)
            if agree == 0: st[b]["llm_all_disagree"] += 1; ex[b].append((r["question"], ra, collections.Counter(la).most_common(2), rule.get("tpl")))
            elif agree < len(la): st[b]["llm_some_disagree"] += 1
            else: st[b]["llm_agree"] += 1
print("biz n norule rule rule_only agree some_dis all_dis")
for b in sorted(st):
    s = st[b]; print(b, s["n"], s["norule"], s["rule"], s["rule_only"], s["llm_agree"], s["llm_some_disagree"], s["llm_all_disagree"])
json.dump(ex, open(sys.argv[0].replace(".py", "_ex.json"), "w"), ensure_ascii=False, indent=0)
