# P5 plan P02/P03: fake model that RECORDS the framework's prompts (calibration data for the expert saliency, training shops only).
# Every conversation is appended to $ZC_RECORD.<pid> (one file per process: no interleaving); the answer is fake_all2's.
import os, json, random
_g = {}; exec(open("/home/user/ZC/Process/tools/v2/fake_all2.py").read(), _g)
_rng = random.Random(os.getpid())
def responder(conv):
    p = os.environ.get("ZC_RECORD")
    if p and _rng.random() < float(os.environ.get("ZC_RECORD_P", "1")):
        with open(f"{p}.{os.getpid()}", "a", encoding="utf-8") as f: f.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
    return _g["responder"](conv)
