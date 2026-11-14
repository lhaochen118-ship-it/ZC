#!/usr/bin/env python3
"""Next part of a training checkpoint chain across Kaggle accounts (user 12/11: "nối checkpoint qua nhiều acc"; plan 3).
The finished part's output (pulled by the runner: <runs>/<part>/*_out/results/<NAME>/ckpt/checkpoint-N, or the final adapter)
-> a private dataset <next_acc>/<slug> laid out <NAME>/ckpt/checkpoint-N + info/ (two top folders: Kaggle keeps the layout)
-> wait until its files are live -> a plan for the next part (the resume jobs, --ds that dataset, ZC_SESSION_S sized to what the
account's quota allows at 2.5 quota-h per wall hour) -> automation/launch_many.py (which checks the quota again).
Exits 2 when the finished part already holds the final adapter (nothing to chain), 3 when the account cannot run a useful part.
Usage: chain_next.py <runs> <finished_part> <NAME> <next_acc> <next_part> <resume.jobs> [--train-data D] [--remaining-h H]
       [--extra-ks K1,K2] [--min-h 0.6] [--dry]"""
import os, sys, json, glob, shutil, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "automation"))
import quota

FACTOR, SETUP_H, MARGIN_S = 2.5, 0.3, 900


def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def kaggle(acc, *a, timeout=900):
    env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser(f"~/.kaggle/tok_{acc}")).read().strip())
    return subprocess.run(["kaggle", *a], capture_output=True, text=True, env=env, timeout=timeout)


def main():
    runs, part, name, acc, nxt, jobs = sys.argv[1:7]
    dry = "--dry" in sys.argv
    outs = glob.glob(os.path.join(runs, part, "*_out", "results", name))
    if not outs: sys.exit(f"no output {runs}/{part}/*_out/results/{name}")
    res = outs[0]
    if os.path.isfile(os.path.join(res, "adapter", "adapter_model.safetensors")) and not glob.glob(os.path.join(res, "ckpt", "checkpoint-*")):
        print(f"{part}: final adapter present, nothing to chain"); sys.exit(2)
    cks = sorted(glob.glob(os.path.join(res, "ckpt", "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks: sys.exit(f"{res}: no checkpoint")
    ck = cks[-1]; step = int(ck.rsplit("-", 1)[1])
    tm = {}
    try: tm = json.load(open(os.path.join(res, "train_metrics.json")))
    except Exception: pass
    left = quota.remaining(acc)
    wall = left / FACTOR - 0.05
    need = float(arg("--remaining-h", "0") or 0)
    sess_h = min(wall, need + SETUP_H + MARGIN_S / 3600 + 0.1) if need else wall
    sess_h = min(sess_h, 10.9)                                         # a full 30 h account gave 11.95 h > launch_many's 11 h cap:
                                                                        # the part was refused (wr30p4, 14/11 00:14)
    if sess_h < float(arg("--min-h", "0.6")): print(f"{acc}: {left:.2f} h quota -> {wall:.2f} h wall: too little"); sys.exit(3)
    slug = f"zc-{nxt}-ck"
    stage = os.path.join(os.path.dirname(runs), "ds_" + nxt)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, name, "ckpt"), exist_ok=True); os.makedirs(os.path.join(stage, "info"), exist_ok=True)
    shutil.copytree(ck, os.path.join(stage, name, "ckpt", os.path.basename(ck)))
    open(os.path.join(stage, "info", "README.txt"), "w").write(f"checkpoint {os.path.basename(ck)} of {name} from {part}\n")
    json.dump({"title": slug, "id": f"{acc}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(stage, "dataset-metadata.json"), "w"))
    print(f"{part} -> {acc}/{slug}: {os.path.basename(ck)} (step {step}; stopped_early {tm.get('stopped_early')}); session {sess_h:.2f} h of {left:.2f} h quota", flush=True)
    if dry: return
    r = kaggle(acc, "datasets", "create", "-p", stage, "-r", "zip", "-q")
    print("dataset:", (r.stdout + r.stderr).strip()[-300:], flush=True)
    t0 = time.time()
    while time.time() - t0 < 1500:
        r = kaggle(acc, "datasets", "files", f"{acc}/{slug}", timeout=120)
        if "adapter_model.safetensors" in r.stdout and f"{name}/ckpt" in r.stdout.replace("\\", "/"): break
        time.sleep(30)
    else: sys.exit("dataset files not live after 25 min")
    plan = [{"acc": acc, "name": nxt, "jobs": jobs, "est_h": round(sess_h, 2),
             "args": ["--ms", "danbth/qwen3-5-9b/transformers/default/1", "--ks", ",".join(["nhatlong1103/zc-assets", "yeshenhi/zc-train-wheels"] +
                      [k for k in (arg("--extra-ks", "") or "").split(",") if k]),
                      "--ds", f"{acc}/{slug}", "--env", f"ZC_SESSION_S={int(sess_h * 3600)},ZC_SESSION_MARGIN={MARGIN_S}", "--no-preflight"]
                     + (["--train-data", arg("--train-data")] if arg("--train-data") else [])}]
    pf = os.path.join(ROOT, "Process", "kaggle", "plans", f"plan_{nxt}.json")
    json.dump(plan, open(pf, "w"), indent=1)
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "automation", "launch_many.py"), runs, pf]).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
