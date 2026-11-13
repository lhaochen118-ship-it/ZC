#!/usr/bin/env python3
"""Publish one staged folder as a PRIVATE dataset <acc>/<slug> on several Kaggle accounts in parallel (private datasets are
not shared across accounts, ops lessons §7.2), then wait until each one's files are live. Layout kept by two top folders:
<NAME>/adapter/... + info/README.txt (Kaggle unwraps a lone top folder).

Usage: publish_many.py <src_adapter_dir> <NAME> <slug> <acc1,acc2,...> [--wait-file adapter_model.safetensors] [--max-min 25]
  src_adapter_dir: a folder holding adapter_config.json + adapter_model.safetensors (hard-linked into the stage, no copy)
Prints one line per account: '<acc> LIVE' or '<acc> FAILED <reason>'; exit 0 when every account is live."""
import os, sys, json, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor


def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def kaggle(acc, *a, timeout=1800):
    env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser(f"~/.kaggle/tok_{acc}")).read().strip())
    return subprocess.run(["kaggle", *a], capture_output=True, text=True, env=env, timeout=timeout)


def stage_dir(src, name, slug, acc, root):
    st = os.path.join(root, f"pub_{slug}_{acc}")
    shutil.rmtree(st, ignore_errors=True)
    os.makedirs(os.path.join(st, name), exist_ok=True); os.makedirs(os.path.join(st, "info"), exist_ok=True)
    shutil.copytree(src, os.path.join(st, name, "adapter"), copy_function=os.link)
    open(os.path.join(st, "info", "README.txt"), "w").write(f"{name} adapter (keeps the folder layout)\n")
    json.dump({"title": slug, "id": f"{acc}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(st, "dataset-metadata.json"), "w"))
    return st


def one(acc, src, name, slug, root, wait_file, max_min):
    try:
        st = stage_dir(src, name, slug, acc, root)
        exists = kaggle(acc, "datasets", "status", f"{acc}/{slug}", timeout=120).stdout.strip().lower() == "ready"
        cmd = ["datasets", "version", "-p", st, "-m", "update", "-r", "zip", "-q"] if exists else ["datasets", "create", "-p", st, "-r", "zip", "-q"]
        r = kaggle(acc, *cmd)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0 and "error" in out.lower(): return f"{acc} FAILED create: {out[-200:]}"
        t0 = time.time()
        while time.time() - t0 < max_min * 60:
            r = kaggle(acc, "datasets", "files", f"{acc}/{slug}", timeout=120)
            if wait_file in r.stdout and name in r.stdout: return f"{acc} LIVE ({'versioned' if exists else 'created'}, {time.time() - t0:.0f}s)"
            time.sleep(20)
        return f"{acc} FAILED: files not live after {max_min} min"
    except Exception as e:
        return f"{acc} FAILED {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(os.path.join(root, f"pub_{slug}_{acc}"), ignore_errors=True)


def main():
    src, name, slug, accs = sys.argv[1:5]
    accs = [a for a in accs.split(",") if a]
    if not os.path.isfile(os.path.join(src, "adapter_config.json")): sys.exit(f"{src}: no adapter_config.json")
    root = os.path.dirname(os.path.abspath(src.rstrip("/")))
    wait_file, max_min = arg("--wait-file", "adapter_model.safetensors"), float(arg("--max-min", "25"))
    with ThreadPoolExecutor(len(accs)) as ex:
        res = list(ex.map(lambda a: one(a, src, name, slug, root, wait_file, max_min), accs))
    for r in res: print(r, flush=True)
    sys.exit(0 if all(" LIVE" in r for r in res) else 1)


if __name__ == "__main__":
    main()
