#!/usr/bin/env python3
"""Change the submitted model (the LoRA soup Docker merges into Qwen3.5-9B at build time, docker/merge_assets.py) in one step:
  1. a local copy of the adapter dataset with the chosen members (hard links) and a new soup.json {"members", "weights"};
  2. src/adapter_hashes.json rewritten (sha256 of every file docker/fetch_assets.sh checks);
  3. with --publish, a new version of the private dataset nhatlong1103/zc-final-adapter (docker/assets.env ADAPTER_DS).
Members are folders <name>/adapter (adapter_config.json + adapter_model.safetensors) found under the --from directories.
Usage: set_final_model.py --members WF3 --weights 0.35 --from <dir>[,<dir>] --out <local dataset dir> [--doc "..."] [--publish]"""
import os, sys, json, glob, shutil, hashlib, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--doc", default=""); ap.add_argument("--publish", action="store_true")
    ap.add_argument("--dataset", default="nhatlong1103/zc-final-adapter")
    a = ap.parse_args()
    names = a.members.split(","); w = [float(x) for x in a.weights.split(",")]
    assert len(names) == len(w), "one weight per member"
    shutil.rmtree(a.out, ignore_errors=True); os.makedirs(a.out)
    for n in names:
        cands = [c for s in a.src.split(",") for c in glob.glob(os.path.join(s, "**", n, "adapter", "adapter_config.json"), recursive=True)]
        assert cands, f"member {n} not found under {a.src}"
        src = os.path.dirname(sorted(cands, key=len)[0]); dst = os.path.join(a.out, n, "adapter"); os.makedirs(dst)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            assert os.path.isfile(os.path.join(src, f)), f"{src}/{f} missing"
            os.link(os.path.join(src, f), os.path.join(dst, f))
    doc = a.doc or (f"DataMate final model: LoRA fine-tunes of Qwen3.5-9B merged into the base weights at Docker build time "
                    f"(docker/merge_assets.py), members {names} x {w} (sum {sum(w):.3f})")
    json.dump({"members": [f"{n}/adapter" for n in names], "weights": w, "_doc": doc}, open(os.path.join(a.out, "soup.json"), "w"), indent=1)
    files = {}
    for p in sorted(glob.glob(os.path.join(a.out, "**", "*"), recursive=True)):
        if os.path.isfile(p) and not p.endswith("dataset-metadata.json"):
            files[os.path.relpath(p, a.out)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump({"dataset": a.dataset, "files": files}, open(os.path.join(ROOT, "src", "adapter_hashes.json"), "w"), indent=1)
    print(f"{a.out}: {len(files)} files; src/adapter_hashes.json rewritten; soup.json = {names} x {w}")
    if a.publish:
        title = a.dataset.split("/")[1]
        json.dump({"title": title, "id": a.dataset, "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(a.out, "dataset-metadata.json"), "w"))
        env = dict(os.environ, KAGGLE_API_TOKEN=open(os.path.expanduser(f"~/.kaggle/tok_{a.dataset.split('/')[0]}")).read().strip())
        r = subprocess.run(["kaggle", "datasets", "version", "-p", a.out, "-m", f"final model {names} x {w}", "-r", "zip", "-q"],
                           capture_output=True, text=True, env=env)
        print((r.stdout + r.stderr).strip()[-300:])


if __name__ == "__main__":
    main()
