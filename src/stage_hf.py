#!/usr/bin/env python3
"""P5 plan (STG): put Hugging Face weights into a Kaggle kernel output (CPU session with internet) so that GPU sessions (no
internet) mount them with --ks; src/run_kaggle.py finds <mount>/**/models/<Name>/config.json for `--model <org>/<Name>`.

Pure standard library (urllib, json, struct): no torch / safetensors needed, works on any CPU image.

    stage_hf.py <split> <out> --repo Qwen/Qwen3.5-4B --name Qwen3.5-4B                    # plain copy of the repo
    stage_hf.py <split> <out> --repo Qwen/Qwen3.6-35B-A3B-FP8 --name <N> --keep keep.json  # MoE expert pruning (REAP keep-list):
        every routed expert not in keep[layer] is dropped, the kept ones renumbered 0..E-1 in keep order, the router rows
        (mlp.gate.weight, bf16 in the official FP8 checkpoint) sliced the same way, the MTP head dropped, num_experts = E.
        The FP8 block scales (weight_scale_inv) travel with their expert, so the result is a valid official-format FP8 model.
<split> is ignored (run_kaggle.py job form). Output: <out>/models/<Name>/ + <out>/stage_manifest.json (sha256 of every file)."""
import os, sys, json, struct, hashlib, argparse, time, urllib.request, re, shutil

HF = "https://huggingface.co/{repo}/resolve/main/{f}"
SKIP_EXT = (".md", ".gitattributes")


def get(url, dst=None, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=300) as r:
                if dst is None: return r.read()
                tmp = dst + ".part"
                with open(tmp, "wb") as f: shutil.copyfileobj(r, f, 64 << 20)
                os.replace(tmp, dst); return dst
        except Exception as e:
            print(f"  retry {i + 1}/{tries} {url}: {type(e).__name__}: {e}", flush=True); time.sleep(5 * (i + 1))
    raise SystemExit(f"download failed: {url}")


def files_of(repo):
    d = json.loads(get(f"https://huggingface.co/api/models/{repo}"))
    return [s["rfilename"] for s in d["siblings"] if not s["rfilename"].endswith(SKIP_EXT)], d.get("sha", "")


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
    meta = h.pop("__metadata__", None)
    return h, base, meta


def write_st(path, tensors, meta=None):
    """tensors: list of (name, dtype, shape, bytes) in file order."""
    hdr, off = {}, 0
    for name, dt, shape, b in tensors:
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}; off += len(b)
    if meta: hdr["__metadata__"] = meta
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for _, _, _, b in tensors: f.write(b)


EXP = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.)(\d+)(\..*)$")
GATE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
FUSED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")   # bf16 checkpoint: [E, ...]


def prune_file(src, dst, keep):
    h, base, meta = read_st(src)
    out = []
    with open(src, "rb") as f:
        def raw(m):
            a, b = m["data_offsets"]; f.seek(base + a); return f.read(b - a)
        for name, m in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if name.startswith("mtp."): continue                        # MTP head (its own 256 experts): not used for decoding
            e = EXP.match(name)
            if e:
                L, j = e.group(2), int(e.group(3))
                if L not in keep or j not in keep[L]: continue
                out.append((f"{e.group(1)}{keep[L].index(j)}{e.group(4)}", m["dtype"], m["shape"], raw(m))); continue
            fu = FUSED.match(name)
            if fu and fu.group(1) in keep:                              # fused 3-D experts: one contiguous block per expert
                E0 = m["shape"][0]; b = raw(m); eb = len(b) // E0
                sl = b"".join(b[j * eb:(j + 1) * eb] for j in keep[fu.group(1)])
                out.append((name, m["dtype"], [len(keep[fu.group(1)])] + m["shape"][1:], sl)); continue
            g = GATE.match(name)
            if g and g.group(1) in keep:
                rows, cols = m["shape"]; b = raw(m); rb = len(b) // rows
                sl = b"".join(b[j * rb:(j + 1) * rb] for j in keep[g.group(1)])
                out.append((name, m["dtype"], [len(keep[g.group(1)]), cols], sl)); continue
            out.append((name, m["dtype"], m["shape"], raw(m)))
    if out: write_st(dst, out, meta)
    return [(n, s) for n, _, s, _ in out]


def nparams(shape):
    p = 1
    for x in shape: p *= x
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--repo", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--keep", default="", help="JSON {keep: {layer: [expert ids]}} (path relative to the code root or absolute)")
    ap.add_argument("--files", default="", help="comma-separated subset of repo files (default: all)")
    ap.add_argument("--src-dir", default="", help="local checkpoint (e.g. a mounted Kaggle Model) instead of downloading --repo; "
                    "a name is searched under /kaggle/input (P5 MoE fine-tuning: bf16 slice in the training kernel)")
    a = ap.parse_args()
    t0 = time.time(); dst = os.path.join(a.out, "models", a.name); os.makedirs(dst, exist_ok=True)
    tmp = "/tmp/stage_dl"; os.makedirs(tmp, exist_ok=True)
    src = None
    if a.src_dir:
        import glob as _g
        src = a.src_dir if os.path.isdir(a.src_dir) else next((os.path.dirname(c) for c in sorted(_g.glob("/kaggle/input/**/config.json", recursive=True))
                                                              if a.src_dir.lower() in c.lower() and _g.glob(os.path.join(os.path.dirname(c), "*.safetensors"))), None)
        if not src: raise SystemExit(f"--src-dir {a.src_dir} not found")
        files = sorted(os.path.relpath(os.path.join(d, f), src) for d, _, fs in os.walk(src) for f in fs if not f.endswith(SKIP_EXT)); sha = "local:" + src
    else: files, sha = files_of(a.repo)
    if a.files: files = [f for f in files if f in a.files.split(",")]
    keep = None
    if a.keep:
        kp = a.keep if os.path.isabs(a.keep) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), a.keep)
        if not os.path.isfile(kp): kp = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.keep)
        keep = {str(k): [int(x) for x in v] for k, v in json.load(open(kp))["keep"].items()}
        E = {len(v) for v in keep.values()}; assert len(E) == 1, f"keep lists of different sizes {E}"; E = E.pop()
        print(f"pruning to {E} experts / layer over {len(keep)} layers", flush=True)
    wmap, total, man = {}, 0, {"repo": a.repo, "revision": sha, "keep": a.keep, "files": {}}
    for f in files:
        p = os.path.join(tmp, f.replace("/", "_"))
        if src: shutil.copyfile(os.path.join(src, f), p) if not f.endswith(".safetensors") else None
        else: get(HF.format(repo=a.repo, f=f), p)
        if src and f.endswith(".safetensors"): p = os.path.join(src, f)              # read in place (no copy of 72 GB)
        if keep is not None and f.endswith(".safetensors"):
            names = prune_file(p, os.path.join(dst, f), keep)
            if not src: os.remove(p)
            for n, s in names: wmap[n] = f; total += nparams(s)
        elif keep is not None and f == "config.json":
            c = json.load(open(p)); tc = c.get("text_config", c); tc["num_experts"] = E
            tc["mtp_num_hidden_layers"] = 0
            c["_pruned_from"] = {"repo": a.repo, "revision": sha, "keep": os.path.basename(a.keep), "experts_per_layer": E}
            json.dump(c, open(os.path.join(dst, f), "w"), indent=2); os.remove(p)
        elif keep is not None and f.endswith(".index.json"):
            os.remove(p)                                                    # rewritten below
        else:
            os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True); shutil.move(p, os.path.join(dst, f))
        print(f"{f} done ({time.time() - t0:.0f}s)", flush=True)
    if keep is not None:
        json.dump({"metadata": {"total_parameters": total}, "weight_map": wmap},
                  open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    for d, _, fs in os.walk(dst):                                           # manifest: sha256 of everything written
        for f in fs:
            h = hashlib.sha256()
            with open(os.path.join(d, f), "rb") as fh:
                for ch in iter(lambda: fh.read(16 << 20), b""): h.update(ch)
            man["files"][os.path.relpath(os.path.join(d, f), dst)] = {"sha256": h.hexdigest(), "size": os.path.getsize(os.path.join(d, f))}
    if keep is not None: man["parameters_written"] = total
    json.dump(man, open(os.path.join(a.out, "stage_manifest.json"), "w"), indent=1)
    gb = sum(v["size"] for v in man["files"].values()) / 1e9
    print(f"staged {a.repo} -> {dst}: {len(man['files'])} files, {gb:.2f} GB" + (f", {total / 1e9:.3f}B parameters" if keep else "") +
          f" in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
