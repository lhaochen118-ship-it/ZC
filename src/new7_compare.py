#!/usr/bin/env python3
"""Paired comparison on strict-zero-shot P5 runs using repository's scorer.

Requires same D1 question IDs, scoring rules, seeds, model-generated rule
provenance, and no access to holdout answers for training/tuning.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

def read_sub(path):
    out={}
    for l in Path(path).read_text('utf-8').splitlines():
        if not l.strip():continue
        row=json.loads(l);k=row['id']
        if k in out:raise ValueError(f'duplicate question id: {k}')
        out[k]=row
    return out

def compare(gt_path,base_path,cand_path):
    import score
    truth=score.load(str(gt_path))
    b,c=read_sub(base_path),read_sub(cand_path)
    if set(b)!=set(c) or set(b)!=set(truth):
        raise ValueError('Baseline, candidate and GT question ID sets differ. Refuse invalid pairing.')
    def value(ans,k):
        if ans.get('type') in (None,'abstain'):return 0
        try:return 1 if score.judge(ans,truth[k])=='correct' else -1
        except Exception:return -1
    B={k:value(v,k) for k,v in b.items()};C={k:value(v,k) for k,v in c.items()}
    n=len(B);d=[C[k]-B[k] for k in B]
    win=sum(z>0 for z in d);loss=sum(z<0 for z in d);same=n-win-loss
    avb=sum(B.values())/n;avc=sum(C.values())/n;delta=avc-avb
    # Paired empirical standard error (not a replacement for bootstrap CI).
    sd=math.sqrt(sum((x-delta)**2 for x in d)/max(n-1,1))
    se=sd/math.sqrt(n)
    return {'questions':n,'baseline_PA':round(avb,6),'candidate_PA':round(avc,6),
            'delta_PA':round(delta,6),'improved':win,'regressed':loss,'ties':same,
            'paired_SE':round(se,6),'paired_z_approx':round(delta/se,3) if se else None,
            'note':'DataMate scorer from repo; no inference run by this script'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gt',required=True)
    ap.add_argument('--base',required=True);ap.add_argument('--candidate',required=True)
    a=ap.parse_args(); print(json.dumps(compare(a.gt,a.base,a.candidate),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
