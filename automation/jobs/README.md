# automation/jobs — job files (one Kaggle job per line, `run_kaggle.py <split> OUT/<name> ...`)

Launched with `automation/launch_many.py <runs_dir> <plan.json>` (plan items: acc, name, jobs, est_h, args) or
`automation/fanout.py`. Results of every job: `handoff/STATUS.md` (table "Tiến độ P2") and `Process/scores.csv`.

- Track R (submission flow on current code): `p2r1a … p2r16pred` (rounds 1-16), `sc*` (screening on subsets),
  `cf*` (confirmation, full sets, 2 seeds), `jsel*` (selector training data).
- Track M (frozen M0 code, worktree `b093d46`): `m0*`, `m7*`, `m2*`, `m4dpo`.
- Training: `w7f` (WF1-4 members), `wg` (WG1-4), `wg2r` (WG5-8), `d3_*` (teacher data D3), `m2*` / `m4dpo` (RFT / DPO).
- `final_predict.jobs`, `p2r6pred`, `p2r12pred`, `p2r16pred`: the container path (`--runner predict`).
- `prepreflight.py "<jobs>"`: runs `automation/preflight.py` and writes the cache mark so `fanout.py` launches at once.
