# automation/jobs — job strings and analysis helpers of the 03/11 chat

- `v22.jobs`: the exact job string of v21 / v22 (train + public seed 0 / 1 / 2; one line, jobs separated by `;`).
  Next version: copy it to `vNN.jobs`, change only what the experiment changes.
- `prepreflight.py <jobs>`: runs `automation/preflight.py` and writes fanout's cache mark (so fanout starts at once).
- `mksub.sh vNN` (env `S` = scratch dir with `runs/`): vote the 3 public runs of vNN → validate → `submissions/vNN/`.
- Analysis of logged runs (no labels needed; they hard-code the old scratch dir — set `S` / paths before use):
  - `rule_vs_llm.py <run>`: per business, questions where every LLM sample disagrees with the rule program;
  - `agree_biz.py <run1> <run2> <run3>`: unanimous / majority / split per business;
  - `alt_diff.py <results>`: rule answers on the chosen vs the runner-up mapping;
  - `scan_maps.py <results>...`: rebuild every candidate mapping and list new-check hits (env `BIZ=dev_003,...`);
  - `two_maps.py <results> <biz> <cand1> <cand2>`: rebuild two candidates, pickle their C for comparison.
