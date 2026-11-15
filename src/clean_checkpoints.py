"""Free disk space (Kaggle keeps at most 20 GB of output): delete the saved weights (best/) of runs you no longer need.
results.json is always kept - hold-out, dev and test predictions of every run stay in summarize.py.

    python clean_checkpoints.py --keep 3                               # dry run: lists what would go
    python clean_checkpoints.py --keep 3 --yes                         # keep the 3 best runs (by hold-out) per experiment
    python clean_checkpoints.py --drop deberta_large__teacher --yes    # all weights of an experiment (after teacher.py)

Kept unless --drop names their experiment: the best --keep successful runs of each experiment and every member of a
teacher (teachers/*_meta.json). Failed runs lose their weights."""
import argparse
import glob
import json
import os
import shutil

import common
from experiments import expand

ap = argparse.ArgumentParser()
ap.add_argument("--keep", type=int, default=3, help="best runs to keep per experiment")
ap.add_argument("--drop", nargs="*", default=[], help="experiments whose weights are all deleted")
ap.add_argument("--yes", action="store_true", help="really delete (otherwise only show)")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    common.enable_smoke()

drop = set(expand(args.drop)) if args.drop else set()
members = set()
for p in glob.glob(os.path.join(common.teachers_dir(), "*_meta.json")):
    members |= set(json.load(open(p))["members"])


def size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


by_exp = {}
for r in common.all_results():
    if common.has_weights(r):
        by_exp.setdefault(r["experiment"], []).append(r)
freed, n = 0, 0
for exp, runs in sorted(by_exp.items()):
    keep = set() if exp in drop else {r["seed"] for r in common.rank_by_holdout([r for r in runs if not r["failed"]])[: args.keep]}
    for r in runs:
        if r["seed"] in keep or (exp not in drop and f"{exp}/s{r['seed']}" in members):
            continue
        path = os.path.join(r["dir"], "best")
        b = size(path)
        freed += b
        n += 1
        print(f"{'deleting' if args.yes else 'would delete'} {path} ({b / 2**20:.0f} MB)")
        if args.yes:
            shutil.rmtree(path)
print(f"{'freed' if args.yes else 'would free'} {freed / 2**30:.2f} GB from {n} run(s)"
      + ("" if args.yes else " - add --yes to delete"))
