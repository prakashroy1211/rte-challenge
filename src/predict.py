"""Predictions for a chosen model (one line per pair: {"idx": ..., "label": "entailment" | "not_entailment"}).

    python predict.py --exp albert_large__kd_freelb --split test                 # best run by hold-out
    python predict.py --exp albert_large__kd_freelb --use soup --split test      # the weight-averaged soup
    python predict.py --exp deberta_base__base --use pruned --split dev          # the vocabulary-pruned model
    python predict.py --exp albert_large__freelb --seed 45 --split dev           # one specific run

--split dev also prints accuracy, the confusion matrix and accuracy on slices (negation, length, word overlap)."""
import argparse
import json
import os

import common
from experiments import get_config

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--use", choices=["best", "soup", "pruned"], default="best")
ap.add_argument("--seed", type=int, help="a specific run instead of the best one by hold-out")
ap.add_argument("--split", choices=["test", "dev"], default="test")
ap.add_argument("--out", help="output .jsonl (default: work/predictions/<exp>_<use>_<split>.jsonl)")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    common.enable_smoke()
get_config(args.exp)

examples = common.load_test() if args.split == "test" else common.load_dev()
if not examples:
    raise SystemExit(f"no {args.split} file in {common.DATA_DIR}/")
if args.use == "pruned":
    from shrink import load_pruned, pruned_loader
    path = os.path.join(common.pruned_dir(), args.exp)
    model, tok, pv = load_pruned(path)
    dl = pruned_loader(examples, pv, tok)
else:
    if args.use == "soup":
        path = os.path.join(common.soups_dir(), args.exp)
    elif args.seed is not None:
        path = os.path.join(common.run_dir(args.exp, args.seed), "best")
    else:
        path = os.path.join(common.chosen_run(args.exp)["dir"], "best")
    model, tok = common.load_model_dir(path)
rc = json.load(open(os.path.join(path, "run_config.json")))
if args.use != "pruned":
    dl = common.loader(examples, tok, rc)
logits, _, acc = common.predict(model, dl, common.amp_dtype(rc))

tag = f"{args.exp}_{args.use}" + (f"_s{args.seed}" if args.seed is not None else "")
out = args.out or os.path.join(common.preds_dir(), f"{tag}_{args.split}.jsonl")
preds = common.write_predictions(out, examples, logits)
print(f"{len(preds)} {args.split} predictions from {path} ({common.count_params(model):,} parameters) -> {out}")
if args.split == "dev":
    golds = [e["label"] for e in examples]
    cm = [[sum(1 for p, g in zip(preds, golds) if g == a and p == b) for b in (0, 1)] for a in (0, 1)]
    print(f"dev accuracy {acc:.4f} | confusion (rows gold, cols pred): entailment {cm[0]}  not_entailment {cm[1]}")
    for k, (n, a) in common.slice_accuracy(examples, preds).items():
        print(f"  {k:<22} n={n:<4} accuracy {a:.3f}")
