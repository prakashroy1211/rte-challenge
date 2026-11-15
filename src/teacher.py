"""Build a distillation teacher: the averaged predictions of the best finished runs of one or more experiments,
on the training pairs plus rule-based perturbed copies of them (no language model is used to make the copies).

    python teacher.py --name deberta_large --exp deberta_large__teacher --top 5
    python teacher.py --name self_albert_large --exp albert_large__freelb --top 5      # born-again (self) teacher

Students then use it through a recipe with teacher=<name> (see experiments.py: kd, kd_freelb, self, self_freelb).
Only train.jsonl is involved: the teacher runs were trained on the same 90% split, and the hold-out / dev numbers
printed here are for the report only."""
import argparse
import json
import os

import numpy as np
import torch

import common
from experiments import expand, get_config

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="teacher name used by the recipes (deberta_large, self_albert_large, ...)")
ap.add_argument("--exp", nargs="+", required=True, help="experiment(s) whose runs form the teacher ensemble")
ap.add_argument("--top", type=int, default=5, help="use the best N successful runs by hold-out accuracy")
ap.add_argument("--n-aug", type=int, default=8, help="perturbed copies per training pair")
ap.add_argument("--aug-p", type=float, default=0.15, help="share of words deleted or masked in a copy")
ap.add_argument("--aug-seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    common.enable_smoke()

exps = expand(args.exp)
runs = common.rank_by_holdout([r for e in exps for r in common.exp_results(e) if common.has_weights(r)])[: args.top]
if not runs:
    raise SystemExit(f"no successful finished runs for {exps} - train them first with sweep.py")
cfg = get_config(exps[0], smoke=args.smoke)
train_ex, hold_ex = common.load_split(cfg)
aug_ex = common.augment(train_ex, args.n_aug if not args.smoke else 1, args.aug_p, args.aug_seed)
all_ex = train_ex + aug_ex
dev_ex = common.load_dev()
print(f"teacher '{args.name}': {len(runs)} member(s) | {len(train_ex)} training pairs + {len(aug_ex)} perturbed copies")

probs = {"all": [], "hold": [], "dev": []}
for r in runs:
    model, tok = common.load_model_dir(os.path.join(r["dir"], "best"))
    rc = json.load(open(os.path.join(r["dir"], "best", "run_config.json")))
    dtype = common.amp_dtype(rc)
    for key, ex in (("all", all_ex), ("hold", hold_ex), ("dev", dev_ex)):
        if not ex:
            continue
        logits, _, _ = common.predict(model, common.loader(ex, tok, rc, batch_size=128), dtype)
        probs[key].append(torch.softmax(torch.tensor(logits), -1).numpy())
    print(f"  member {r['experiment']} s{r['seed']}: hold-out {r['holdout_acc']:.4f}"
          + (f", dev {r['dev_acc']:.4f}" if "dev_acc" in r else ""), flush=True)
    del model
    torch.cuda.empty_cache()

mean = {k: np.mean(v, axis=0) for k, v in probs.items() if v}          # ensemble = averaged probabilities
logp = np.log(np.clip(mean["all"], 1e-6, 1.0))
os.makedirs(common.teachers_dir(), exist_ok=True)
with open(common.teacher_path(args.name), "w") as f:
    for e, lp in zip(all_ex, logp):
        f.write(json.dumps({"idx": e["idx"], "premise": e["premise"], "hypothesis": e["hypothesis"],
                            "label": e["label"], "teacher_logp": [round(float(x), 6) for x in lp]}) + "\n")


def acc(p, ex): return float(np.mean([int(np.argmax(x)) == e["label"] for x, e in zip(p, ex)]))


gold_agree = acc(mean["all"][: len(train_ex)], train_ex)
aug_flip = float(np.mean([int(np.argmax(a)) != e["label"] for a, e in
                          zip(mean["all"][len(train_ex):], train_ex * max(1, len(aug_ex) // max(1, len(train_ex))))]))
meta = {"name": args.name, "members": [f"{r['experiment']}/s{r['seed']}" for r in runs], "n_train": len(train_ex),
        "n_aug": len(aug_ex), "aug_p": args.aug_p, "aug_seed": args.aug_seed,
        "ensemble_holdout_acc": acc(mean["hold"], hold_ex),
        "ensemble_dev_acc": acc(mean["dev"], dev_ex) if dev_ex else None,
        "agreement_with_gold_on_train": gold_agree,
        "share_of_copies_where_teacher_differs_from_original_label": aug_flip}
json.dump(meta, open(os.path.join(common.teachers_dir(), f"{args.name}_meta.json"), "w"), indent=2)
print(f"ensemble hold-out {meta['ensemble_holdout_acc']:.4f}"
      + (f" | ensemble dev {meta['ensemble_dev_acc']:.4f}" if dev_ex else "")
      + f" | agrees with gold on train {gold_agree:.3f} | differs from the original label on {100 * aug_flip:.1f}% of copies")
print(f"saved {common.teacher_path(args.name)}")
