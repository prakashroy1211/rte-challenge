"""Greedy model soup (Wortsman et al., 2022): average the WEIGHTS of several seeds of one experiment into a single
model with the same parameter count. Start from the best run by hold-out accuracy; add the next-best run only if
hold-out accuracy does not drop. Dev accuracy is measured once, at the end.

    python soup.py --exp albert_large__freelb albert_large__kd_freelb [--max 15]"""
import argparse
import json
import os

import torch

import common
from experiments import expand

ap = argparse.ArgumentParser()
ap.add_argument("--exp", nargs="+", required=True)
ap.add_argument("--max", type=int, default=15, help="consider at most the best N runs")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    common.enable_smoke()

dev_ex, test_ex = common.load_dev(), common.load_test()
for exp in expand(args.exp):
    runs = common.rank_by_holdout([r for r in common.exp_results(exp) if common.has_weights(r)])[: args.max]
    if len(runs) < 2:
        print(f"{exp}: soup skipped (fewer than 2 successful runs)")
        continue
    model, tok = common.load_model_dir(os.path.join(runs[0]["dir"], "best"))
    rc = json.load(open(os.path.join(runs[0]["dir"], "best", "run_config.json")))
    dtype = common.amp_dtype(rc)
    _, hold_ex = common.load_split(rc)
    hold_dl = common.loader(hold_ex, tok, rc)

    def state(run):
        return {k: v.float().cpu() for k, v in common.load_model_dir(os.path.join(run["dir"], "best"))[0].state_dict().items()}

    def holdout(sd, m=model):
        m.load_state_dict(sd)
        _, loss, acc = common.predict(m, hold_dl, dtype)
        return acc, loss

    soup, used = state(runs[0]), [runs[0]]
    best_acc, best_loss = holdout(soup)
    print(f"### {exp}: start s{runs[0]['seed']} hold-out {best_acc:.4f}")
    for r in runs[1:]:
        cand_sd, n = state(r), len(used)
        cand = {k: (soup[k] * n + cand_sd[k]) / (n + 1) if soup[k].is_floating_point() else soup[k] for k in soup}
        acc, loss = holdout(cand)
        keep = acc >= best_acc                     # Wortsman et al.: add if hold-out accuracy does not drop
        print(f"  + s{r['seed']:<5} hold-out {acc:.4f}  {'kept' if keep else 'rejected'}")
        if keep:
            soup, best_acc, best_loss = cand, acc, loss
            used.append(r)
    model.load_state_dict(soup)
    out = os.path.join(common.soups_dir(), exp)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    json.dump({**rc, "soup_of": [r["seed"] for r in used]}, open(os.path.join(out, "run_config.json"), "w"), indent=2)
    res = {"experiment": exp, "kind": "soup", "ingredients": [r["seed"] for r in used], "holdout_acc": best_acc,
           "best_single_holdout_acc": runs[0]["holdout_acc"], "params": common.count_params(model),
           "model": rc["model"], "recipe": rc["recipe"], "family": rc["family"]}
    if dev_ex:
        res["dev_logits"], _, res["dev_acc"] = common.predict(model, common.loader(dev_ex, tok, rc), dtype)
    if test_ex:
        res["test_logits"], _, _ = common.predict(model, common.loader(test_ex, tok, rc), dtype)
    json.dump(res, open(os.path.join(out, "soup.json"), "w"))
    print(f"  soup of {len(used)} run(s): hold-out {best_acc:.4f}"
          + (f" | dev {res['dev_acc']:.4f} (best single run by hold-out: dev {runs[0].get('dev_acc', float('nan')):.4f})" if dev_ex else "")
          + f" -> {out}")
    del model
    torch.cuda.empty_cache()
