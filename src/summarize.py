"""Everything trained so far, in one table + CSV files + an accuracy-vs-parameters plot.

    python summarize.py                       # all experiments
    python summarize.py --min-runs 5          # ladder uses only recipes with >= 5 successful runs

Recipes are RANKED BY HOLD-OUT mean (the 10% of train.jsonl), never by dev, so the dev numbers stay an honest
estimate. Outputs in work/summary/: experiments.csv, runs.csv, ladder.csv, accuracy_vs_params.png"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import warnings

import common
from experiments import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--min-runs", type=int, default=3, help="successful runs a recipe needs to enter the ladder")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    common.enable_smoke()
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ttest_ind
except ImportError:
    ttest_ind = None

os.makedirs(common.summary_dir(), exist_ok=True)
runs = common.all_results()
if not runs:
    raise SystemExit("no finished runs yet")
order = {m: i for i, m in enumerate(MODELS)}
for _m, _spec in MODELS.items():                     # continued-pretraining models sort right after their base model
    if "continued_from" in _spec:
        order[_m] = order[_spec["continued_from"]] + 0.5


def mean_sd(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


# ---------------------------------------------------------------- per experiment
by_exp = {}
for r in runs:
    by_exp.setdefault(r["experiment"], []).append(r)
rows = []
for exp, rs in by_exp.items():
    ok = [r for r in rs if not r["failed"]]
    dev = [r["dev_acc"] for r in ok if "dev_acc" in r]
    ho_m, ho_s = mean_sd([r["holdout_acc"] for r in ok])
    dv_m, dv_s = mean_sd(dev)
    chosen = common.rank_by_holdout(ok)[0] if ok else None
    rows.append({"experiment": exp, "family": rs[0]["family"], "model": rs[0]["model"], "recipe": rs[0]["recipe"],
                 "params": rs[0]["params"], "runs_ok": len(ok), "runs_total": len(rs),
                 "holdout_mean": ho_m, "holdout_sd": ho_s, "dev_mean": dv_m, "dev_sd": dv_s,
                 "dev_se": dv_s / len(dev) ** 0.5 if dev else float("nan"),
                 "chosen_seed": chosen["seed"] if chosen else None,
                 "chosen_dev": chosen.get("dev_acc") if chosen else None,
                 "minutes_per_run": st.mean(r["minutes"] for r in rs), "_dev": dev})
for row in rows:                                   # compare every recipe with the same model's "base" recipe
    ref = next((x for x in rows if x["experiment"] == f"{row['model']}__base"), None)
    row["dev_delta_vs_base"] = row["dev_mean"] - ref["dev_mean"] if ref and ref is not row else None
    row["p_vs_base"] = (float(ttest_ind(row["_dev"], ref["_dev"], equal_var=False).pvalue)
                        if ttest_ind and ref and ref is not row and len(row["_dev"]) > 1 and len(ref["_dev"]) > 1 else None)
rows.sort(key=lambda x: (x["family"], order.get(x["model"], 99), -x["holdout_mean"]))

# soups and pruned models (single models, so no spread)
extras = []
for kind, pattern in (("soup", os.path.join(common.soups_dir(), "*", "soup.json")),
                      ("pruned", os.path.join(common.pruned_dir(), "*", "pruned.json"))):
    for p in glob.glob(pattern):
        d = json.load(open(p))
        extras.append({"experiment": f"{d['experiment']} [{kind}]", "family": d["family"], "model": d["model"],
                       "recipe": d["recipe"], "params": d["params"], "holdout": d["holdout_acc"], "dev": d.get("dev_acc"),
                       "note": (f"{len(d['ingredients'])} runs averaged" if kind == "soup"
                                else f"vocab {d['vocab_kept']:,} (from {d['full_params'] / 1e6:.1f}M)")})


def pct(x): return "   -  " if x is None or x != x else f"{100 * x:6.2f}"


print(f"{'experiment':<36}{'params':>8}{'ok/runs':>9}{'hold-out mean+-sd':>19}{'dev mean+-sd':>17}{'+-SE':>7}"
      f"{'chosen':>8}{'dev vs base':>12}{'p':>7}{'min':>6}")
for r in rows:
    delta = "" if r["dev_delta_vs_base"] is None else f"{100 * r['dev_delta_vs_base']:+.2f}"
    p = "" if r["p_vs_base"] is None else f"{r['p_vs_base']:.3f}"
    print(f"{r['experiment']:<36}{r['params'] / 1e6:>7.1f}M{str(r['runs_ok']) + '/' + str(r['runs_total']):>9}"
          f"   {pct(r['holdout_mean'])} +-{pct(r['holdout_sd'])}  {pct(r['dev_mean'])} +-{pct(r['dev_sd'])}{pct(r['dev_se']):>7}"
          f"{pct(r['chosen_dev']):>8}{delta:>12}{p:>7}{r['minutes_per_run']:>6.1f}")
for e in extras:
    print(f"{e['experiment']:<36}{e['params'] / 1e6:>7.1f}M{'':>9}   {pct(e['holdout'])}{'':>10}{pct(e['dev'])}{'':>10}{e['note']}")
print("hold-out / dev = mean over successful runs; chosen = dev of the run with the best hold-out accuracy;"
      " p = Welch t-test of dev accuracies vs the same model's 'base' recipe")

with open(os.path.join(common.summary_dir(), "experiments.csv"), "w", newline="") as f:
    keys = [k for k in rows[0] if not k.startswith("_")]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows({k: r[k] for k in keys} for r in rows)
    for e in extras:
        w.writerow({"experiment": e["experiment"], "family": e["family"], "model": e["model"], "recipe": e["recipe"],
                    "params": e["params"], "holdout_mean": e["holdout"], "dev_mean": e["dev"]})
with open(os.path.join(common.summary_dir(), "runs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "seed", "failed", "best_epoch", "holdout_acc", "dev_acc", "train_ce_at_best", "minutes"])
    for r in sorted(runs, key=lambda r: (r["experiment"], r["seed"])):
        w.writerow([r["experiment"], r["seed"], r["failed"], r["best_epoch"], r["holdout_acc"], r.get("dev_acc"),
                    r["train_ce_at_best"], round(r["minutes"], 2)])

# ---------------------------------------------------------------- the ladder: best recipe per model size
ladder = []
for fam in sorted({r["family"] for r in rows}):
    for m in sorted({r["model"] for r in rows if r["family"] == fam}, key=lambda m: order.get(m, 99)):
        cands = [r for r in rows if r["model"] == m and r["runs_ok"] >= args.min_runs and r["recipe"] != "teacher"]
        cands = cands or [r for r in rows if r["model"] == m and r["runs_ok"] >= args.min_runs]
        if cands:
            ladder.append(max(cands, key=lambda r: r["holdout_mean"]))
print(f"\nLADDER - best recipe per model by HOLD-OUT mean (>= {args.min_runs} successful runs):")
print(f"{'family':<9}{'model':<24}{'recipe':<14}{'params':>9}{'hold-out':>10}{'dev mean+-sd':>17}{'chosen run dev':>16}")
prev = {}
for r in ladder:
    p = prev.get(r["family"])
    step = (f"   vs {p['model']}: x{r['params'] / p['params']:.1f} params, {100 * (r['dev_mean'] - p['dev_mean']):+.2f} dev"
            if p else "")
    print(f"{r['family']:<9}{r['model']:<24}{r['recipe']:<14}{r['params'] / 1e6:>8.1f}M{pct(r['holdout_mean']):>10}"
          f"{pct(r['dev_mean']):>10} +-{pct(r['dev_sd'])}{pct(r['chosen_dev']):>16}{step}")
    prev[r["family"]] = r
with open(os.path.join(common.summary_dir(), "ladder.csv"), "w", newline="") as f:
    keys = ["family", "model", "recipe", "params", "runs_ok", "holdout_mean", "dev_mean", "dev_sd", "dev_se", "chosen_seed", "chosen_dev"]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows({k: r[k] for k in keys} for r in ladder)

# ---------------------------------------------------------------- plot: dev accuracy vs parameters (one axis, log x)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt and ladder:
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    COLORS = {"albert": "#2a78d6", "deberta": "#eb6834"}          # validated categorical slots 1 and 2
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    labels = []
    for fam in sorted({r["family"] for r in ladder}):
        best_at = {}                       # one point per size: e.g. albert_large vs albert_large_rtd -> best by hold-out
        for r in ladder:
            if r["family"] == fam and r["dev_mean"] == r["dev_mean"]:
                k = round(r["params"] / 1e5)
                if k not in best_at or r["holdout_mean"] > best_at[k]["holdout_mean"]:
                    best_at[k] = r
        pts = sorted(best_at.values(), key=lambda r: r["params"])
        if not pts:
            continue
        xs = [r["params"] / 1e6 for r in pts]
        ys = [100 * r["dev_mean"] for r in pts]
        es = [100 * r["dev_se"] for r in pts]
        c = COLORS.get(fam, INK2)
        ax.plot(xs, ys, color=c, lw=2, zorder=2, label=f"{'ALBERT-v2' if fam == 'albert' else 'DeBERTa-v3'} (best recipe per size)")
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=8, color=c, mec="white", mew=2, elinewidth=1.5, capsize=3, zorder=3)
        labels.extend((x, y, f"{r['model'].split('_', 1)[1]} ({r['recipe']})") for x, y, r in zip(xs, ys, pts))
    ax.set_xscale("log")
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000)
             if min(r["params"] for r in ladder) / 2e6 <= t <= max(r["params"] for r in ladder) * 2 / 1e6]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("parameters (millions, log scale)", color=INK2)
    ax.set_ylabel("dev accuracy, % (mean of seeds ± SE)", color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    # direct labels: try four positions around each point, keep the first that overlaps nothing already drawn
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken = [ax.get_legend().get_window_extent(rend)]
    taken += [ax.transData.transform_bbox(matplotlib.transforms.Bbox([[x * 0.97, y - 0.15], [x * 1.03, y + 0.15]]))
              for x, y, _ in labels]
    axbox = ax.get_window_extent(rend)
    for x, y, text in labels:
        for dx, dy, ha in ((7, -13, "left"), (-7, 7, "right"), (7, 7, "left"), (-7, -13, "right")):
            a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK2)
            bb = a.get_window_extent(rend)
            inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
            if inside and not any(bb.overlaps(t) for t in taken):
                break
            a.remove()
        else:
            a = ax.annotate(text, (x, y), xytext=(7, -13), textcoords="offset points", fontsize=8, color=INK2)
            bb = a.get_window_extent(rend)
        taken.append(bb)
    ax.set_title("RTE dev accuracy vs model size", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    out = os.path.join(common.summary_dir(), "accuracy_vs_params.png")
    fig.savefig(out)
    print(f"\nplot: {out}")
print(f"tables: {common.summary_dir()}/experiments.csv, runs.csv, ladder.csv")
