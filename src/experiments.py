"""Experiment grid. An experiment is named  <model>__<recipe>, e.g.  albert_large__freelb.
To try a new idea, add a recipe here (never change an existing one after its runs exist - results would mix)."""
import fnmatch
import glob
import json
import os

# Settings every experiment starts from (a model or a recipe can override any of them).
DEFAULTS = dict(
    max_len=256,            # tokens per (premise, hypothesis) pair
    epochs=10,
    evals_per_epoch=1,      # hold-out evaluations per epoch; the best one is the checkpoint we keep
    batch_size=16,
    grad_accum=1,
    lr=None,                # per model, see MODELS
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    label_smoothing=0.0,
    val_ratio=0.1,          # 10% of train.jsonl held out to pick the checkpoint and to rank recipes (never dev)
    split_seed=1234,        # same hold-out split for every model, recipe and seed
    use_amp=True,           # bf16 autocast on GPUs that support it
    adv="none",             # "none" | "freelb" | "smart"
    adv_steps=2,            # FreeLB ascent steps per batch
    adv_init=0.01,          # initial perturbation, relative to each token's embedding norm
    adv_lr=0.03,            # ascent step, relative to each token's embedding norm
    adv_max=0.0,            # FreeLB: max perturbation (relative); 0 = unbounded
    adv_eps=0.02,           # SMART: perturbation radius (relative)
    smart_alpha=1.0,        # SMART: weight of the smoothness term
    teacher=None,           # distillation: name of a teacher file made by teacher.py
    kd_T=2.0,               # distillation temperature
    kd_alpha=0.7,           # weight of the teacher term on gold pairs (perturbed copies use the teacher only)
    model_kwargs={},
)

# Pretrained checkpoints only (self-supervised, no task head). mem_gb = rough GPU memory per run for sweep.py;
# small_gpu_micro = micro-batch used automatically on GPUs under 24 GB (T4, P100) - same effective batch of 16.
MODELS = {
    "albert_base":    dict(hub="albert/albert-base-v2",    family="albert", lr=2e-5, warmup_ratio=0.25, mem_gb=5),
    "albert_large":   dict(hub="albert/albert-large-v2",   family="albert", lr=1e-5, warmup_ratio=0.25, mem_gb=8),
    "albert_xlarge":  dict(hub="albert/albert-xlarge-v2",  family="albert", lr=1e-5, warmup_ratio=0.25, mem_gb=14, small_gpu_micro=8),
    "albert_xxlarge": dict(hub="albert/albert-xxlarge-v2", family="albert", lr=1e-5, warmup_ratio=0.25, mem_gb=28, small_gpu_micro=4),
    "deberta_xsmall": dict(hub="microsoft/deberta-v3-xsmall", family="deberta", lr=4e-5, mem_gb=5),
    "deberta_small":  dict(hub="microsoft/deberta-v3-small",  family="deberta", lr=3e-5, mem_gb=6),
    "deberta_base":   dict(hub="microsoft/deberta-v3-base",   family="deberta", lr=2e-5, mem_gb=12),
    "deberta_large":  dict(hub="microsoft/deberta-v3-large",  family="deberta", lr=6e-6, mem_gb=28, small_gpu_micro=4),
}

# Models made by pretrain.py (work/pretrained/<name>/) are added automatically, with the settings of the model they
# started from:  albert_large_rtd__freelb, albert_large_mlm__base, ...
_base = os.environ.get("RTE_WORK_DIR", "work")
for _meta in sorted(glob.glob(os.path.join(_base, "pretrained", "*", "pretrain_meta.json"))
                    + glob.glob(os.path.join(_base + "_smoke", "pretrained", "*", "pretrain_meta.json"))):
    _name = os.path.basename(os.path.dirname(_meta))
    _from = json.load(open(_meta))["base_model"]
    if _name not in MODELS and _from in MODELS:
        MODELS[_name] = {**MODELS[_from], "hub": "local:" + os.path.abspath(os.path.dirname(_meta)), "continued_from": _from}

FREELB = dict(adv="freelb", adv_steps=2, adv_init=0.01, adv_lr=0.03)
KD = dict(epochs=3, evals_per_epoch=4, warmup_ratio=0.1)    # 1 epoch = gold pairs + all perturbed copies (~9x data)

RECIPES = {
    "base":        {},                                              # our stable recipe (ALBERT: lr 1e-5, 25% warmup)
    "long":        dict(epochs=20, warmup_ratio=0.1),               # Mosbach et al. 2021: small lr, train longer
    "freelb":      dict(**FREELB),                                  # adversarial training on the embeddings
    "freelb_hi":   dict(adv="freelb", adv_steps=3, adv_init=0.02, adv_lr=0.1, adv_max=0.2),   # stronger perturbation
    "smart":       dict(adv="smart", adv_lr=0.02, adv_eps=0.02, smart_alpha=1.0),
    "kd":          dict(teacher="deberta_large", **KD),             # distil the DeBERTa-v3-large ensemble
    "kd_freelb":   dict(teacher="deberta_large", **KD, **FREELB),
    "self":        dict(teacher="self_{model}", **KD),              # born-again: teacher = ensemble of the same model
    "self_freelb": dict(teacher="self_{model}", **KD, **FREELB),
    "teacher":     dict(evals_per_epoch=2),                         # for the DeBERTa-v3-large teacher runs
}

ALL_EXPERIMENTS = [f"{m}__{r}" for m in MODELS for r in RECIPES]


def get_config(exp, smoke=False, micro_batch=None):
    if "__" not in exp:
        raise SystemExit(f"experiment names look like <model>__<recipe>, e.g. albert_large__freelb (got '{exp}')")
    model, recipe = exp.split("__", 1)
    if model not in MODELS:
        raise SystemExit(f"unknown model '{model}'. Models: {', '.join(MODELS)}")
    if recipe not in RECIPES:
        raise SystemExit(f"unknown recipe '{recipe}'. Recipes: {', '.join(RECIPES)}")
    m = MODELS[model]
    cfg = {**DEFAULTS, **{k: v for k, v in m.items() if k in DEFAULTS}, **RECIPES[recipe]}
    cfg.update(experiment=exp, model=model, recipe=recipe, hub=m["hub"], family=m["family"],
               mem_gb=m["mem_gb"] * (1.5 if cfg["adv"] == "smart" else 1.0))
    cfg["small_gpu_micro"] = m.get("small_gpu_micro")
    if cfg["teacher"]:
        cfg["teacher"] = cfg["teacher"].format(model=model)
    if micro_batch and micro_batch != cfg["batch_size"]:      # same effective batch, split into accumulated pieces
        eff = cfg["batch_size"] * cfg["grad_accum"]
        if eff % micro_batch:
            raise SystemExit(f"--micro-batch {micro_batch} must divide the effective batch {eff}")
        cfg["batch_size"], cfg["grad_accum"] = micro_batch, eff // micro_batch
    if smoke:
        cfg.update(epochs=1, evals_per_epoch=2)
    return cfg


def expand(patterns):
    """'albert_large__*' or '*__kd' -> matching experiment names (order kept, no duplicates)."""
    out = []
    for p in patterns:
        hits = [e for e in ALL_EXPERIMENTS if fnmatch.fnmatch(e, p)] if any(c in p for c in "*?[") else [p]
        if not hits:
            raise SystemExit(f"no experiment matches '{p}'")
        for h in hits:
            get_config(h)                      # validates the name
            if h not in out:
                out.append(h)
    return out
