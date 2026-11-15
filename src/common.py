"""Shared code for the RTE experiments: data, models, the training loop (plain / FreeLB / SMART / distillation),
evaluation and file layout. Every other script imports this module."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # deterministic cuBLAS (set before CUDA starts)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")      # keep run logs readable
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import contextlib
import csv
import glob
import json
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABEL2ID = {"entailment": 0, "not_entailment": 1}
ID2LABEL = {0: "entailment", 1: "not_entailment"}
MODEL_KEYS = ("input_ids", "attention_mask", "token_type_ids")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAILED_CE = 0.5          # a run FAILED if the gold-label training loss at its saved checkpoint is still >= this

# ---------------------------------------------------------------- paths
DATA_DIR = os.environ.get("RTE_DATA_DIR", "data")
SMOKE = False            # --smoke: tiny data, 1 epoch, separate output folder (pipeline check only)


def enable_smoke():
    global SMOKE
    SMOKE = True


def work_dir():
    return os.environ.get("RTE_WORK_DIR", "work") + ("_smoke" if SMOKE else "")


def runs_dir(): return os.path.join(work_dir(), "runs")
def teachers_dir(): return os.path.join(work_dir(), "teachers")
def soups_dir(): return os.path.join(work_dir(), "soups")
def pruned_dir(): return os.path.join(work_dir(), "pruned")
def summary_dir(): return os.path.join(work_dir(), "summary")
def preds_dir(): return os.path.join(work_dir(), "predictions")
def run_dir(exp, seed): return os.path.join(runs_dir(), exp, f"s{seed}")


def resolve_model(hub_name):
    """Hub name -> what from_pretrained should load: 'local:<dir>' (our continued-pretraining outputs) -> <dir>;
    otherwise the hub name, or a local path if RTE_MODEL_MAP points to a JSON map (offline machines / tests)."""
    if hub_name.startswith("local:"):
        return hub_name[len("local:"):]
    path = os.environ.get("RTE_MODEL_MAP")
    return json.load(open(path)).get(hub_name, hub_name) if path else hub_name


# ---------------------------------------------------------------- data
def parse_label(raw):
    if raw is None or raw == "" or str(raw) == "-1":
        return None
    if isinstance(raw, str) and raw.strip() in LABEL2ID:
        return LABEL2ID[raw.strip()]
    return int(raw)


def read_examples(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tsv", ".csv"):
        with open(path, encoding="utf-8", newline="") as f:
            recs = list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",",
                                       quoting=csv.QUOTE_NONE if ext == ".tsv" else csv.QUOTE_MINIMAL))
    else:
        text = open(path, encoding="utf-8").read().strip()
        try:
            recs = json.loads(text)
            recs = recs if isinstance(recs, list) else recs.get("data", [])
        except json.JSONDecodeError:
            recs = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [{"idx": r.get("idx", r.get("index", i)),
             "premise": str(r.get("premise", r.get("sentence1"))).strip(),
             "hypothesis": str(r.get("hypothesis", r.get("sentence2"))).strip(),
             "label": parse_label(r.get("label"))} for i, r in enumerate(recs)]


def find_data(kind):
    names = {"train": ["train"], "dev": ["val", "dev", "validation"], "test": ["test"]}[kind]
    for n in names:
        for ext in (".jsonl", ".json", ".tsv", ".csv"):
            p = os.path.join(DATA_DIR, n + ext)
            if os.path.exists(p):
                return p
    return None


def stratified_split(examples, ratio, seed):
    rng = random.Random(seed)
    train, hold = [], []
    for lab in (0, 1):
        group = [e for e in examples if e["label"] == lab]
        rng.shuffle(group)
        n = int(round(len(group) * ratio))
        hold += group[:n]
        train += group[n:]
    rng.shuffle(train)
    return train, hold


def load_split(cfg):
    """The 90/10 train / hold-out split of train.jsonl. The same split for every model, recipe and seed."""
    path = find_data("train")
    assert path, f"no train file in {DATA_DIR}/ (expected train.jsonl)"
    labelled = [e for e in read_examples(path) if e["label"] is not None]
    if SMOKE:
        labelled = labelled[:80]
    return stratified_split(labelled, cfg["val_ratio"], cfg["split_seed"])


def load_dev():
    path = find_data("dev")
    if not path:
        return None
    ex = read_examples(path)
    assert all(e["label"] is not None for e in ex), f"{path} has no labels"
    return ex[:24] if SMOKE else ex


def load_test():
    path = find_data("test")
    if not path:
        return None
    ex = read_examples(path)
    return ex[:8] if SMOKE else ex


def augment(examples, n, p, seed):
    """Rule-based perturbed copies for distillation (no language model involved): every word is deleted with
    probability p/2 or replaced by [MASK] with probability p/2. Copies carry NO gold label - a perturbation can
    change the answer - so the teacher's probabilities are their only target."""
    rng = random.Random(seed + 7)

    def perturb(text):
        out = []
        for w in text.split():
            r = rng.random()
            if r < p / 2:
                continue
            out.append("[MASK]" if r < p else w)
        return " ".join(out) if out else text

    return [{"idx": f"aug{k}_{e['idx']}", "premise": perturb(e["premise"]), "hypothesis": perturb(e["hypothesis"]),
             "label": None} for k in range(n) for e in examples]


class RTEDataset(Dataset):
    """[CLS] premise [SEP] hypothesis [SEP], truncated to max_len (the longer text is cut first)."""

    def __init__(self, examples, tokenizer, max_len):
        enc = tokenizer([e["premise"] for e in examples], [e["hypothesis"] for e in examples],
                        truncation=True, max_length=max_len)
        keys = [k for k in MODEL_KEYS if k in enc]
        self.items = []
        for i, e in enumerate(examples):
            item = {k: enc[k][i] for k in keys}
            item["labels"] = e["label"] if e["label"] is not None else -100
            if "teacher" in e:
                item["teacher"] = e["teacher"]
            self.items.append(item)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_collate(tokenizer):
    def collate(batch):                                   # dynamic padding to the longest pair in the batch
        out = dict(tokenizer.pad([{k: b[k] for k in MODEL_KEYS if k in b} for b in batch],
                                 padding=True, return_tensors="pt"))
        out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        if "teacher" in batch[0]:
            out["teacher"] = torch.tensor([b["teacher"] for b in batch], dtype=torch.float)
        return out
    return collate


def loader(examples, tokenizer, cfg, shuffle=False, seed=0, batch_size=None):
    ds = RTEDataset(examples, tokenizer, cfg["max_len"])
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size or (cfg["batch_size"] if shuffle else 64), shuffle=shuffle,
                      generator=gen, collate_fn=make_collate(tokenizer))


# ---------------------------------------------------------------- teachers (distillation)
def teacher_path(name): return os.path.join(teachers_dir(), f"{name}.jsonl")


def load_teacher_examples(name, train_ex):
    """Training pairs + perturbed copies, each with the teacher's log-probabilities (written by teacher.py)."""
    path = teacher_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"teacher '{name}' not found ({path}) - run teacher.py --name {name} ... first")
    rows = [json.loads(line) for line in open(path)]
    gold_idx = sorted(str(r["idx"]) for r in rows if r["label"] is not None)
    assert gold_idx == sorted(str(e["idx"]) for e in train_ex), \
        "teacher file was built on a different train/hold-out split - rebuild it with teacher.py"
    return [{"idx": r["idx"], "premise": r["premise"], "hypothesis": r["hypothesis"], "label": r["label"],
             "teacher": r["teacher_logp"]} for r in rows]


# ---------------------------------------------------------------- models
# Challenge rule: start only from PRETRAINED checkpoints, never from one already fine-tuned on RTE / NLI / instructions.
FORBIDDEN_NAME_PARTS = ("mnli", "snli", "anli", "xnli", "nli", "rte", "glue", "superglue", "flan", "instruct")


def check_rules(name, loading_info):
    low = name.lower().split("/")[-1]
    bad = [p for p in FORBIDDEN_NAME_PARTS if p in low]
    assert not bad, f"RULES: '{name}' looks NLI/RTE/instruction fine-tuned ({bad}) - not allowed"
    missing = [str(k) for k in loading_info.get("missing_keys", [])]
    assert any("classifier" in k for k in missing), \
        f"RULES: '{name}' already contains a classification head (fine-tuned on some task) - not allowed"
    return sorted(k for k in missing if "classifier" in k or "pooler" in k)


def build_model(cfg):
    model, info = AutoModelForSequenceClassification.from_pretrained(
        resolve_model(cfg["hub"]), num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        output_loading_info=True, **cfg.get("model_kwargs", {}))
    new_head = check_rules(cfg["hub"], info)
    print(f"rules check passed: {cfg['hub']} is pretrained-only; newly initialised head: {new_head}")
    return model.float()                                 # fp32 master weights (some checkpoints are stored in fp16)


def count_params(model): return sum(p.numel() for p in model.parameters())   # shared ALBERT weights counted once


def load_model_dir(path):
    model = AutoModelForSequenceClassification.from_pretrained(path).float().to(DEVICE).eval()
    return model, AutoTokenizer.from_pretrained(path)


def amp_dtype(cfg=None):
    """bf16 on GPUs that have it (A100, H100/H200, GB10); fp16 + loss scaling on older ones (T4, P100, V100)."""
    if DEVICE.type != "cuda" or (cfg is not None and not cfg.get("use_amp", True)):
        return None
    # NOT torch.cuda.is_bf16_supported(): it also returns True when bf16 is only EMULATED (T4), which is very slow
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def gpu_memory_gb():
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if DEVICE.type == "cuda" else float("inf")


def make_scaler(dtype):
    if dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def backward(x, scaler):
    (scaler.scale(x) if scaler is not None else x).backward()


def finite(t):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def autocast(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict(model, dl, dtype=None):
    """Returns (logits list, mean CE loss or None, accuracy or None)."""
    model.eval()
    logits_all, tot, correct, n = [], 0.0, 0, 0
    for batch in dl:
        inputs = {k: batch[k].to(DEVICE) for k in MODEL_KEYS if k in batch}
        with autocast(dtype):
            logits = model(**inputs).logits.float()
        logits_all += logits.cpu().tolist()
        labels = batch["labels"].to(DEVICE)
        has = labels != -100
        if has.any():
            tot += F.cross_entropy(logits[has], labels[has], reduction="sum").item()
            correct += (logits[has].argmax(-1) == labels[has]).sum().item()
            n += has.sum().item()
    return logits_all, (tot / n if n else None), (correct / n if n else None)


def accuracy(logits, examples):
    return sum(int(np.argmax(l)) == e["label"] for l, e in zip(logits, examples)) / len(examples)


# ---------------------------------------------------------------- losses
def task_loss(logits, labels, teacher, cfg):
    """Gold pairs: cross-entropy (or alpha*KD + (1-alpha)*CE with a teacher). Perturbed copies: KD only.
    Returns (loss, summed gold CE, number of gold pairs)."""
    has = labels != -100
    ce_each = F.cross_entropy(logits, labels.clamp(min=0), reduction="none", label_smoothing=cfg["label_smoothing"])
    ce_each = ce_each * has
    if teacher is None:
        loss = ce_each.sum() / has.sum().clamp(min=1)
    else:
        T, a = cfg["kd_T"], cfg["kd_alpha"]
        t = F.log_softmax(teacher / T, -1)
        kd_each = (t.exp() * (t - F.log_softmax(logits / T, -1))).sum(-1) * T * T      # KL(teacher || student)
        loss = torch.where(has, a * kd_each + (1 - a) * ce_each, kd_each).mean()
    return loss, ce_each.detach().sum().item(), int(has.sum().item())


def sym_kl(p_logits, q_logits):
    p, q = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return ((p.exp() * (p - q)).sum(-1) + (q.exp() * (q - p)).sum(-1)).mean()


def unit(x):  # per-token unit vectors
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def train_step(model, batch, cfg, accum, dtype, scaler=None):
    """Forward + backward for one batch (gradients are accumulated, the optimizer step happens outside).
    adv="none":   plain training.
    adv="freelb": FreeLB (Zhu et al., ICLR 2020): K ascent steps on a perturbation of the word embeddings; the
                  parameter gradients of all K perturbed passes are accumulated ("free" adversarial training).
    adv="smart":  SMART-style smoothness term (Jiang et al., ACL 2020): CE on the clean input + alpha * symmetric KL
                  between clean and adversarially perturbed predictions (one ascent step).
    Perturbation sizes are RELATIVE to each token's embedding norm, so the same values work for ALBERT and DeBERTa."""
    inputs = {k: batch[k].to(DEVICE) for k in MODEL_KEYS if k in batch}
    labels = batch["labels"].to(DEVICE)
    teacher = batch["teacher"].to(DEVICE) if "teacher" in batch else None
    adv = cfg["adv"]
    if adv == "none":
        with autocast(dtype):
            logits = model(**inputs).logits.float()
        loss, ce_sum, n_gold = task_loss(logits, labels, teacher, cfg)
        backward(loss / accum, scaler)
        return loss.item(), ce_sum, n_gold

    emb_layer = model.get_input_embeddings()
    ids = inputs["input_ids"]
    rest = {k: v for k, v in inputs.items() if k != "input_ids"}
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    with torch.no_grad():
        norms = emb_layer(ids).norm(dim=-1, keepdim=True)            # per-token embedding norm

    if adv == "freelb":
        K = cfg["adv_steps"]
        delta = unit(torch.randn(*ids.shape, emb_layer.embedding_dim, device=DEVICE)) * cfg["adv_init"] * norms * mask
        total, first = 0.0, None
        for k in range(K):
            delta.requires_grad_()
            with autocast(dtype):
                logits = model(inputs_embeds=emb_layer(ids) + delta, **rest).logits.float()
            loss, ce_sum, n_gold = task_loss(logits, labels, teacher, cfg)
            backward(loss / (K * accum), scaler)
            total += loss.item() / K
            first = first or (ce_sum, n_gold)
            if k == K - 1:
                break
            delta = (delta.detach() + cfg["adv_lr"] * norms * unit(finite(delta.grad.detach()))) * mask
            if cfg["adv_max"] > 0:                                    # project onto the per-token ball
                scale = (cfg["adv_max"] * norms / (delta.norm(dim=-1, keepdim=True) + 1e-12)).clamp(max=1.0)
                delta = delta * scale
        return total, first[0], first[1]

    if adv == "smart":
        with autocast(dtype):
            logits = model(**inputs).logits.float()
        loss, ce_sum, n_gold = task_loss(logits, labels, teacher, cfg)
        emb = emb_layer(ids)
        noise = (unit(torch.randn_like(emb)) * 1e-3 * norms * mask).requires_grad_()
        with autocast(dtype):
            adv_logits = model(inputs_embeds=emb.detach() + noise, **rest).logits.float()
        g, = torch.autograd.grad(sym_kl(adv_logits, logits.detach()) * 1024.0, noise)   # scaled: fp16-safe, direction only
        noise = noise.detach() + cfg["adv_lr"] * norms * unit(finite(g))
        noise = noise * (cfg["adv_eps"] * norms / (noise.norm(dim=-1, keepdim=True) + 1e-12)).clamp(max=1.0) * mask
        with autocast(dtype):
            adv_logits = model(inputs_embeds=emb + noise, **rest).logits.float()
        total = loss + cfg["smart_alpha"] * sym_kl(adv_logits, logits)
        backward(total / accum, scaler)
        return total.item(), ce_sum, n_gold

    raise ValueError(f"unknown adv mode {adv}")


# ---------------------------------------------------------------- one training run
def eval_points(steps_per_epoch, epochs, per_epoch):
    pts = set()
    for e in range(epochs):
        for j in range(1, per_epoch + 1):
            pts.add(max(1, int(round(steps_per_epoch * (e + j / per_epoch)))))
    return pts


def train_run(cfg, seed):
    exp = cfg["experiment"]
    out = run_dir(exp, seed)
    if os.path.exists(os.path.join(out, "results.json")):
        print(f"skip {exp} seed {seed}: already trained")
        return json.load(open(os.path.join(out, "results.json")))
    os.makedirs(out, exist_ok=True)
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    t_start = time.time()
    dtype = amp_dtype(cfg)
    scaler = make_scaler(dtype)

    tok = AutoTokenizer.from_pretrained(resolve_model(cfg["hub"]))
    train_ex, hold_ex = load_split(cfg)
    n_gold = len(train_ex)
    if cfg.get("teacher"):
        train_ex = load_teacher_examples(cfg["teacher"], train_ex)
    dev_ex, test_ex = load_dev(), load_test()

    train_dl = loader(train_ex, tok, cfg, shuffle=True, seed=seed)
    hold_dl = loader(hold_ex, tok, cfg)
    model = build_model(cfg).to(DEVICE)
    n_params = count_params(model)
    accum = cfg["grad_accum"]
    precision = str(dtype).split(".")[-1] if dtype is not None else "fp32"
    batch_txt = f"batch {cfg['batch_size']}" + (f" x {accum} accumulated" if accum > 1 else "")
    print(f"=== {exp} | seed {seed} | {cfg['hub']} | {n_params:,} params | train {n_gold} gold"
          f"{f' + {len(train_ex) - n_gold} teacher-labelled copies' if cfg.get('teacher') else ''} | hold-out {len(hold_ex)}"
          f" | adv {cfg['adv']} | lr {cfg['lr']} | {cfg['epochs']} epochs | {batch_txt} | {DEVICE} {precision} ===", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], eps=1e-6)  # bias-corrected Adam
    steps_per_epoch = math.ceil(len(train_dl) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
    evals = eval_points(steps_per_epoch, cfg["epochs"], cfg["evals_per_epoch"])

    log, best, best_state = [], None, None
    step, w_loss, w_cnt, w_ce, w_gold, t0, diverged = 0, 0.0, 0, 0.0, 0, time.time(), False
    for epoch in range(cfg["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            loss, ce_sum, ng = train_step(model, batch, cfg, accum, dtype, scaler)
            if not math.isfinite(loss):
                diverged = True
                break
            w_loss += loss
            w_cnt += 1
            w_ce += ce_sum
            w_gold += ng
            if (i + 1) % accum and i + 1 != len(train_dl):
                continue
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            if scaler is not None:
                scaler.step(opt)                  # skipped automatically if fp16 gradients overflowed
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step not in evals:
                continue
            _, h_loss, h_acc = predict(model, hold_dl, dtype)
            row = {"step": step, "epoch": round(step / steps_per_epoch, 3), "train_loss": w_loss / max(w_cnt, 1),
                   "train_ce": w_ce / max(w_gold, 1), "holdout_loss": h_loss, "holdout_acc": h_acc}
            log.append(row)
            better = best is None or (h_acc, -h_loss) > (best["holdout_acc"], -best["holdout_loss"])
            if better:
                best = row
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            print(f"step {step:5d} (epoch {row['epoch']:5.2f}) | train loss {row['train_loss']:.4f} gold CE "
                  f"{row['train_ce']:.4f} | hold-out loss {h_loss:.4f} acc {h_acc:.4f} | {time.time() - t0:.0f}s"
                  + ("  <- best" if better else ""), flush=True)
            w_loss, w_cnt, w_ce, w_gold = 0.0, 0, 0.0, 0
            model.train()
        if diverged:
            print("!! loss became NaN/inf - stopping this run", flush=True)
            break

    if best is None:                                   # diverged before the first evaluation
        best = {"step": 0, "epoch": 0.0, "train_loss": float("nan"), "train_ce": float("nan"),
                "holdout_loss": float("nan"), "holdout_acc": 0.0}
        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_path = os.path.join(out, "best")
    model.save_pretrained(best_path)
    tok.save_pretrained(best_path)
    json.dump({**cfg, "seed": seed}, open(os.path.join(best_path, "run_config.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=2)

    res = {"experiment": exp, "model": cfg["model"], "recipe": cfg["recipe"], "family": cfg["family"],
           "hub": cfg["hub"], "seed": seed, "params": n_params, "best_step": best["step"], "best_epoch": best["epoch"],
           "holdout_acc": best["holdout_acc"], "holdout_loss": best["holdout_loss"],
           "train_ce_at_best": best["train_ce"], "final_train_ce": log[-1]["train_ce"] if log else float("nan"),
           "teacher": cfg.get("teacher"), "minutes": (time.time() - t_start) / 60}
    res["failed"] = not (best["train_ce"] < FAILED_CE)     # NaN (diverged before any checkpoint) also counts as failed
    if dev_ex:
        res["dev_logits"], _, res["dev_acc"] = predict(model, loader(dev_ex, tok, cfg), dtype)
    if test_ex:
        res["test_logits"], _, _ = predict(model, loader(test_ex, tok, cfg), dtype)
    json.dump(res, open(os.path.join(out, "results.json"), "w"))  # written last = the run is complete
    print(f"done: saved checkpoint at epoch {best['epoch']} | hold-out {best['holdout_acc']:.4f}"
          + (f" | dev {res['dev_acc']:.4f}" if dev_ex else "") + (" | FAILED (did not fit the training data)" if res["failed"] else "")
          + f" | {res['minutes']:.1f} min", flush=True)
    return res


# ---------------------------------------------------------------- reading finished runs
def all_results():
    out = []
    for p in glob.glob(os.path.join(runs_dir(), "*", "s*", "results.json")):
        r = json.load(open(p))
        r["dir"] = os.path.dirname(p)
        out.append(r)
    return out


def exp_results(exp, include_failed=False):
    res = sorted((r for r in all_results() if r["experiment"] == exp), key=lambda r: r["seed"])
    return res if include_failed else [r for r in res if not r["failed"]]


def has_weights(run):
    return os.path.exists(os.path.join(run["dir"], "best", "config.json"))


def rank_by_holdout(runs):
    return sorted(runs, key=lambda r: (-r["holdout_acc"], r["holdout_loss"]))


def chosen_run(exp):
    """Best successful run of an experiment by HOLD-OUT accuracy (decided without looking at dev)."""
    runs = [r for r in exp_results(exp) if has_weights(r)]
    assert runs, f"no successful runs with saved weights for {exp}"
    return rank_by_holdout(runs)[0]


# ---------------------------------------------------------------- outputs
def write_predictions(path, examples, logits):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    probs = torch.softmax(torch.tensor(logits), -1)[:, 0].tolist()
    preds = [int(np.argmax(l)) for l in logits]
    with open(path, "w") as f:
        for e, p in zip(examples, preds):
            f.write(json.dumps({"idx": e["idx"], "label": ID2LABEL[p]}) + "\n")
    with open(re.sub(r"\.jsonl$", "", path) + "_detailed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "premise", "hypothesis", "gold", "pred", "p_entailment"])
        for e, p, pr in zip(examples, preds, probs):
            w.writerow([e["idx"], e["premise"], e["hypothesis"], ID2LABEL.get(e["label"], ""), ID2LABEL[p], round(pr, 4)])
    return preds


NEG = {"not", "no", "never", "n't", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}


def _words(t): return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t.lower())


def slice_accuracy(examples, preds):
    def acc(mask):
        sel = [p == e["label"] for e, p, m in zip(examples, preds, mask) if m]
        return len(sel), (sum(sel) / len(sel) if sel else float("nan"))
    plen = [len(_words(e["premise"])) for e in examples]
    ovl = [len(set(_words(e["hypothesis"])) & set(_words(e["premise"]))) / max(len(set(_words(e["hypothesis"]))), 1)
           for e in examples]
    neg = [bool(NEG & set(_words(e["premise"] + " " + e["hypothesis"]))) for e in examples]
    return {"negation": acc(neg), "no negation": acc([not x for x in neg]),
            "premise <=30 words": acc([n <= 30 for n in plen]), "premise >30 words": acc([n > 30 for n in plen]),
            "overlap >=0.8": acc([o >= 0.8 for o in ovl]), "overlap <0.8": acc([o < 0.8 for o in ovl]),
            "gold entailment": acc([e["label"] == 0 for e in examples]),
            "gold not_entailment": acc([e["label"] == 1 for e in examples])}
