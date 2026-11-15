"""Vocabulary pruning for DeBERTa-v3 models (most of their parameters are the 128K-token embedding table).
Keeps every token of train.jsonl + the K most common tokens + all single characters; any other word is split
into kept pieces. K is chosen on the HOLD-OUT split: the smallest K whose hold-out accuracy drops by at most
--tolerance. (ALBERT's embedding table is already small - 30K x 128 - so pruning it gains little.)

    python shrink.py --exp deberta_base__base [--use soup]"""
import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import common
from experiments import expand


class PrunedVocab:
    """Encodes text for a model whose embedding table keeps only `keep_ids` (old ids, sorted). A token outside the
    kept set is split into the longest kept pieces of its text, e.g. '▁Hutchinson' -> '▁Hut' 'chin' 'son'."""

    def __init__(self, tok, keep_ids, max_len):
        self.tok, self.max_len = tok, max_len
        self.new_id = {old: new for new, old in enumerate(keep_ids)}
        special = set(tok.all_special_ids)
        self.str2new = {tok.convert_ids_to_tokens(old): new for old, new in self.new_id.items() if old not in special}
        self.max_piece = max(len(k) for k in self.str2new)
        self.unk, self.cls, self.sep = (self.new_id[tok.unk_token_id], self.new_id[tok.cls_token_id],
                                        self.new_id[tok.sep_token_id])
        assert self.new_id[tok.pad_token_id] == tok.pad_token_id, "pad token must keep its id"
        self.cache, self.stats = {}, {"tokens": 0, "split": 0, "unk": 0}

    def _decompose(self, piece):
        out, i = [], 0
        while i < len(piece):
            for j in range(min(len(piece), i + self.max_piece), i, -1):
                if piece[i:j] in self.str2new:
                    out.append(self.str2new[piece[i:j]])
                    i = j
                    break
            else:
                out.append(self.unk)
                i += 1
        return out

    def _map(self, ids):
        out = []
        for i in ids:
            self.stats["tokens"] += 1
            if i in self.new_id:
                out.append(self.new_id[i])
                continue
            if i not in self.cache:
                self.cache[i] = self._decompose(self.tok.convert_ids_to_tokens(i))
            self.stats["split"] += 1
            self.stats["unk"] += self.cache[i].count(self.unk)
            out.extend(self.cache[i])
        return out

    def encode(self, premise, hypothesis):
        p = self._map(self.tok(premise, add_special_tokens=False)["input_ids"])
        h = self._map(self.tok(hypothesis, add_special_tokens=False)["input_ids"])[: self.max_len - 3]
        p = p[: self.max_len - 3 - len(h)]
        ids = [self.cls] + p + [self.sep] + h + [self.sep]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "token_type_ids": [0] * (len(p) + 2) + [1] * (len(h) + 1)}


class PrunedDataset(Dataset):
    def __init__(self, examples, pv):
        self.items = [{**pv.encode(e["premise"], e["hypothesis"]),
                       "labels": e["label"] if e["label"] is not None else -100} for e in examples]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def pruned_loader(examples, pv, tok):
    return DataLoader(PrunedDataset(examples, pv), batch_size=64, collate_fn=common.make_collate(tok))


def pruned_embedding(full_emb, keep_ids, pad_id):
    new = nn.Embedding(len(keep_ids), full_emb.embedding_dim, padding_idx=pad_id)
    new.weight.data.copy_(full_emb.weight.data[torch.tensor(keep_ids, device=full_emb.weight.device)])
    return new.to(full_emb.weight.device)


def keep_ids_for(tok, texts, top_k, V):
    ids = set(tok.all_special_ids) | set(range(min(top_k, V)))
    for p, h in texts:
        ids.update(tok(p, h)["input_ids"])
    n_tok = min(V, len(tok))
    for i, piece in enumerate(tok.convert_ids_to_tokens(list(range(n_tok)))):
        if piece is not None and len(piece.replace("▁", "")) <= 1:
            ids.add(i)
    return sorted(i for i in ids if i < V)


def load_pruned(path):
    model, tok = common.load_model_dir(path)
    meta = json.load(open(os.path.join(path, "vocab_keep.json")))
    return model, tok, PrunedVocab(tok, meta["keep_ids"], meta["max_len"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--use", choices=["best", "soup"], default="best", help="prune the best run (by hold-out) or the soup")
    ap.add_argument("--top-k", type=int, nargs="+", default=[5000, 10000, 20000, 40000])
    ap.add_argument("--tolerance", type=float, default=0.015, help="max hold-out accuracy drop (0.015 = ~4 examples)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        common.enable_smoke()
    dev_ex, test_ex = common.load_dev(), common.load_test()

    for exp in expand(args.exp):
        src = os.path.join(common.soups_dir(), exp) if args.use == "soup" else os.path.join(common.chosen_run(exp)["dir"], "best")
        model, tok = common.load_model_dir(src)
        rc = json.load(open(os.path.join(src, "run_config.json")))
        if rc["family"] != "deberta":
            print(f"{exp}: skipped - pruning is meant for DeBERTa-v3 (128K vocabulary)")
            continue
        dtype = common.amp_dtype(rc)
        train_ex, hold_ex = common.load_split(rc)
        labelled = train_ex + hold_ex
        full_emb = model.get_input_embeddings()
        V = full_emb.num_embeddings
        full_params = common.count_params(model)
        _, _, ho_full = common.predict(model, common.loader(hold_ex, tok, rc), dtype)

        sweep = []
        for K in args.top_k:
            keep_tr = keep_ids_for(tok, [(e["premise"], e["hypothesis"]) for e in train_ex], K, V)   # for the choice
            model.set_input_embeddings(pruned_embedding(full_emb, keep_tr, tok.pad_token_id))
            _, _, ho = common.predict(model, pruned_loader(hold_ex, PrunedVocab(tok, keep_tr, rc["max_len"]), tok), dtype)
            keep = keep_ids_for(tok, [(e["premise"], e["hypothesis"]) for e in labelled], K, V)     # final vocabulary
            model.set_input_embeddings(pruned_embedding(full_emb, keep, tok.pad_token_id))
            row = {"top_k": K, "kept": len(keep), "params": common.count_params(model), "holdout_acc": ho, "_keep": keep}
            if dev_ex:
                pv = PrunedVocab(tok, keep, rc["max_len"])
                row["_dev_logits"], _, row["dev_acc"] = common.predict(model, pruned_loader(dev_ex, pv, tok), dtype)
                row["dev_tokens_split"] = pv.stats["split"] / max(pv.stats["tokens"], 1)
            sweep.append(row)
        ok = [r for r in sweep if r["holdout_acc"] >= ho_full - args.tolerance]
        best = min(ok, key=lambda r: r["params"]) if ok else max(sweep, key=lambda r: (r["holdout_acc"], -r["params"]))

        print(f"\n### {exp} ({args.use}): full {full_params / 1e6:.1f}M params, hold-out {ho_full:.4f}")
        for r in sweep:
            print(f"  top {r['top_k']:>6,} | vocab {r['kept']:>7,} | {r['params'] / 1e6:6.1f}M | hold-out {r['holdout_acc']:.4f}"
                  + (f" | dev {r['dev_acc']:.4f}" if dev_ex else "") + ("   <- chosen" if r is best else ""))

        model.set_input_embeddings(pruned_embedding(full_emb, best["_keep"], tok.pad_token_id))
        model.config.vocab_size = best["kept"]
        out = os.path.join(common.pruned_dir(), exp)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        json.dump({**rc, "pruned_from": src}, open(os.path.join(out, "run_config.json"), "w"), indent=2)
        json.dump({"keep_ids": best["_keep"], "original_vocab_size": V, "top_k": best["top_k"], "max_len": rc["max_len"]},
                  open(os.path.join(out, "vocab_keep.json"), "w"))
        res = {"experiment": exp, "kind": "pruned", "source": args.use, "params": best["params"], "full_params": full_params,
               "top_k": best["top_k"], "vocab_kept": best["kept"], "holdout_acc_full": ho_full, "holdout_acc": best["holdout_acc"],
               "model": rc["model"], "recipe": rc["recipe"], "family": rc["family"],
               "sweep": [{k: v for k, v in r.items() if not k.startswith("_")} for r in sweep]}
        if dev_ex:
            res["dev_acc"], res["dev_logits"] = best["dev_acc"], best["_dev_logits"]
        if test_ex:
            res["test_logits"], _, _ = common.predict(model, pruned_loader(test_ex, PrunedVocab(tok, best["_keep"], rc["max_len"]), tok), dtype)
        json.dump(res, open(os.path.join(out, "pruned.json"), "w"))
        print(f"  saved {out}")
        del model, full_emb
        torch.cuda.empty_cache()
