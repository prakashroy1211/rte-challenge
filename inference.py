"""Stand-alone inference for the submitted model (needs only: pip install torch transformers sentencepiece).

    python inference.py --model albert_large_freelb --data test.jsonl
    python inference.py --model albert_large_freelb --data val.jsonl          # labelled file: also prints accuracy

--model = the saved model folder (config.json, model.safetensors, spiece.model / tokenizer files)
--data  = .jsonl / .json / .tsv / .csv with premise + hypothesis (also accepts sentence1 / sentence2) and an id column
Writes <out>.jsonl  ({"idx": ..., "label": "entailment" | "not_entailment"} per line)  and  <out>.csv  (idx,label).
Runs on a GPU if there is one, otherwise on the CPU; pairs are batched by length so the CPU run stays fast."""
import argparse
import csv
import json
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ID2LABEL = {0: "entailment", 1: "not_entailment"}


def read_rows(path):
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
    rows = []
    for i, r in enumerate(recs):
        rows.append({"idx": r.get("idx", r.get("index", r.get("id", i))),
                     "premise": str(r.get("premise", r.get("sentence1", ""))).strip(),
                     "hypothesis": str(r.get("hypothesis", r.get("sentence2", ""))).strip(),
                     "label": r.get("label")})
    missing = sum(1 for r in rows if not r["premise"] or not r["hypothesis"])
    if missing:
        print(f"WARNING: {missing} row(s) without premise/hypothesis text - check the column names: {list(recs[0])}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="saved model folder")
    ap.add_argument("--data", required=True, help="test / dev file")
    ap.add_argument("--out", help="output name without extension (default: <data name>_predictions)")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    out = args.out or os.path.splitext(os.path.basename(args.data))[0] + "_predictions"
    out = out[:-6] if out.endswith(".jsonl") else out

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).float().to(device).eval()
    rows = read_rows(args.data)
    print(f"model {args.model}: {sum(p.numel() for p in model.parameters()):,} parameters | {len(rows)} pairs | {device}")

    enc = tok([r["premise"] for r in rows], [r["hypothesis"] for r in rows], truncation=True, max_length=args.max_len)
    order = sorted(range(len(rows)), key=lambda i: len(enc["input_ids"][i]))       # similar lengths together
    preds = [None] * len(rows)
    with torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            ids = order[b:b + args.batch_size]
            batch = tok.pad([{k: enc[k][i] for k in enc.keys()} for i in ids], return_tensors="pt").to(device)
            for i, p in zip(ids, model(**batch).logits.argmax(-1).tolist()):
                preds[i] = p
            done = min(b + args.batch_size, len(order))
            if done == len(order) or (b // args.batch_size) % 20 == 0:
                print(f"  {done}/{len(order)} pairs | {time.time() - t0:.0f}s", flush=True)

    with open(out + ".jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"idx": r["idx"], "label": ID2LABEL[p]}) + "\n")
    with open(out + ".csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label"])
        w.writerows([r["idx"], ID2LABEL[p]] for r, p in zip(rows, preds))
    n_ent = preds.count(0)
    print(f"wrote {out}.jsonl and {out}.csv | {n_ent} entailment, {len(preds) - n_ent} not_entailment | "
          f"{time.time() - t0:.0f}s in total")

    gold = [r["label"] for r in rows]
    norm = {"entailment": 0, "not_entailment": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    if all(g in norm for g in gold):
        acc = sum(int(p == norm[g]) for p, g in zip(preds, gold)) / len(gold)
        print(f"accuracy on this labelled file: {acc:.4f}")


if __name__ == "__main__":
    main()
