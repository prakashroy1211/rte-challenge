# Data

Put the course files here (they are not committed):

| File | Pairs | Use |
|---|---|---|
| `train.jsonl` | 2,490 | training (2,241) + stratified hold-out (249, split seed 1234) |
| `val.jsonl` | 277 | dev set: evaluation only |
| `test.jsonl` | hidden | test predictions |

One JSON object per line: `{"premise": "...", "hypothesis": "...", "label": "entailment" | "not_entailment", "idx": 0}`.
`.json`, `.tsv` and `.csv` files with `sentence1`/`sentence2` columns and integer labels (0 = entailment,
1 = not_entailment) also work. Test files may omit `label`.
