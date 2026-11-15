# CS6315E Neural Networks and Deep Learning: RTE Challenge

**Author:** Prakash Roy

Recognizing Textual Entailment (RTE): given a premise and a hypothesis, predict `entailment` or `not_entailment`.

The submitted model is **ALBERT-large-v2 fine-tuned with FreeLB adversarial training**. It has **17,686,018
parameters, all trainable** (full fine-tuning, no LoRA or adapters), and scores **82.3% on the dev set**. Over five
seeds, FreeLB raises dev accuracy from 80.5 ± 1.2% (plain fine-tuning) to 82.5 ± 0.9% (Welch's t-test, p = 0.02)
without adding any parameters.

| Deliverable | Where |
|---|---|
| Code (training + evaluation) | [`src/`](src), [`notebooks/rte_kaggle_t4.ipynb`](notebooks/rte_kaggle_t4.ipynb), [`inference.py`](inference.py) |
| Model card | [`MODEL_CARD.md`](MODEL_CARD.md) |
| Dev predictions | [`predictions/`](predictions) |
| Report (PDF + LaTeX source) | [`report/rte_report.pdf`](report/rte_report.pdf) |
| Result tables | [`results/experiments.csv`](results/experiments.csv) (per recipe), [`results/runs.csv`](results/runs.csv) (all 45 runs) |
| Trained weights | GitHub Release `v1.0`, file `albert_large_freelb.zip` (about 63 MB) |

## Results

ALBERT-large-v2 (17.7M parameters) with each training recipe, 5 seeds each (mean ± sd). A 10% hold-out of
`train.jsonl` picks each run's checkpoint and ranks the recipes; dev is only used for reporting.

| Recipe | Hold-out acc. (%) | Dev acc. (%) | Dev vs. plain (p) |
|---|---|---|---|
| plain fine-tuning (baseline) | 83.8 ± 1.4 | 80.5 ± 1.2 | – |
| long (20 epochs) | 83.2 ± 0.3 | 79.4 ± 0.7 | −1.2 (0.10) |
| **FreeLB** | **86.3 ± 1.3** | **82.5 ± 0.9** | **+1.9 (0.02)** |
| SMART | 85.4 ± 1.9 | 80.6 ± 2.5 | +0.1 (0.96) |
| kd (1 of 5 runs failed) | 82.5 ± 1.8 | 81.2 ± 0.9 | +0.7 (0.33) |
| kd + FreeLB | 83.8 ± 0.5 | 82.5 ± 0.7 | +1.9 (0.02) |
| self | 83.8 ± 3.2 | 81.9 ± 2.7 | +1.4 (0.31) |
| self + FreeLB | 85.5 ± 1.3 | 82.2 ± 0.7 | +1.7 (0.03) |
| *Teacher: DeBERTa-v3-large (435.1M)* | *90.8 ± 1.2* | *90.0 ± 0.8* | – |

The submitted model is the run with the highest hold-out accuracy among all 40 ALBERT-large runs: FreeLB, seed 46
(hold-out 87.6%, dev 82.3%).

## Inference with the submitted model

Needs only Python 3.9+ and `pip install torch transformers sentencepiece`. Runs on a GPU if there is one, otherwise
on the CPU. Pairs are batched by length, so the CPU is fast enough for the test set.

```bash
git clone https://github.com/prakashroy1211/rte-challenge.git
cd rte-challenge
pip install torch transformers sentencepiece

# weights: download albert_large_freelb.zip from the v1.0 release, then
unzip albert_large_freelb.zip            # -> albert_large_freelb/ (config.json, model.safetensors, tokenizer.json, ...)

python inference.py --model albert_large_freelb --data data/val.jsonl     # dev: prints accuracy 0.8231
python inference.py --model albert_large_freelb --data data/test.jsonl    # -> test_predictions.jsonl + .csv
```

`inference.py` reads `.jsonl`, `.json`, `.tsv` or `.csv` files with `premise`/`hypothesis` (or
`sentence1`/`sentence2`) and an `idx`/`id` column. It writes one `{"idx": ..., "label": ...}` line per pair, plus a
CSV copy. If the file has labels, it also prints the accuracy. No internet connection is needed once the model
folder is on disk.

## Reproducing the reported numbers

### Data

Put the course files in `data/` (they are not part of this repository):

```
data/train.jsonl    # 2,490 pairs: {"premise": ..., "hypothesis": ..., "label": "entailment" | "not_entailment", "idx": ...}
data/val.jsonl      # 277 dev pairs, same format
data/test.jsonl     # hidden test set, labels optional
```

### Option A: the Kaggle notebook (what produced the report)

[`notebooks/rte_kaggle_t4.ipynb`](notebooks/rte_kaggle_t4.ipynb) contains every script in `src/` and runs on
Kaggle's free **GPU T4 x2**. Add the data as an input, then commit the notebook once per session. For each later
session, attach the previous version's output as an input; finished runs are skipped.

| Session | Config cell | What runs | Time |
|---|---|---|---|
| 1 | `PHASES = ["check", "1"]` | plain, long, FreeLB and SMART on ALBERT-large, plus the DeBERTa-v3-large teacher runs, 5 seeds each | ~6.2 h |
| 2 | `PHASES = ["2", "3"]`, `SELF_FROM = "freelb"` | distillation from DeBERTa-v3-large (kd, kd+FreeLB), then self-distillation (self, self+FreeLB) | ~5 h |
| 3 | `PHASES = ["6"]` | soups, summary tables, dev predictions and error slices (CPU is enough) | ~5 min |

Outputs go to `results/` in the notebook output: `experiments.csv`, `runs.csv`, and
`albert_large__freelb_best_dev.jsonl` (the dev predictions).

### Option B: the scripts on any GPU

```bash
pip install -r requirements.txt          # run everything from the repository root (data in data/, outputs in work/)
G="--gpus 0 --mem 15 --max-jobs 1"       # one 16 GB GPU; e.g. "--gpus 0,1" for two, a larger --mem for bigger GPUs

python src/download_models.py albert_large deberta_large     # downloads checkpoints, checks the rules
python src/sweep.py --exp albert_large__freelb --seeds 2 --smoke $G     # 2-minute pipeline check

# 1. ALBERT-large recipes and the DeBERTa-v3-large teacher runs, seeds 42-46
python src/sweep.py --exp albert_large__base albert_large__long albert_large__freelb albert_large__smart \
                      deberta_large__teacher --seeds 5 $G

# 2. distillation from the DeBERTa-v3-large ensemble (the 3 best teacher runs by hold-out, 4 perturbed copies per pair)
python src/teacher.py --name deberta_large --exp deberta_large__teacher --top 3 --n-aug 4
python src/sweep.py --exp albert_large__kd albert_large__kd_freelb --seeds 5 $G

# 3. self-distillation (teacher = the 3 best FreeLB runs by hold-out)
python src/teacher.py --name self_albert_large --exp albert_large__freelb --top 3 --n-aug 4
python src/sweep.py --exp albert_large__self albert_large__self_freelb --seeds 5 $G

# 4. soups, tables, and dev predictions of the submitted model
python src/soup.py --exp albert_large__freelb albert_large__self_freelb albert_large__kd_freelb
python src/summarize.py                                          # -> work/summary/experiments.csv, runs.csv
python src/predict.py --exp albert_large__freelb --use best --split dev     # dev predictions + error slices
python src/predict.py --exp albert_large__freelb --use best --split test    # test predictions
```

The submitted model is saved as `work/runs/albert_large__freelb/s46/best`, a standard Hugging Face folder (load it
with `AutoModelForSequenceClassification.from_pretrained`). One FreeLB run takes about 26 minutes on a T4.

**Determinism.** Runs are seeded and use deterministic cuDNN/cuBLAS settings. GPU kernels can still differ in the
last digits between GPU types, so expect recipe means within a few tenths of a point of the table above.

## Code

| File | What it does |
|---|---|
| `src/common.py` | data loading and the 90/10 hold-out split, model loading with rule checks, the training loop (plain, FreeLB, SMART, distillation), prediction, result tables |
| `src/experiments.py` | every model and recipe; an experiment is named `<model>__<recipe>`, e.g. `albert_large__freelb` |
| `src/train.py` | trains one experiment with one seed |
| `src/sweep.py` | runs many experiments × seeds on one or more GPUs and skips finished runs |
| `src/teacher.py` | builds a teacher ensemble and its soft labels for the gold pairs and their perturbed copies |
| `src/soup.py` | greedy weight averaging of the seeds of one recipe (kept only if hold-out accuracy does not drop) |
| `src/summarize.py` | tables (mean ± sd over seeds, Welch's t-test against plain fine-tuning) and the accuracy-vs-parameters plot |
| `src/predict.py` | dev/test predictions and error-analysis slices for the best run of an experiment |
| `src/download_models.py`, `src/clean_checkpoints.py`, `src/shrink.py` | download and check checkpoints; free disk space; DeBERTa vocabulary pruning |
| `inference.py` | stand-alone inference with a saved model folder (only needs torch, transformers and sentencepiece) |
| `earlier_experiments/` | exploratory DeBERTa-v3 runs (LoRA, distillation, vocabulary pruning) behind some of the comparison points in the report's accuracy-vs-parameters figure |

## Protocol and rules

- **Data.** Only `train.jsonl` is used for training. A stratified 10% hold-out (249 pairs, split seed 1234) is the
  same for every model, recipe and seed. It picks each run's checkpoint and ranks recipes. `val.jsonl` is never used
  for training or selection.
- **Failure rule.** A run counts as failed if its gold-label training loss at the saved checkpoint is still ≥ 0.5.
  Failed runs are reported and excluded from the means.
- **Checkpoints.** Only self-supervised pretrained checkpoints (`albert/albert-*-v2`, `microsoft/deberta-v3-*`) are
  used. Every load checks that the checkpoint has no classification head and that its name has no NLI/RTE tuning.
- **No external labels.** No external NLI/RTE data and no LLM-generated data. Distillation teachers are our own
  models trained on the same 90% split. Their extra inputs are rule-based perturbed copies of the training pairs
  (random word deletion or `[MASK]`, p = 0.15).
