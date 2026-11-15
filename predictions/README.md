# Predictions

| File | Content |
|---|---|
| `dev_predictions.jsonl` | the submitted model's predictions on the 277 dev pairs, one `{"idx": ..., "label": ...}` per line (dev accuracy 82.31%) |
| `test_predictions.jsonl` | test-set predictions (added once the test set is released) |

Both files are made by
`python inference.py --model albert_large_freelb --data data/<split>.jsonl --out predictions/<split>_predictions`
(or, from a training run, `python src/predict.py --exp albert_large__freelb --use best --split <dev|test>`).

Dev confusion matrix (rows = gold, columns = predicted):

| | entailment | not_entailment |
|---|---|---|
| **entailment** (146) | 124 | 22 |
| **not_entailment** (131) | 27 | 104 |
