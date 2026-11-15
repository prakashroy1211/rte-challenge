# Earlier experiments

Exploratory runs from before the main ALBERT-large study. They were run with 1 seed each and are used only as
comparison points in the report's accuracy-vs-parameters figure, not for the submitted model.

- `explore_deberta_lora_distill_prune.ipynb`: the Kaggle notebook (with its outputs). It fine-tunes DeBERTa-v3
  xsmall, small and base with LoRA or full fine-tuning, distills base into xsmall, and prunes the vocabulary of each
  model to the tokens seen in the training data.
- `accuracy_vs_size.csv`: the notebook's summary table (total, pruned and trainable parameters, dev and hold-out
  accuracy for each run).

The ALBERT-base and ALBERT-xlarge comparison points can be re-run with the main pipeline:

```bash
python src/sweep.py --exp albert_base__base albert_xlarge__base --seeds 5 --gpus 0 --mem 15 --max-jobs 1
```
