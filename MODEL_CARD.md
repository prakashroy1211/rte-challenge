# Model Card: ALBERT-large-v2 + FreeLB (RTE)

**Author:** Prakash Roy · CS6315E Neural Networks and Deep Learning, RTE Challenge

One ALBERT-large-v2 model, fine-tuned with FreeLB adversarial training, labels a premise–hypothesis pair as
`entailment` or `not_entailment`. It has **17,686,018 parameters, all trainable**, and scores **82.3% on the RTE dev
set**.

## Architecture and parameters

- **Backbone:** [albert/albert-large-v2](https://huggingface.co/albert/albert-large-v2), a self-supervised pretrained
  checkpoint with no task head. It has 24 transformer layers that share one set of weights: hidden size 1024,
  16 attention heads, feed-forward 4096, factorized 128-dim embeddings, and a 30,000-token SentencePiece vocabulary.
- **Head:** pooled `[CLS]` vector (1024) → new linear layer → 2 logits.
- **Parameters:** 17,686,018 total = 17,686,018 trainable. This is full fine-tuning, with no LoRA or adapters.
- **Input → output:** `[CLS] premise [SEP] hypothesis [SEP]`, at most 256 tokens (the longer text is cut first)
  → softmax over 2 logits → argmax label.
- **Weights:** `albert_large_freelb.zip` in the repository's `v1.0` release (a standard Hugging Face folder,
  saved by training as `work/runs/albert_large__freelb/s46/best`). Inference:
  `python inference.py --model albert_large_freelb --data test.jsonl`.

## Training hyperparameters

| Setting | Value |
| --- | --- |
| Data | 2,241 pairs of train.jsonl for training, 249 held out (stratified, seed 1234). Dev is used only to evaluate. |
| Optimizer | AdamW, learning rate 1e-5, weight decay 0.01, eps 1e-6, gradient clipping 1.0 |
| Schedule | linear warmup over 25% of steps, then linear decay to 0 |
| Batch, length | 16 pairs, up to 256 tokens |
| Epochs | 10; the saved checkpoint is the epoch with the best hold-out accuracy (epoch 8) |
| Loss | cross-entropy |
| FreeLB | 2 ascent steps per batch; initial perturbation 1% and step 3% of each token's embedding norm |
| Seed | 46 |
| Precision, hardware | fp16 with loss scaling, one Kaggle T4, 26 minutes |

## Results

| Model | Parameters | Hold-out acc. (%) | Dev acc. (%) |
| --- | --- | --- | --- |
| **Submitted model** | 17,686,018 | 87.6 | 82.3 |
| FreeLB recipe, average of 5 seeds | 17,686,018 | 86.3 ± 1.3 | 82.5 ± 0.9 |
| Plain fine-tuning, average of 5 seeds | 17,686,018 | 83.8 ± 1.4 | 80.5 ± 1.2 |

FreeLB adds 1.9 dev points over plain fine-tuning of the same model on average (p = 0.02, 5 seeds each), without
adding any parameters.

**Rules:** the checkpoint has no classification head, which is verified when it is loaded, and its name carries no
NLI/RTE tuning. The only labels used are those in train.jsonl.
