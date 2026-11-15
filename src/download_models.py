"""Download the pretrained checkpoints once (before starting parallel runs) and check each against the rules:
the name must not suggest NLI/RTE/instruction tuning, and the checkpoint must NOT contain a classification head.
Also prints the parameter count of each model with its new 2-class head.

    python download_models.py                                   # every model in experiments.py
    python download_models.py albert_large albert_xxlarge deberta_large"""
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import common
from experiments import MODELS

names = sys.argv[1:] or list(MODELS)
for key in names:
    hub = MODELS[key]["hub"]
    AutoTokenizer.from_pretrained(common.resolve_model(hub))
    model, info = AutoModelForSequenceClassification.from_pretrained(common.resolve_model(hub), num_labels=2,
                                                                     output_loading_info=True)
    head = common.check_rules(hub, info)
    print(f"{key:<16} {hub:<30} {common.count_params(model) / 1e6:7.1f}M params | rules OK, new head: {head}", flush=True)
