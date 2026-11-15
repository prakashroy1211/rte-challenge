"""Train ONE run:  python train.py --exp albert_large__freelb --seed 42
Usually started by sweep.py; run it by hand to debug a single run (add --smoke for a 1-minute pipeline check)."""
import argparse

import common
from experiments import get_config

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, help="<model>__<recipe>, see experiments.py")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--smoke", action="store_true", help="tiny data + 1 epoch into work_smoke/ (checks the pipeline)")
ap.add_argument("--micro-batch", type=int, help="split each batch of 16 into pieces of this size (less GPU memory, same result)")
args = ap.parse_args()

if args.smoke:
    common.enable_smoke()
cfg = get_config(args.exp, smoke=args.smoke)
micro = args.micro_batch
if micro is None and common.gpu_memory_gb() < 24:            # 16 GB GPUs (T4, P100): smaller pieces, same batch of 16
    sizes = [m for m in (cfg["small_gpu_micro"], 8 if cfg["adv"] == "smart" else None) if m]   # SMART: 3 passes/step
    micro = min(sizes) if sizes else None
common.train_run(get_config(args.exp, smoke=args.smoke, micro_batch=micro), args.seed)
