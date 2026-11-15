"""Run many experiments x seeds in parallel on one or more GPUs. Finished runs are skipped, so you can stop
(Ctrl-C) and restart any time.

    python sweep.py --exp albert_large__base albert_large__freelb --seeds 10
    python sweep.py --exp "albert_large__*" --seeds 20 --mem 120 --max-jobs 8
    python sweep.py --exp deberta_large__teacher --seeds 8 --gpus 0,1

Seeds are interleaved (seed 42 of every experiment first, then 43, ...), so if you stop early every
experiment has about the same number of seeds. Each run's log: work/runs/<experiment>/s<seed>/train.log"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque

import common
from experiments import expand, get_config

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, help="experiment names or patterns like 'albert_large__*'")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds per experiment")
    ap.add_argument("--first-seed", type=int, default=42)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, e.g. 0,1")
    ap.add_argument("--mem", type=float, default=120, help="GB of GPU memory to fill per GPU (H200: 141 GB)")
    ap.add_argument("--max-jobs", type=int, default=8, help="max parallel runs per GPU")
    ap.add_argument("--micro-batch", type=int, help="pass --micro-batch to every run (less memory, same result)")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline check into work_smoke/")
    args = ap.parse_args()
    if args.smoke:
        common.enable_smoke()

    exps = expand(args.exp)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    missing = sorted({get_config(e)["teacher"] for e in exps if get_config(e)["teacher"]
                      and not os.path.exists(common.teacher_path(get_config(e)["teacher"]))})
    if missing:
        skip = [e for e in exps if get_config(e)["teacher"] in missing]
        print(f"!! teacher file(s) missing: {missing} -> skipping {skip}\n   build them with teacher.py first")
        exps = [e for e in exps if e not in skip]
    jobs = deque((e, s) for s in seeds for e in exps
                 if not os.path.exists(os.path.join(common.run_dir(e, s), "results.json")))
    total = len(jobs)
    print(f"{len(exps)} experiment(s) x {len(seeds)} seed(s): {total} run(s) to do, "
          f"{len(exps) * len(seeds) - total} already finished")
    if not jobs:
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    used = {g: 0.0 for g in gpus}
    count = {g: 0 for g in gpus}
    running, retried, done, crashed = [], set(), 0, []

    def mem_of(job):
        return args.mem if job in retried else get_config(job[0])["mem_gb"]   # an OOM retry gets a GPU alone

    def start(job, gpu):
        exp, seed = job
        d = common.run_dir(exp, seed)
        os.makedirs(d, exist_ok=True)
        logf = open(os.path.join(d, "train.log"), "a")
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--exp", exp, "--seed", str(seed)] + (["--smoke"] if args.smoke else []) \
            + (["--micro-batch", str(args.micro_batch)] if args.micro_batch else [])
        env = {"OMP_NUM_THREADS": "4", **os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
        m = mem_of(job)
        used[gpu] += m
        count[gpu] += 1
        running.append({"p": p, "job": job, "gpu": gpu, "mem": m, "log": logf, "t0": time.time()})

    def stop_all(*_):
        print("\nstopping all runs ...")
        for r in running:
            r["p"].terminate()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    deadline = float(os.environ.get("RTE_DEADLINE", "inf"))     # unix time; set by the Kaggle notebook (12 h sessions)
    while jobs or running:
        past_deadline = time.time() > deadline              # no new runs; the running ones finish
        if past_deadline and not running:
            print(f"\nsession time limit reached: {len(jobs)} run(s) left - run the same command in the next session")
            break
        # first-fit: launch every queued run that fits on some GPU (an oversized run starts on an idle GPU)
        for job in ([] if past_deadline else list(jobs)):
            for g in gpus:
                fits = used[g] + mem_of(job) <= args.mem and count[g] < args.max_jobs
                if fits or count[g] == 0:
                    jobs.remove(job)
                    start(job, g)
                    break
        time.sleep(3)
        for r in list(running):
            code = r["p"].poll()
            if code is None:
                continue
            running.remove(r)
            r["log"].close()
            used[r["gpu"]] -= r["mem"]
            count[r["gpu"]] -= 1
            exp, seed = r["job"]
            res_path = os.path.join(common.run_dir(exp, seed), "results.json")
            if code == 0 and os.path.exists(res_path):
                done += 1
                res = json.load(open(res_path))
                dev = f" | dev {res['dev_acc']:.4f}" if "dev_acc" in res else ""
                print(f"[{done}/{total}] {exp} s{seed} | hold-out {res['holdout_acc']:.4f}{dev} | epoch {res['best_epoch']}"
                      f" | {res['minutes']:.1f} min" + ("  FAILED (did not fit training data)" if res["failed"] else ""), flush=True)
                continue
            tail = open(os.path.join(common.run_dir(exp, seed), "train.log")).read()[-1500:]
            if "out of memory" in tail.lower() and r["job"] not in retried:
                retried.add(r["job"])
                jobs.append(r["job"])
                print(f"   {exp} s{seed}: out of GPU memory - will retry with a GPU to itself", flush=True)
                continue
            crashed.append(r["job"])
            print(f"!! {exp} s{seed} crashed (exit {code}). Last lines of its log:\n{tail}", flush=True)

    print(f"\nfinished: {done} run(s) trained, {len(crashed)} crashed {crashed if crashed else ''}")
    print("next: python summarize.py")


if __name__ == "__main__":
    main()
