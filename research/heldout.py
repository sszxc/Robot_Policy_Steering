"""Re-evaluate the best z of every finished research.optimize chain (and z = 0 per init) on N held-out noise seeds.

Held-out seeds seed + init + 1000 * (HELD + j) never overlap the optimize seeds; all chains of an init share them,
so methods are paired. One task per run; appends to OUTPUT_DIR/results.jsonl and skips items already there.
python -m research.heldout --task-id 7 --roots out/libero/smolvla/icl/hpc out/icl-local --output-dir out/icl-heldout/t7
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from research import rollout
from research.evaluation.provenance import record
from research.inject import INJECT

HELD = 10000  # optimize seeds use rep * K + k, far below this
KEYS = ("layers", "dim", "alpha", "w", "w_seed", "seed", "suite")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rollout.add_args(parser)
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="dirs holding */*/provenance.json")
    parser.add_argument("--n", type=int, default=32, help="held-out noise seeds per z")
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--w", default="random")
    parser.add_argument("--w-seed", type=int, default=0)
    return parser.parse_args()


def method_name(cfg):
    return ("llm" if cfg.get("text_only") else cfg["method"]) + ("-perm" if cfg.get("perm") else "")


def items(args):
    """Best z (policy coordinates) of every finished chain of the task, plus z = 0 per init."""
    out, inits = [], set()
    for root in args.roots:
        for prov in sorted(root.glob("*/*/provenance.json")):
            cfg = json.loads(prov.read_text())["config"]
            if cfg["task_id"] != args.task_id:
                continue
            assert all(cfg[k] == vars(args)[k] for k in KEYS), f"inject/seed config differs: {prov}"
            for f in sorted((prov.parent / "chains").glob("i*-r*.jsonl")):
                cands = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                if not (cands and (max(c["rate"] for c in cands) == 1 or len(cands) >= cfg["budget"])):
                    continue  # running or empty
                init, rep = map(int, re.match(r"i(\d+)-r(\d+)", f.stem).groups())
                best = max(cands, key=lambda c: (c["rate"], -c["mean_steps"]))
                inits.add(init)
                out.append(dict(key=str(f.resolve()), method=method_name(cfg), init=init, rep=rep,
                                z_policy=best.get("z_policy", best["z"]), cand=best["i"], n_cands=len(cands),
                                train_rate=best["rate"]))
    return [dict(key=f"zero/i{i}", method="zero", init=i, rep=-1, z_policy=[0.0] * args.dim, cand=0, n_cands=0,
                 train_rate=None) for i in sorted(inits)] + out


def main():
    args = parse_args()
    out = args.output_dir
    assert out is not None, "--output-dir is required (re-run into it to resume)"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "provenance.json").exists():
        (out / "provenance.json").write_text(json.dumps(record(vars(args)), indent=2, default=str) + "\n")
    res_f = out / "results.jsonl"
    done = {json.loads(l)["key"] for l in res_f.read_text().splitlines()} if res_f.exists() else set()
    todo = [it for it in items(args) if it["key"] not in done]
    print(f"task {args.task_id}: {len(todo)} items x {args.n} seeds to run, {len(done)} done", flush=True)
    if not todo:
        return
    policy, env, procs = rollout.load(args)
    _, set_z = INJECT["expert_film"](policy, layers=[int(l) for l in args.layers.split(",")], dim=args.dim,
                                     alpha=args.alpha, w=args.w, seed=args.w_seed)
    eps = [(i, j) for i in range(len(todo)) for j in range(args.n)]
    res, B = {}, args.batch_size
    for s in range(0, len(eps), B):
        batch = eps[s:s + B]
        pad = batch + [batch[0]] * (B - len(batch))
        set_z(np.stack([todo[i]["z_policy"] for i, _ in pad]))
        r = rollout.run_batch(policy, env, procs, [todo[i]["init"] for i, _ in pad], args.seed,
                              noise_seeds=[args.seed + todo[i]["init"] + 1000 * (HELD + j) for i, j in pad])
        for b, e in enumerate(batch):
            res[e] = (bool(r["success"][b]), int(r["steps"][b]))
        for i in sorted({i for i, j in batch if j == args.n - 1}):  # item complete
            succ, steps = zip(*(res[(i, j)] for j in range(args.n)))
            with open(res_f, "a") as f:
                f.write(json.dumps(todo[i] | {"success": list(succ), "steps": list(steps),
                                              "rate": float(np.mean(succ))}) + "\n")
            print(f"{todo[i]['method']} i{todo[i]['init']} r{todo[i]['rep']}: train {todo[i]['train_rate']} "
                  f"-> held-out {np.mean(succ):.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
