"""Per-task z search (steer2 protocol): every candidate runs once on each search init; best z per chain re-run on held-out inits.

--inject noise: fixed initial flow noise x0 = U z (Golden Ticket; dim 1600 = full noise). --inject expert_film: FiLM z
with paired flow noise (seed + init + 1000 * rep), as research.optimize. CEM per chain (repeat), candidate 0 = z = 0;
a chain stops at budget or full success on the search inits. Held-out: inits × reps with noise seed + init + 1000 * j
(as steer2 val), plus z = 0 with --zero. Re-running into the same --output-dir resumes.
"""
import argparse
import json
import time

import numpy as np

from research import rollout
from research.evaluation.provenance import record
from research.inject import INJECT
from research.optimize import quantize, score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    rollout.add_args(parser)
    parser.add_argument("--inject", choices=INJECT, required=True)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--w", default="random")
    parser.add_argument("--w-seed", type=int, default=0, help="FiLM W seed / noise U seed")
    parser.add_argument("--search-inits", default="0-7")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--budget", type=int, default=48, help="candidates per chain, incl. z = 0")
    parser.add_argument("--box", type=float, default=1.5)
    parser.add_argument("--cem-pop", type=int, default=6)
    parser.add_argument("--cem-elite", type=int, default=2)
    parser.add_argument("--cem-sigma", type=float, default=0.5)
    parser.add_argument("--heldout-inits", default="10-39")
    parser.add_argument("--heldout-reps", type=int, default=2)
    parser.add_argument("--zero", action="store_true", help="also run z = 0 on the held-out inits")
    return parser.parse_args()


def ids(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def propose(args, cands, rep):
    """Next CEM generation (minus candidates already logged), replayed from the log."""
    mean, sigma = np.zeros(args.dim), np.full(args.dim, args.cem_sigma)
    gens = [cands[i:i + args.cem_pop] for i in range(0, len(cands), args.cem_pop)]
    full = [g for g in gens if len(g) == args.cem_pop]
    for g in full:
        elite = np.array([c["z"] for c in sorted(g, key=score, reverse=True)[:args.cem_elite]])
        mean, sigma = elite.mean(0), np.maximum(0.5 * sigma + 0.5 * elite.std(0), 0.1)
    g = len(full)
    rng = np.random.default_rng([args.seed, args.task_id, rep, g])
    Z = quantize(mean + sigma * rng.standard_normal((args.cem_pop, args.dim)), args.box)
    if g == 0:
        Z[0] = 0
    n = min(args.cem_pop, args.budget - g * args.cem_pop)
    return [(z, g) for z in Z[len(cands) - g * args.cem_pop:n]]


def run_eps(args, policy, env, procs, set_z, eps):
    """eps: [(z, init, noise_seed)] -> [(success, steps)] in batches of B (last batch padded)."""
    B, out = args.batch_size, []
    for s in range(0, len(eps), B):
        batch = eps[s:s + B]
        pad = batch + [batch[0]] * (B - len(batch))
        set_z(np.stack([z for z, _, _ in pad]))
        r = rollout.run_batch(policy, env, procs, [i for _, i, _ in pad], args.seed, noise_seeds=[n for _, _, n in pad])
        out += [(bool(r["success"][b]), int(r["steps"][b])) for b in range(len(batch))]
    return out


def main():
    args = parse_args()
    out = args.output_dir
    assert out is not None, "--output-dir is required (re-run into it to resume)"
    (out / "chains").mkdir(parents=True, exist_ok=True)
    if not (out / "provenance.json").exists():
        (out / "provenance.json").write_text(json.dumps(record(vars(args)), indent=2, default=str) + "\n")
    policy, env, procs = rollout.load(args)
    if args.inject == "noise":
        dim, set_z = INJECT["noise"](policy, dim=args.dim, seed=args.w_seed)
    else:
        dim, set_z = INJECT["expert_film"](policy, layers=[int(l) for l in args.layers.split(",")],
                                           dim=args.dim, alpha=args.alpha, w=args.w, seed=args.w_seed)
    assert dim == args.dim
    T = env.call("_max_episode_steps")[0]
    inits = ids(args.search_inits)
    logs = {r: out / "chains" / f"r{r}.jsonl" for r in range(args.repeats)}
    chains = {r: [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else [] for r, f in logs.items()}
    done = lambda c: len(c) >= args.budget or any(x["rate"] == 1 for x in c)

    while active := [r for r in chains if not done(chains[r])]:
        t0 = time.time()
        props = {r: propose(args, chains[r], r) for r in active}
        eps = [(z, i, args.seed + i + 1000 * r) for r in active for z, _ in props[r] for i in inits]
        res = iter(run_eps(args, policy, env, procs, set_z, eps))
        for r in active:
            for z, g in props[r]:
                rr = [next(res) for _ in inits]
                ok = [n for s, n in rr if s]
                cand = {"i": len(chains[r]), "gen": g, "z": [float(v) for v in z], "success": [s for s, _ in rr],
                        "steps": [n for _, n in rr], "rate": float(np.mean([s for s, _ in rr])),
                        "mean_steps": float(np.mean(ok)) if ok else float(T), "t": time.time()}
                chains[r].append(cand)
                with open(logs[r], "a") as f:
                    f.write(json.dumps(cand) + "\n")
        msg = " ".join(f"r{r}:best {max(c['rate'] for c in chains[r]):.2f}" for r in active)
        print(f"round: {len(eps)} eps {time.time() - t0:.0f}s | {msg}", flush=True)

    held = out / "heldout.jsonl"
    seen = {json.loads(l)["key"] for l in held.read_text().splitlines()} if held.exists() else set()
    items = [("zero", np.zeros(args.dim), None)] if args.zero else []
    for r, c in chains.items():
        best = max(c, key=score)
        items.append((f"r{r}", np.array(best["z"]), best))
    for key, z, best in items:
        if key in seen:
            continue
        eps = [(z, i, args.seed + i + 1000 * j) for i in ids(args.heldout_inits) for j in range(args.heldout_reps)]
        rr = run_eps(args, policy, env, procs, set_z, eps)
        rec = {"key": key, "z": [float(v) for v in z], "cand": best and best["i"], "n_cands": best and len(chains[int(key[1:])]),
               "train_rate": best and best["rate"], "init": [i for _, i, _ in eps], "noise_seed": [n for _, _, n in eps],
               "success": [s for s, _ in rr], "steps": [n for _, n in rr], "rate": float(np.mean([s for s, _ in rr]))}
        with open(held, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"heldout {key}: {rec['rate']:.3f}", flush=True)
    print(f"done -> {out}")
    env.close()


if __name__ == "__main__":
    main()
