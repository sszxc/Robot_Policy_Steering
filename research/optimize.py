"""Optimize z on fixed init states, measuring sample efficiency (candidates until solved).

A chain = (init, repeat). Every candidate z of a chain runs on the same K flow-noise seeds
(seed + init + 1000 * (repeat * K + k), as research.run reps); candidate 0 is z = 0. --method vlm asks
a VLM in context (history + frames of the last candidate), cem samples Gaussian generations.
Chains share the env batch and stop at K/K success or --budget candidates. Re-running into the
same --output-dir resumes from the chain logs.
"""
import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

from platforms.paths import repo_root
from research import rollout
from research.evaluation.provenance import record
from research.inject import INJECT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    rollout.add_args(parser)
    parser.add_argument("--method", choices=["vlm", "cem"], required=True)
    parser.add_argument("--inits", default="0", help="comma list of init state ids")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--k", type=int, default=8, help="noise seeds per candidate")
    parser.add_argument("--budget", type=int, default=30, help="candidates per chain, incl. z = 0")
    parser.add_argument("--box", type=float, default=1.5, help="z coordinates in [-box, box], 0.1 grid")
    parser.add_argument("--layers", default="7,15,23,31")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--w", default="random")
    parser.add_argument("--w-seed", type=int, default=0)
    parser.add_argument("--cem-pop", type=int, default=6)
    parser.add_argument("--cem-elite", type=int, default=2)
    parser.add_argument("--cem-sigma", type=float, default=0.5)
    parser.add_argument("--model", default="qwen38-27b")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--frames", type=int, default=5, help="frames per rollout image")
    parser.add_argument("--view-every", type=int, default=5, help="capture policy images every N steps")
    parser.add_argument("--perm", action="store_true", help="per-chain random signed permutation between the optimizer's z and the policy's")
    return parser.parse_args()


# ---------- chains ----------

def score(c):
    """Sort key: success rate, then fewer mean steps of successful rollouts."""
    return (c["rate"], -c["mean_steps"])


def quantize(z, box):
    return np.round(np.clip(np.asarray(z, float), -box, box), 1) + 0.0  # no -0.0


class Chain:
    def __init__(self, args, init, rep, log):
        self.init, self.rep, self.log = init, rep, log
        self.name = f"i{init}-r{rep}"
        self.seeds = [args.seed + init + 1000 * (rep * args.k + k) for k in range(args.k)]
        rng = np.random.default_rng([args.seed, init, rep, 7])  # policy z = M @ optimizer z
        self.M = (np.eye(args.dim)[rng.permutation(args.dim)] * rng.choice([-1, 1], args.dim)) if args.perm else np.eye(args.dim)
        self.cands = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        self.budget = args.budget
        self.last_views = None  # {k: [T', H, 2W, 3]} of the last candidate (vlm)

    @property
    def done(self):
        return len(self.cands) >= self.budget or any(c["rate"] == 1 for c in self.cands)

    def add(self, cand):
        cand["i"] = len(self.cands)
        self.cands.append(cand)
        with open(self.log, "a") as f:
            f.write(json.dumps(cand) + "\n")


# ---------- CEM ----------

def cem_propose(args, chain):
    """Next generation (whole, minus candidates already logged). Generations are replayed from the
    log so a resumed run continues the same search."""
    mean, sigma = np.zeros(args.dim), np.full(args.dim, args.cem_sigma)
    gens = [chain.cands[i:i + args.cem_pop] for i in range(0, len(chain.cands), args.cem_pop)]
    full = [g for g in gens if len(g) == args.cem_pop]
    for g in full:
        elite = np.array([c["z"] for c in sorted(g, key=score, reverse=True)[:args.cem_elite]])
        mean, sigma = elite.mean(0), np.maximum(0.5 * sigma + 0.5 * elite.std(0), 0.1)
    g = len(full)
    rng = np.random.default_rng([args.seed, chain.init, chain.rep, g])
    Z = quantize(mean + sigma * rng.standard_normal((args.cem_pop, args.dim)), args.box)
    if g == 0:
        Z[0] = 0
    n_done = len(chain.cands) - g * args.cem_pop
    n = min(args.cem_pop, chain.budget - g * args.cem_pop)
    return [(z, {"source": "cem", "gen": g}) for z in Z[n_done:n]]


# ---------- VLM ----------

PROMPT = """You are optimizing a steering vector z for a pretrained robot manipulation policy (SmolVLA) in the LIBERO simulator.

Task given to the robot: "{task}"

z has {dim} continuous coordinates. It is added to the policy's internal activations (FiLM modulation of the action expert); z = 0 is the unmodified pretrained policy. The coordinates are abstract directions with no known meaning: infer their effects only from the evidence below, and move them independently (do not scale all coordinates together).
Each candidate z is evaluated on the SAME {k} rollouts (same initial scene, same {k} fixed random seeds), so differences between candidates come only from z. The score R is the number of successful rollouts out of {k}; ties are broken by fewer steps (episode limit {T} steps).
Bounds: each coordinate in [-{box}, {box}] with one decimal digit. From earlier experiments, |z| (Euclidean norm) above about 3 often breaks the policy, while |z| around 1 to 3 can change behavior substantially.

Evaluated candidates so far:
{history}

{images}
Goal: reach {k}/{k} successes with as few new candidates as possible. This is candidate {n} of at most {budget}; propose ONE new z that has not been evaluated yet.

Reason briefly, then end your answer with exactly one block:
<param>
params[0]: <x0>, params[1]: <x1>, ..., params[{last}]: <x{last}>
</param>"""


def history_text(chain, T):
    lines = []
    for c in chain.cands:
        z = ", ".join(f"{v:.1f}" for v in c["z"])
        steps = f"mean steps of successes {c['mean_steps']:.0f}" if c["rate"] > 0 else "all failed"
        lines.append(f"cand {c['i']}: z = [{z}] -> R = {sum(c['success'])}/{len(c['success'])} ({steps})")
    return "\n".join(lines)


def rollout_image(views, steps, n_frames):
    """n_frames evenly over the episode, stacked top to bottom; each row agentview | wrist, labeled."""
    T = min(len(views), (steps - 1) // VIEW_EVERY + 1)
    idx = sorted({round(i * (T - 1) / (n_frames - 1)) for i in range(n_frames)})
    rows = []
    for i in idx:
        row = np.ascontiguousarray(views[i])
        label = f"step {i * VIEW_EVERY}"
        cv2.putText(row, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    return np.concatenate(rows, 0)


def pick_rollouts(cand):
    """One successful and one failed rollout of the candidate; two of one kind if all agree."""
    succ = cand["success"]
    ok, bad = [k for k, s in enumerate(succ) if s], [k for k, s in enumerate(succ) if not s]
    return [ok[0], bad[0]] if ok and bad else (ok or bad)[:2]


def call_vlm(args, prompt, images, temperature):
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(
            cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()}}
        for img in images]
    r = requests.post(f"{API['base']}/chat/completions", timeout=(5, 300),
                      headers={"Authorization": f"Bearer {API['key']}"},
                      json={"model": args.model, "messages": [{"role": "user", "content": content}],
                            "temperature": temperature, "max_tokens": args.max_tokens,
                            "cache": {"no-cache": True}})  # LiteLLM proxy caches identical requests ~60 s
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    return (msg.get("content") or ""), (msg.get("reasoning_content") or msg.get("reasoning") or ""), d.get("usage", {})


def parse_params(text, dim):
    m = re.findall(r"<param\b[^>]*>(.*?)</param\s*>", text, flags=re.S | re.I)
    if not m:
        return None
    vals = dict((int(i), float(v)) for i, v in re.findall(r"params\[(\d+)\]\s*:\s*([-+]?\d*\.?\d+)", m[-1]))
    return np.array([vals[i] for i in range(dim)]) if all(i in vals for i in range(dim)) else None


def vlm_propose(args, chain, task, T, out):
    if not chain.cands:
        return [(np.zeros(args.dim), {"source": "zero"})]
    n = len(chain.cands)
    last = chain.cands[-1]
    shown = pick_rollouts(last)
    images = [rollout_image(chain.last_views[k], last["steps"][k], args.frames) for k in shown] if chain.last_views else []
    desc = [f"Image {j + 1}: rollout {k} of cand {last['i']} ({'SUCCESS at step ' + str(last['steps'][k]) if last['success'][k] else 'FAILED, ran ' + str(last['steps'][k]) + ' steps'})"
            for j, k in enumerate(shown)] if images else []
    img_text = ("Attached images show the most recent candidate (cand {}): {} frames over the episode, top to bottom, "
                "each row = third-person camera (left) | wrist camera (right).\n{}\n").format(last["i"], args.frames, "\n".join(desc)) if images else ""
    prompt = PROMPT.format(task=task, dim=args.dim, k=args.k, T=T, box=args.box, history=history_text(chain, T),
                           images=img_text, n=n, budget=chain.budget, last=args.dim - 1)
    d = out / "vlm" / chain.name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"prompt_{n:03d}.txt").write_text(prompt)
    for j, img in enumerate(images):
        cv2.imwrite(str(d / f"image_{n:03d}_{j}.jpg"), img[:, :, ::-1])
    seen = {tuple(c["z"]) for c in chain.cands}
    log, t0 = [], time.time()
    for retry in range(args.retries):
        if retry:
            time.sleep(10 * 2 ** (retry - 1))
        temp = min(1.0, args.temperature + 0.3 * retry)
        try:
            content, reasoning, usage = call_vlm(args, prompt, images, temp)
        except Exception as e:  # network / server errors: retry
            log.append(f"--- retry {retry} temp {temp:.2f}: error {e}")
            continue
        log.append(f"--- retry {retry} temp {temp:.2f} usage {usage}\n--- reasoning ---\n{reasoning}\n--- content ---\n{content}")
        z = parse_params(content, args.dim)
        if z is None:
            continue
        z = quantize(z, args.box)
        if tuple(z.tolist()) in seen:
            continue
        (d / f"response_{n:03d}.txt").write_text("\n\n".join(log))
        return [(z, {"source": "vlm" if retry == 0 else f"vlm_retry{retry}", "latency": time.time() - t0,
                     "usage": usage, "temperature": temp})]
    (d / f"response_{n:03d}.txt").write_text("\n\n".join(log))
    rng = np.random.default_rng([args.seed, chain.init, chain.rep, n])
    z = quantize(args.cem_sigma * rng.standard_normal(args.dim), args.box)
    return [(z, {"source": "fallback_random", "latency": time.time() - t0})]


# ---------- main loop ----------

API = {}
VIEW_EVERY = 5


def load_api():
    env = repo_root() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.lstrip().startswith("#") and v:
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    API["base"] = os.environ.get("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1").rstrip("/")
    API["key"] = os.environ["OPENAI_API_KEY"]


def main():
    global VIEW_EVERY
    args = parse_args()
    VIEW_EVERY = args.view_every
    out = args.output_dir
    assert out is not None, "--output-dir is required (re-run into it to resume)"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "provenance.json").exists():
        (out / "provenance.json").write_text(json.dumps(record(vars(args)), indent=2, default=str) + "\n")
    if args.method == "vlm":
        load_api()
    (out / "chains").mkdir(exist_ok=True)
    chains = [Chain(args, int(i), r, out / "chains" / f"i{int(i)}-r{r}.jsonl")
              for i in args.inits.split(",") for r in range(args.rep_start, args.rep_start + args.repeats)]
    policy, env, procs = rollout.load(args)
    dim, set_z = INJECT["expert_film"](policy, layers=[int(l) for l in args.layers.split(",")],
                                       dim=args.dim, alpha=args.alpha, w=args.w, seed=args.w_seed)
    assert dim == args.dim
    task, T = env.call("task_description")[0], env.call("_max_episode_steps")[0]
    B = args.batch_size
    pool = ThreadPoolExecutor(max(1, len(chains)))
    while active := [c for c in chains if not c.done]:
        t0 = time.time()
        if args.method == "vlm":
            props = list(pool.map(lambda c: vlm_propose(args, c, task, T, out), active))
        else:
            props = [cem_propose(args, c) for c in active]
        t_prop = time.time() - t0
        # episodes: (chain, candidate index in props, k)
        eps = [(ci, j, k) for ci, p in enumerate(props) for j in range(len(p)) for k in range(args.k)]
        res = {e: None for e in eps}
        for s in range(0, len(eps), B):
            batch = eps[s:s + B]
            pad = batch + [batch[0]] * (B - len(batch))
            set_z(np.stack([active[ci].M @ props[ci][j][0] for ci, j, _ in pad]))
            r = rollout.run_batch(policy, env, procs, [active[ci].init for ci, _, _ in pad], args.seed,
                                  noise_seeds=[active[ci].seeds[k] for ci, _, k in pad],
                                  view_every=VIEW_EVERY if args.method == "vlm" else 0)
            for b, e in enumerate(batch):
                res[e] = (bool(r["success"][b]), int(r["steps"][b]),
                          r["views"][b] if args.method == "vlm" else None)
        for ci, (c, p) in enumerate(zip(active, props)):
            for j, (z, meta) in enumerate(p):
                rr = [res[(ci, j, k)] for k in range(args.k)]
                succ, steps = [x[0] for x in rr], [x[1] for x in rr]
                ok = [n for s, n in zip(succ, steps) if s]
                c.add({"z": [float(v) for v in z], "z_policy": [float(v) for v in c.M @ z], "success": succ, "steps": steps, "rate": float(np.mean(succ)),
                       "mean_steps": float(np.mean(ok)) if ok else float(T), "t": time.time(), **meta})
                c.last_views = {k: rr[k][2] for k in range(args.k)} if args.method == "vlm" else None
        msg = " ".join(f"{c.name}:{c.cands[-1]['rate']:.2f}/best {max(x['rate'] for x in c.cands):.2f}" for c in active)
        print(f"round: {len(eps)} eps, propose {t_prop:.0f}s, total {time.time() - t0:.0f}s | {msg}", flush=True)
    print(f"all chains done -> {out}")
    env.close()


if __name__ == "__main__":
    main()
