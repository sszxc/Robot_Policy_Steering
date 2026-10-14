"""Figures + tables for docs/results/smolvla_ticket_object.html (per-task ticket search, all plotly).

ROOT/t{T}/{arm}/{chains/r*.jsonl, heldout.jsonl} from research.ticket; steer2 numbers from --steer2 report.json.
Writes OUT_DIR/*.html fragments (div only) and, with --template, the page (fragments inlined, plotly.min.js next to it).
python -m research.plot_ticket out/ticket-local --out out/ticket-figs --template research/ticket_report.html \
    --page docs/results/smolvla_ticket_object.html
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binomtest

from research.plot_icl import page
from research.plot_steer2 import wilson

TASKS = (0, 7, 9, 4, 2)
ARMS = ("film8", "noise8", "noise1600")
NAME = {"film8": "FiLM-z8", "noise8": "noise-z8", "noise1600": "noise-1600", "zero": "z=0"}
COL = {"film8": "#2a78d6", "noise8": "#eb6834", "noise1600": "#1baa7a", "zero": "#52514e", "steer2": "#8a63d2"}
LAYOUT = dict(template="plotly_white", font=dict(size=12), margin=dict(l=60, r=20, t=50, b=50),
              legend=dict(orientation="h", y=-0.18))


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load(root):
    """{(task, arm): {"chains": {rep: [cand]}, "held": {key: rec}}} for runs that exist."""
    data = {}
    for t in TASKS:
        for a in ARMS:
            d = root / f"t{t}" / a
            if not d.exists():
                continue
            chains = {int(f.stem[1:]): jl(f) for f in sorted((d / "chains").glob("r*.jsonl"))}
            data[t, a] = {"chains": {r: c for r, c in chains.items() if c}, "held": {h["key"]: h for h in jl(d / "heldout.jsonl")}}
    return data


def best_so_far(c):
    return np.maximum.accumulate([x["rate"] for x in c])


def first_full(c):
    return next((x["i"] for x in c if x["rate"] == 1), None)


def zero_held(data, t):
    return data.get((t, "film8"), {}).get("held", {}).get("zero")


def paired(h, z):
    """wins / losses of h vs z=0 on the same (init, rep j) held-out pairs (both lists in the same order)."""
    assert h["init"] == z["init"]
    s, s0 = np.array(h["success"]), np.array(z["success"])
    w, l = int((s & ~s0).sum()), int((~s & s0).sum())
    return w, l, binomtest(w, w + l).pvalue if w + l else 1.0


def save(fig, out, name):
    (out / f"{name}.html").write_text(fig.to_html(include_plotlyjs=False, full_html=False,
                                                  config={"displaylogo": False}, div_id=name))


def fig_search(data, out):
    tasks = [t for t in TASKS if any((t, a) in data for a in ARMS)]
    fig = make_subplots(1, len(tasks), subplot_titles=[f"t{t}" for t in tasks], shared_yaxes=True)
    shown = set()
    for col, t in enumerate(tasks, 1):
        for a in ARMS:
            for r, c in data.get((t, a), {}).get("chains", {}).items():
                x = [x["i"] for x in c]
                fig.add_trace(go.Scatter(x=x, y=[x["rate"] for x in c], mode="markers", marker=dict(color=COL[a], size=5, opacity=0.35),
                                         showlegend=False, hovertemplate=f"{NAME[a]} r{r} cand %{{x}}: %{{y:.3f}}<extra></extra>"), 1, col)
                fig.add_trace(go.Scatter(x=x, y=best_so_far(c), mode="lines", line=dict(color=COL[a], dash="solid" if r == 0 else "dash"),
                                         name=NAME[a], legendgroup=a, showlegend=a not in shown,
                                         hovertemplate=f"{NAME[a]} r{r} cand %{{x}}: best %{{y:.3f}}<extra></extra>"), 1, col)
                shown.add(a)
    fig.update_xaxes(title_text="candidate (0 = z=0)")
    fig.update_yaxes(title_text="success on search inits 0–7", range=[-0.05, 1.05], row=1, col=1)
    fig.update_layout(**LAYOUT, height=360, title="Search: candidate success (dots) and best so far (lines; dashed = repeat 1)")
    save(fig, out, "search")


def fig_gen0(data, out):
    """Generation-0 random candidates (z=0 excluded): how often a prior sample already helps."""
    fig = go.Figure()
    for a in ARMS:
        xs, ys = [], []
        for t in TASKS:
            for c in data.get((t, a), {}).get("chains", {}).values():
                for x in c:
                    if x["gen"] == 0 and x["i"] > 0:
                        xs.append(f"t{t}"), ys.append(x["rate"])
        if ys:
            fig.add_trace(go.Box(x=xs, y=ys, name=NAME[a], marker_color=COL[a], boxpoints="all", jitter=0.4, pointpos=0))
    for t in TASKS:
        z = [c[0]["rate"] for c in data.get((t, "film8"), {}).get("chains", {}).values()]
        if z:
            fig.add_trace(go.Scatter(x=[f"t{t}"], y=[np.mean(z)], mode="markers", marker=dict(symbol="x", size=11, color=COL["zero"]),
                                     name="z=0 (FiLM cand 0)", showlegend=t == TASKS[0], hovertemplate="z=0: %{y:.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT, boxmode="group", height=380, yaxis=dict(title="success on search inits 0–7", range=[-0.05, 1.05]),
                      title="Generation 0: random prior samples (5 per chain)")
    save(fig, out, "gen0")


def fig_heldout(data, steer2, out):
    fig = go.Figure()
    shown = set()
    for t in TASKS:
        rows = []
        if (z := zero_held(data, t)):
            rows.append(("zero", "z=0", z["success"]))
        for a in ARMS:
            for k, h in sorted(data.get((t, a), {}).get("held", {}).items()):
                if k != "zero":
                    rows.append((a, f"{NAME[a]} {k}", h["success"]))
        for a, lab, s in rows:
            p, lo, hi = wilson(sum(s), len(s))
            fig.add_trace(go.Scatter(x=[[f"t{t}"], [lab]], y=[p], error_y=dict(type="data", symmetric=False, array=[hi - p], arrayminus=[p - lo]),
                                     mode="markers", marker=dict(color=COL[a], size=9), name=NAME[a], legendgroup=a, showlegend=a not in shown,
                                     hovertemplate=f"t{t} {lab}: %{{y:.3f}} (n={len(s)})<extra></extra>"))
            shown.add(a)
        for h in steer2.get("heldout", []):
            if h["task"] == t and h["w"] == "random-s0":
                fig.add_trace(go.Scatter(x=[[f"t{t}"], ["steer2 grid pick (Sol)"]], y=[h["held"]], mode="markers",
                                         marker=dict(color=COL["steer2"], size=9, symbol="diamond-open"), name="steer2 FiLM grid pick, random-s0 (Sol)",
                                         legendgroup="steer2", showlegend="steer2" not in shown,
                                         hovertemplate=f"t{t} steer2 pick: %{{y:.3f}}; steer2 z=0 {h['zero']:.3f}<extra></extra>"))
                shown.add("steer2")
    fig.update_layout(**LAYOUT, height=420, yaxis=dict(title="held-out success (inits 10–39 × 2)", range=[-0.05, 1.05]),
                      title="Held-out: best z per chain vs z=0, 95% Wilson intervals")
    save(fig, out, "heldout")


def tab_summary(data, steer2, out):
    s2 = {h["task"]: h for h in steer2.get("heldout", []) if h["w"] == "random-s0"}
    z0s = steer2.get("z0", {})
    rows = []
    for t in TASKS:
        z = zero_held(data, t)
        for a in ARMS:
            d = data.get((t, a))
            if not d:
                continue
            for r, c in d["chains"].items():
                h = d["held"].get(f"r{r}")
                best = max(c, key=lambda x: (x["rate"], -x["mean_steps"]))
                cell = held = "–"
                if h:
                    p, lo, hi = wilson(sum(h["success"]), len(h["success"]))
                    held = f"{p:.2f} [{lo:.2f}, {hi:.2f}]"
                    if z:
                        w, l, pv = paired(h, z)
                        cell = f"{w}/{l}, p={pv:.2g}"
                ff = first_full(c)
                ref = f"{s2[t]['held']:.2f} / {z0s[f't{t}']['val'][0]:.2f}" if t in s2 else "–"
                cells = [f"t{t}", NAME[a], r, f"{c[0]['rate']:.2f}", len(c), "–" if ff is None else ff,
                         f"{best['rate']:.2f} (cand {best['i']})", held, f"{z['rate']:.2f}" if z else "–", cell, ref]
                rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    html = ("<table><tr><th>task</th><th>arm</th><th>rep</th><th>cand 0 (z=0)</th><th>cands used</th><th>first 8/8</th>"
            "<th>best in-sample</th><th>held-out [95% CI]</th><th>z=0 held-out</th><th>wins/losses vs z=0, sign test</th>"
            "<th>steer2 pick / z=0 (Sol)</th></tr>" + "".join(rows) + "</table>")
    (out / "tab_summary.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--steer2", type=Path, default=Path("out/libero/smolvla/steer2/hpc/report.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--page", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.root)
    steer2 = json.loads(args.steer2.read_text()) if args.steer2.exists() else {}
    fig_search(data, args.out)
    fig_gen0(data, args.out)
    fig_heldout(data, steer2, args.out)
    tab_summary(data, steer2, args.out)
    if args.template:
        page(args.template, args.out, args.page)


if __name__ == "__main__":
    main()
