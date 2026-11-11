"""Figures + tables for docs/results/smolvla_film_split_object.html: coupled (w=random) vs split FiLM-z8 projection.

G = outputs of experiments/libero/film_split/local.sh (openloop-*, grid-*, held-*, zero-*); T = research.ticket root.
Run keys: random (coupled), split, split-in / split-bn (split with γ on x̂: per-token InstanceNorm / leave-one-out BN stats).
python -m research.plot_film_split out/libero/smolvla/film_split out/ticket-local --out out/film-split-figs \
    --template research/film_split_report.html --page docs/results/smolvla_film_split_object.html
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research import analyze, plot_ticket as pt
from research.optimize import score
from research.plot_icl import page
from research.plot_steer2 import wilson

TASKS = (0, 7)
# group -> (run key, axes, colour); split axes < 4 drive γ only, axes ≥ 4 β only; split-in/bn grids hold the γ axes only
GROUPS = {"coupled": ("random", range(8), "#2a78d6"), "split γ axes": ("split", range(4), "#e87ba4"),
          "split β axes": ("split", range(4, 8), "#eda100"), "γ·x̂ axes, instance": ("split-in", range(4), "#6250d6"),
          "γ·x̂ axes, BN": ("split-bn", range(4), "#008300")}
RUNS = ("random", "split", "split-in", "split-bn")
ARMS = ("film8", "film8split", "noise8")
HELD = {"random-all": ("coupled pick", "#2a78d6"), "split-gamma": ("split pick, γ axes", "#e87ba4"),
        "split-beta": ("split pick, β axes", "#eda100"), "split-in-gamma": ("γ·x̂ pick, instance", "#6250d6"),
        "split-bn-gamma": ("γ·x̂ pick, BN", "#008300")}
ZERO = "#52514e"
save = pt.save


def layout(fig, **kw):
    fig.update_layout(**{**pt.LAYOUT, **kw})


def fig_openloop(G, out, rep):
    """First-action RMS Δa vs axis scale (fixed flow noise), mean over the group's axes, band = min–max over axes."""
    fig = make_subplots(1, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    for col, t in enumerate(TASKS, 1):
        for name, (w, axes, color) in GROUPS.items():
            f = G / f"openloop-t{t}-{'coupled' if w == 'random' else w}" / "openloop.npz"
            if not f.exists():
                continue
            d = np.load(f)
            A = d["dose"].shape[-1]
            per = np.linalg.norm(d["dose"][..., 0, :], axis=-1).mean(0)[list(axes)] / np.sqrt(A)  # [axes, scales]
            x = np.r_[0, d["scales"]]
            per = np.c_[np.zeros(len(per)), per]
            fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[per.max(0), per.min(0)[::-1]], fill="toself", fillcolor=color,
                            opacity=0.15, line=dict(width=0), legendgroup=name, showlegend=False, hoverinfo="skip",
                            row=1, col=col)
            fig.add_scatter(x=x, y=per.mean(0), mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8),
                            name=name, legendgroup=name, showlegend=col == 1, row=1, col=col,
                            hovertemplate=f"{name} c=%{{x}}: RMS Δa0 %{{y:.3f}}<extra></extra>")
            rep.setdefault("openloop", {}).setdefault(f"t{t}", {})[name] = dict(zip(map(str, x), per.mean(0).round(4).tolist()))
            nsd = float(d["noise_a"][..., 0, :].std(1).mean())
        fig.add_hline(y=nsd, line=dict(color=ZERO, dash="dash"), row=1, col=col,
                      annotation_text="flow-noise sd at z=0", annotation_position="top left")
        rep["openloop"][f"t{t}"]["noise_sd"] = round(nsd, 4)
    fig.update_xaxes(title_text="axis scale c (z = c·√8·e_k)")
    fig.update_yaxes(title_text="RMS Δa of first action (normalized)", row=1, col=1)
    layout(fig, height=380, title="Open loop: action change per axis (96 saved inputs, fixed flow noise)")
    save(fig, out, "openloop")


def grid_runs(G, t):
    R = {w: G / f"grid-t{t}-{w}" for w in RUNS}
    return {w: analyze.load(d) for w, d in R.items() if (d / "DONE").exists()}


def group_eps(ep, axes, c):
    m = np.isin(ep["dir"], list(axes)) & (ep["scale"] == c) & (ep["ep_z"] > 0)
    return {k: v[m] if isinstance(v, np.ndarray) and len(v) == len(ep["ep_z"]) else v for k, v in ep.items()}


def zero_eps(G, t, ep):
    """z=0 episodes: the 8-rep zero-t{t} run if present (noise reference), else the grid's own z=0 (rep 0 only)."""
    d = G / f"zero-t{t}"
    z = analyze.load(d) if (d / "DONE").exists() else ep
    return {k: v[z["ep_z"] == 0] if isinstance(v, np.ndarray) and len(v) == len(z["ep_z"]) else v for k, v in z.items()}


def fig_grid(G, out, rep):
    """Success on inits 0–7 vs signed scale, mean over the group's axes (band = min–max over axes); c=0 = z=0."""
    fig = make_subplots(1, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    for col, t in enumerate(TASKS, 1):
        runs = grid_runs(G, t)
        if not runs:
            continue
        ep0 = next(iter(runs.values()))
        p0 = ep0["success"][ep0["ep_z"] == 0].mean()
        for name, (w, axes, color) in GROUPS.items():
            if w not in runs:
                continue
            ep = runs[w]
            cs = np.unique(ep["scale"][ep["ep_z"] > 0])
            S = np.array([[ep["success"][(ep["dir"] == k) & (ep["scale"] == c)].mean() for c in cs] for k in axes])
            i0 = np.searchsorted(cs, 0)
            x, S = np.insert(cs, i0, 0), np.insert(S, i0, p0, axis=1)
            fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[S.max(0), S.min(0)[::-1]], fill="toself", fillcolor=color,
                            opacity=0.12, line=dict(width=0), legendgroup=name, showlegend=False, hoverinfo="skip",
                            row=1, col=col)
            fig.add_scatter(x=x, y=S.mean(0), mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8),
                            name=name, legendgroup=name, showlegend=col == 1, row=1, col=col,
                            hovertemplate=f"{name} c=%{{x}}: mean success %{{y:.3f}} ({len(axes)} axes × 8 inits)<extra></extra>")
            rep.setdefault("grid", {}).setdefault(f"t{t}", {})[name] = dict(zip(map(str, x), S.mean(0).round(3).tolist()))
        rep["grid"][f"t{t}"]["zero"] = float(p0)
        if len(runs) > 1:  # every run contains the same z=0 episodes: bitwise reproducibility check
            s0 = [e["steps"][e["ep_z"] == 0] for e in runs.values()]
            rep["grid"][f"t{t}"]["zero_identical"] = all(np.array_equal(s0[0], s) for s in s0[1:])
        fig.add_scatter(x=[0], y=[p0], mode="markers", marker=dict(color=ZERO, size=11, symbol="x"), name="z = 0",
                        legendgroup="z0", showlegend=col == 1, row=1, col=col, hovertemplate="z=0: %{y:.3f}<extra></extra>")
    fig.update_xaxes(title_text="axis scale c (z = c·√8·e_k)")
    fig.update_yaxes(title_text="success, inits 0–7", range=[-0.05, 1.05], row=1, col=1)
    layout(fig, height=400, title="Closed-loop grid: success vs scale (band = min–max over axes)")
    save(fig, out, "grid")


def fig_grid_heat(G, out):
    """Per-axis success, coupled | split | γ·x̂ (instance, BN) side by side; buttons select the task."""
    fig = make_subplots(1, 3, subplot_titles=["coupled", "split (axes 0–3 γ, 4–7 β)", "split, γ on x̂ (γ axes)"],
                        horizontal_spacing=0.07)
    vis = []
    for t in TASKS:
        runs = grid_runs(G, t)
        for col, ws in enumerate((["random"], ["split"], ["split-in", "split-bn"]), 1):
            if not all(w in runs for w in ws):
                continue
            S, ylab = [], []
            for w in ws:
                ep = runs[w]
                cs = np.unique(ep["scale"][ep["ep_z"] > 0])
                K = len(np.unique(ep["dir"][ep["ep_z"] > 0]))
                S += [[ep["success"][(ep["dir"] == k) & (ep["scale"] == c)].mean() for c in cs] for k in range(K)]
                ylab += [f"e{k}" if w == "random" else f"e{k} ({'γ' if k < 4 else 'β'})" if w == "split"
                         else f"e{k} γ, {w[6:]}" for k in range(K)]
            fig.add_heatmap(z=S, x=[f"{c:+g}" for c in cs], y=ylab, zmin=0, zmax=1, colorscale="Blues", showscale=col == 1,
                            visible=t == TASKS[0], row=1, col=col, colorbar=dict(title="success"),
                            hovertemplate="%{y}, c %{x}: success %{z:.2f}<extra></extra>")
            vis.append(t)
    if not vis:
        return
    buttons = [dict(label=f"t{t}", method="update", args=[{"visible": [v == t for v in vis]}]) for t in TASKS]
    fig.update_xaxes(title_text="axis scale c")
    layout(fig, height=420, title="Per-axis grid success (8 inits per cell)", margin=dict(l=80, r=20, t=100, b=50),
           updatemenus=[dict(buttons=buttons, type="buttons", direction="right", x=0, y=1.25, xanchor="left")])
    fig.update_yaxes(autorange="reversed")
    save(fig, out, "grid_heat")


def fig_effect(G, out, rep):
    """Paired with z=0 of the same init: eef-path change (RMS cm over fixed steps) and outcome flips."""
    fig = make_subplots(2, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_xaxes=True, vertical_spacing=0.1)
    for col, t in enumerate(TASKS, 1):
        runs = grid_runs(G, t)
        if not runs:
            continue
        for name, (w, axes, color) in GROUPS.items():
            if w not in runs:
                continue
            ep = runs[w]
            zero = zero_eps(G, t, ep)
            s0 = {r: zero["success"][(zero["ep_init"] == r) & (zero["ep_rep"] == 0)][0] for r in np.unique(zero["ep_init"])}
            xs, eff, flip = [], [], []
            for c in np.unique(ep["scale"][ep["ep_z"] > 0]):
                g = group_eps(ep, axes, c)
                m = analyze.metrics(g, zero)
                xs.append(c), eff.append(m["effect_rms_cm"])
                flip.append(np.mean([s != s0[r] for s, r in zip(g["success"], g["ep_init"])]))
                noise = m["noise_rms_cm"]
            fig.add_scatter(x=xs, y=eff, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8), name=name,
                            legendgroup=name, showlegend=col == 1, row=1, col=col,
                            hovertemplate=f"{name} c=%{{x}}: eef change %{{y:.2f}} cm<extra></extra>")
            fig.add_scatter(x=xs, y=flip, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8), name=name,
                            legendgroup=name, showlegend=False, row=2, col=col,
                            hovertemplate=f"{name} c=%{{x}}: outcome differs from z=0 in %{{y:.2f}} of episodes<extra></extra>")
            rep.setdefault("effect", {}).setdefault(f"t{t}", {})[name] = {
                "c": [float(c) for c in xs], "eef_cm": np.round(eff, 2).tolist(), "flip": np.round(flip, 3).tolist()}
        if np.isfinite(noise):
            fig.add_hline(y=noise, line=dict(color=ZERO, dash="dash"), row=1, col=col,
                          annotation_text="z=0, other flow noise", annotation_position="top left")
            nf = np.mean([zero["success"][i] != s0[zero["ep_init"][i]] for i in np.flatnonzero(zero["ep_rep"] > 0)])
            fig.add_hline(y=nf, line=dict(color=ZERO, dash="dash"), row=2, col=col)
            rep["effect"][f"t{t}"]["noise"] = {"eef_cm": round(float(noise), 2), "flip": round(float(nf), 3)}
    fig.update_xaxes(title_text="axis scale c", row=2)
    fig.update_yaxes(title_text="eef path change vs z=0, RMS cm", row=1, col=1)
    fig.update_yaxes(title_text="outcome flip rate vs z=0", range=[-0.02, 1], row=2, col=1)
    layout(fig, height=580, title="Closed-loop effect size, paired with z=0 on the same init and flow noise")
    save(fig, out, "effect")


def held_runs(G, T, t):
    """{key: rec} with the run.py held-out of each grid pick plus z=0 from the ticket film8 arm (same pairs)."""
    z = pt.zero_held(pt.load(T, ("film8",)), t)
    recs = {"zero": z} if z else {}
    for key in HELD:
        d = G / f"held-t{t}-{key}"
        if not (d / "DONE").exists():
            continue
        ep = analyze.load(d)
        o = np.lexsort((ep["ep_rep"], ep["ep_init"]))  # ticket order: init, then rep
        pick = [l for l in (G / "logs" / "pick.log").read_text().splitlines() if f"best-t{t}-{key}.npy" in l]
        recs[key] = {"init": ep["ep_init"][o].tolist(), "success": ep["success"][o].tolist(),
                     "rate": float(ep["success"].mean()), "pick": pick[-1].split(": ", 1)[1] if pick else ""}
    return recs


def fig_tab_held(G, T, out, rep):
    fig, rows = go.Figure(), []
    for t in TASKS:
        recs = held_runs(G, T, t)
        z = recs.get("zero")
        for key, (name, color) in [("zero", ("z = 0", ZERO))] + list(HELD.items()):
            if key not in recs:
                continue
            h = recs[key]
            p, lo, hi = wilson(sum(h["success"]), len(h["success"]))
            fig.add_scatter(x=[[f"t{t}"], [name]], y=[p], error_y=dict(type="data", symmetric=False, array=[hi - p],
                            arrayminus=[p - lo]), mode="markers", marker=dict(color=color, size=11), name=name,
                            legendgroup=key, showlegend=t == TASKS[0], hovertemplate=f"t{t} {name}: %{{y:.3f}} [{lo:.2f}, {hi:.2f}]<extra></extra>")
            if key == "zero":
                continue
            w, l, pv = pt.paired(h, z) if z else (0, 0, np.nan)
            rows.append([f"t{t}", name, h["pick"], f"{p:.2f} [{lo:.2f}, {hi:.2f}]", f"{z['rate']:.2f}" if z else "–",
                         f"{w}/{l}, p={pv:.2g}"])
            rep.setdefault("held", {}).setdefault(f"t{t}", {})[key] = {"rate": p, "wins": w, "losses": l, "p": pv,
                                                                       "zero": z and z["rate"]}
    layout(fig, height=400, yaxis=dict(title="held-out success (inits 10–39 × 2), 95% CI", range=[-0.05, 1.05]),
           title="Held-out: best grid z per projection (split: best γ axis and best β axis separately)")
    save(fig, out, "grid_held")
    head = ["task", "pick", "grid z (row, axis, scale, grid rate)", "held-out [95% CI]", "z=0 held-out", "wins/losses vs z=0, sign test"]
    (out / "tab_grid_held.html").write_text("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
                                            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def tab_cem_z(data, out, rep):
    """Where each chain's best z lies: |z| on the γ half vs the β half, and cand 0 (z=0) reproducibility vs coupled."""
    rows = []
    for t in pt.TASKS:
        for a in ("film8", "film8split"):
            for r, c in data.get((t, a), {}).get("chains", {}).items():
                best = max(c, key=score)
                z = np.array(best["z"])
                same = ""
                if a == "film8split" and (ref := data.get((t, "film8"), {}).get("chains", {}).get(r)):
                    same = "yes" if ref[0]["success"] == c[0]["success"] and ref[0]["steps"] == c[0]["steps"] else "no"
                h = data[t, a]["held"].get(f"r{r}")
                rows.append([f"t{t}", pt.NAME[a], r, best["i"], f"{best['rate']:.2f}", f"{np.linalg.norm(z[:4]):.2f}",
                             f"{np.linalg.norm(z[4:]):.2f}", f"{h['rate']:.2f}" if h else "–", same])
                rep.setdefault("cem_z", []).append({"task": t, "arm": a, "rep": r, "zg": float(np.linalg.norm(z[:4])),
                                                    "zb": float(np.linalg.norm(z[4:])), "held": h and h["rate"], "cand0_same": same})
    head = ["task", "arm", "rep", "best cand", "in-sample", "‖z[0:4]‖ (split: γ)", "‖z[4:8]‖ (split: β)", "held-out",
            "cand 0 identical to coupled"]
    (out / "tab_cem_z.html").write_text("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
                                        "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("G", type=Path)
    parser.add_argument("T", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--page", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    fig_openloop(args.G, args.out, rep)
    fig_grid(args.G, args.out, rep)
    fig_grid_heat(args.G, args.out)
    fig_effect(args.G, args.out, rep)
    fig_tab_held(args.G, args.T, args.out, rep)
    data = pt.load(args.T, ARMS)
    pt.fig_search(data, args.out, ARMS)
    pt.fig_heldout(data, {}, args.out, ARMS)
    pt.tab_summary(data, {}, args.out, ARMS)
    tab_cem_z(data, args.out, rep)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    if args.template:
        page(args.template, args.out, args.page)


if __name__ == "__main__":
    main()
