"""Figures + tables for docs/results/smolvla_film_ablation_object.html: FiLM z dim and layer position × count ablation.

G = experiments/libero/film_ablation/local.sh outputs, F = film_split outputs (R′ d8 grids), T = research.ticket root (z=0 held-out).
Axes are d16 split names: γ 0-7, β 8-15 (R′ d8 = γ 0-3, β 8-11). Variant keys: {f,m,l,s}{#layers}; s4 = R′.
python -m research.plot_film_ablation out/libero/smolvla/film_ablation out/libero/smolvla/film_split out/ticket-local \
    --out out/film-ablation-figs --template research/film_ablation_report.html --page docs/results/smolvla_film_ablation_object.html
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots

from research import plot_ticket as pt
from research.plot_icl import page
from research.plot_steer2 import wilson

TASKS = (0, 7)
POS = {"front": "f", "mid": "m", "late": "l", "spread": "s"}
COUNTS = (1, 2, 4, 16)
LAYERS = {"f1": "3", "f2": "3,5", "f4": "1,3,5,7", "m1": "15", "m2": "15,17", "m4": "13,15,17,19", "l1": "27",
          "l2": "27,29", "l4": "25,27,29,31", "s1": "31", "s2": "15,31", "s4": "7,15,23,31", "s16": "1,3,…,31 (all CA)"}
ALPHA = {1: 0.6, 2: 0.4243, 4: 0.3, 16: 0.15}
COLOR = {"front": "#2a78d6", "mid": "#eb6834", "late": "#1baf7a", "spread": "#eda100"}  # reference categorical slots 1-4
TYPE = {"γ": (range(8), "solid"), "β": (range(8, 16), "dash")}
ZERO = "#52514e"
SYMBOL = {1: "circle", 2: "square", 4: "diamond", 16: "triangle-up"}


def layout(fig, **kw):
    fig.update_layout(**{**pt.LAYOUT, **kw})


def variants():
    """(position, #layers, key) for every layer variant that has a design (16 layers only as spread)."""
    return [(p, n, f"{s}{n}") for p, s in POS.items() for n in COUNTS if f"{s}{n}" in LAYERS]


def cells(d):
    """Axis-grid cells [{a, c, s}] (s = success per init, init order) and z=0 success of one run.py grid."""
    r = np.load(d / "results.npz")
    D, out = r["Z"].shape[1], []
    for k in range(1, len(r["Z"])):
        m = r["ep_z"] == k
        a = int(np.abs(r["Z"][k]).argmax())
        out.append({"a": a if a < D // 2 else a - D // 2 + 8, "c": round(float(r["Z"][k][a] / D**0.5), 3),
                    "s": r["success"][m][np.argsort(r["ep_init"][m])]})
    m = r["ep_z"] == 0
    return out, r["success"][m][np.argsort(r["ep_init"][m])], r["steps"][m][np.argsort(r["ep_init"][m])]


def done(d):
    return (d / "DONE").exists()


def grid_cells(G, F, t, v):
    """Cells of variant v on task t (None if a run is missing). s4 = R′ from film_split: γ·x̂ axes from split-in,
    β axes from split (β is the same with or without centring); d16 = R′ + the 8 new axes."""
    if v in ("s4", "d16"):
        if not (done(F / f"grid-t{t}-split-in") and done(F / f"grid-t{t}-split")):
            return None
        g, z0, _ = cells(F / f"grid-t{t}-split-in")
        b = [c for c in cells(F / f"grid-t{t}-split")[0] if c["a"] >= 8]
        cs = [c for c in g if c["a"] < 8] + b
        if v == "d16":
            if not done(G / f"grid-t{t}-d16"):
                return None
            cs += cells(G / f"grid-t{t}-d16")[0]
        return cs, z0
    d = G / f"grid-t{t}-{v}"
    return cells(d)[:2] if done(d) else None


def axis_best(cs, axes):
    """{axis: best cell rate} over the axis's 6 scales."""
    best = {}
    for c in cs:
        if c["a"] in axes:
            best[c["a"]] = max(best.get(c["a"], 0), c["s"].mean())
    return best


def dose(cs, z0, axes):
    """(c values incl. 0, mean success over the axes at each c)."""
    xs = sorted({c["c"] for c in cs if c["a"] in axes})
    ys = [np.mean([c["s"].mean() for c in cs if c["a"] in axes and c["c"] == x]) for x in xs]
    i = np.searchsorted(xs, 0)
    return np.insert(xs, i, 0), np.insert(ys, i, z0.mean())


def picks(G, t):
    f = G / f"picks-t{t}.txt"
    return dict(l.split() for l in f.read_text().splitlines()) if f.exists() else {}


def held(G, t, name):
    """Held-out record of a pick name in ticket order (init, then rep), or None."""
    d = picks(G, t).get(name)
    if not d or not done(G / d):
        return None
    r = np.load(G / d / "results.npz")
    o = np.lexsort((r["ep_rep"], r["ep_init"]))
    return {"init": r["ep_init"][o].tolist(), "success": r["success"][o].tolist(), "rate": float(r["success"].mean()),
            "key": d.split("-", 3)[-1]}


def ol(G, t, v):
    """(openloop npz, {d16 axis: column}) of a variant; s4 and d16 share the d16 run."""
    f = G / f"openloop-t{t}-{'d16' if v in ('s4', 'd16') else v}" / "openloop.npz"
    if not f.exists():
        return None, {}
    d = np.load(f)
    K = d["dose"].shape[1]
    cols = {a: a for a in range(16)} if K == 16 else {a if a < 4 else a + 4: a for a in range(8)}
    if v == "s4":
        cols = {a: i for a, i in cols.items() if a % 8 < 4}
    return d, cols


def gain(d, cols, axes, scale):
    """Mean over axes of the RMS first-action change at an open-loop scale (normalized units)."""
    j = list(d["scales"]).index(scale)
    k = [i for a, i in cols.items() if a in axes]
    A = d["dose"].shape[-1]
    return float((np.linalg.norm(d["dose"][:, k, j, 0, :], axis=-1).mean(0) / np.sqrt(A)).mean())


def eff_rank(d, cols, axes=range(16)):
    """Participation ratio of the singular values of the chunk Jacobian ∂a/∂z on the chosen columns, mean over inputs."""
    J = d["jac"][..., [i for a, i in cols.items() if a in axes]]
    sv = np.linalg.svd(J.reshape(J.shape[0], -1, J.shape[-1]), compute_uv=False)
    return float(((sv**2).sum(1) ** 2 / (sv**4).sum(1)).mean())


def held_stats(h, z):
    p, lo, hi = wilson(sum(h["success"]), len(h["success"]))
    w, l, pv = pt.paired(h, z) if z else (0, 0, np.nan)
    return p, lo, hi, w, l, pv


def fig_openloop(G, out, rep):
    """First-action RMS Δa at c = 1 per variant: x = #layers, one line per position, γ solid / β dashed."""
    fig = make_subplots(1, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    for col, t in enumerate(TASKS, 1):
        nsd = None
        for p in POS:
            for ty, (axes, dash) in TYPE.items():
                xs, ys = [], []
                for pp, n, v in variants():
                    d, cols = ol(G, t, v)
                    if pp == p and d is not None:
                        xs.append(str(n)), ys.append(gain(d, cols, axes, 1.0))
                        nsd = float(d["noise_a"][..., 0, :].std(1).mean())
                        rep.setdefault("openloop", {}).setdefault(f"t{t}", {}).setdefault(v, {})[ty] = round(ys[-1], 4)
                if xs:
                    fig.add_scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=COLOR[p], width=2, dash=dash),
                                    marker=dict(size=8), name=f"{p}, {ty} axes", legendgroup=f"{p}{ty}", showlegend=col == 1,
                                    row=1, col=col, hovertemplate=f"{p} %{{x}} layers, {ty}: RMS Δa0 %{{y:.3f}}<extra></extra>")
        if nsd:
            fig.add_hline(y=nsd, line=dict(color=ZERO, dash="dot"), row=1, col=col,
                          annotation_text="flow-noise sd at z=0", annotation_position="top left")
    fig.update_xaxes(title_text="FiLM layers", type="category", categoryorder="array", categoryarray=[str(n) for n in COUNTS])
    fig.update_yaxes(title_text="RMS Δa of first action at c = 1", rangemode="tozero", row=1, col=1)
    layout(fig, height=420, title="Open loop: action change per axis at c = 1 (96 saved inputs, fixed flow noise)")
    pt.save(fig, out, "openloop")


def fig_dose(G, F, out, rep):
    """Grid success vs c, mean over axes; rows γ / β, columns position, one line per #layers; buttons select the task."""
    fig = make_subplots(2, 4, subplot_titles=[f"{p}, {ty} axes" for ty in TYPE for p in POS], shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.03)
    vis = []
    for t in TASKS:
        for row, (ty, (axes, _)) in enumerate(TYPE.items(), 1):
            for col, p in enumerate(POS, 1):
                for pp, n, v in variants():
                    g = grid_cells(G, F, t, v) if pp == p else None
                    if g is None:
                        continue
                    x, y = dose(g[0], g[1], [a for a in axes if a % 8 < 4])
                    fig.add_scatter(x=x, y=y, mode="lines+markers", line=dict(width=2, color=COLOR[p]),
                                    marker=dict(size=8, symbol=SYMBOL[n]), opacity=0.4 + 0.15 * COUNTS.index(n),
                                    name=f"{n} layer{'s' * (n > 1)}", legendgroup=str(n), showlegend=row == col == 1,
                                    visible=t == TASKS[0], row=row, col=col,
                                    hovertemplate=f"{v} ({LAYERS[v]}), {ty}, c=%{{x}}: success %{{y:.2f}}<extra></extra>")
                    vis.append(t)
                    rep.setdefault("dose", {}).setdefault(f"t{t}", {}).setdefault(v, {})[ty] = dict(
                        zip(map(str, x), np.round(y, 3).tolist()))
    if not vis:
        return
    buttons = [dict(label=f"t{t}", method="update", args=[{"visible": [x == t for x in vis]}]) for t in TASKS]
    fig.update_xaxes(title_text="c", row=2)
    fig.update_yaxes(range=[-0.05, 1.05])
    fig.update_yaxes(title_text="success, inits 0–7", col=1)
    layout(fig, height=620, title="Grid success vs scale c (mean over 4 axes; c = 0 is z = 0)", margin=dict(l=60, r=20, t=110, b=50),
           updatemenus=[dict(buttons=buttons, type="buttons", direction="right", x=1, y=1.16, xanchor="right")])
    pt.save(fig, out, "dose")


def fig_factorial(G, F, T, out, rep):
    """Two figures over (position × #layers): mean grid success at |c| = 0.5, and held-out of the best γ / β pick."""
    for name, title in (("mean05", "Grid: mean success over all 8 axes at c = ±0.5"),
                        ("heldfac", "Held-out success of the best γ pick and the best β pick (n = 60, 95% CI)")):
        rows = ("γ", "β") if name == "heldfac" else ("all",)
        fig = make_subplots(len(rows), len(TASKS), subplot_titles=[f"t{t}, {r} pick" if name == "heldfac" else f"t{t}"
                                                                    for r in rows for t in TASKS],
                            shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.12)
        for col, t in enumerate(TASKS, 1):
            z = pt.zero_held(pt.load(T, ("film8",)), t)
            for row, ty in enumerate(rows, 1):
                ref = None
                for p in POS:
                    xs, ys, lo, hi, tips = [], [], [], [], []
                    for pp, n, v in variants():
                        if pp != p:
                            continue
                        if name == "mean05":
                            g = grid_cells(G, F, t, v)
                            if g is None:
                                continue
                            y = np.mean([c["s"].mean() for c in g[0] if abs(c["c"]) == 0.5 and c["a"] % 8 < 4])
                            ref = g[1].mean()
                            xs.append(str(n)), ys.append(y), lo.append(y), hi.append(y), tips.append(v)
                        else:
                            h = held(G, t, {"s4": "d8"}.get(v, v) + ("-gamma" if ty == "γ" else "-beta"))
                            if h is None:
                                continue
                            pr, l_, h_, w, l, pv = held_stats(h, z)
                            xs.append(str(n)), ys.append(pr), lo.append(l_), hi.append(h_)
                            tips.append(f"{v} {h['key']}: {w}/{l} vs z=0, p={pv:.2g}")
                    if xs:
                        fig.add_scatter(x=xs, y=ys, error_y=dict(type="data", symmetric=False, array=np.subtract(hi, ys),
                                                                 arrayminus=np.subtract(ys, lo), visible=name == "heldfac"),
                                        mode="lines+markers", line=dict(color=COLOR[p], width=2), marker=dict(size=9),
                                        name=p, legendgroup=p, showlegend=row == col == 1, row=row, col=col, text=tips,
                                        hovertemplate="%{text}: %{y:.2f}<extra></extra>")
                        rep.setdefault(name, {}).setdefault(f"t{t}", {}).setdefault(ty, {})[p] = dict(zip(xs, np.round(ys, 3).tolist()))
                ref = z["rate"] if name == "heldfac" and z else ref
                if ref is not None:
                    fig.add_hline(y=ref, line=dict(color=ZERO, dash="dot"), row=row, col=col,
                                  annotation_text="z = 0", annotation_position="bottom right")
        fig.update_xaxes(title_text="FiLM layers", type="category", categoryorder="array",
                         categoryarray=[str(n) for n in COUNTS], row=len(rows))
        fig.update_yaxes(range=[-0.05, 1.05])
        layout(fig, height=330 * len(rows) + 90, title=title)
        pt.save(fig, out, name)


def fig_dim(G, F, T, out, rep):
    """d ablation on R′ layers: k axes per type (d = 2k). Grid best of the first k axes, mean best of every k-subset of the
    8 axes, and held-out of the first-k pick."""
    fig = make_subplots(1, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    K = (1, 2, 4, 8)
    sub = {1: "d1", 2: "d4", 4: "d8", 8: "d16"}
    for col, t in enumerate(TASKS, 1):
        g = grid_cells(G, F, t, "d16")
        z = pt.zero_held(pt.load(T, ("film8",)), t)
        for ty, (axes, dash) in TYPE.items():
            color = "#2a78d6" if ty == "γ" else "#eb6834"
            if g is not None:
                best = axis_best(g[0], axes)
                order = sorted(best)
                first = [max(best[a] for a in order[:k]) for k in K]
                mean = [np.mean([max(best[a] for a in s) for s in itertools.combinations(order, k)]) for k in K]
                fig.add_scatter(x=list(K), y=first, mode="lines+markers", line=dict(color=color, width=2, dash="dot"),
                                marker=dict(size=8), name=f"{ty}: grid best of first k axes", row=1, col=col, showlegend=col == 1,
                                legendgroup=f"{ty}g", hovertemplate=f"{ty} k=%{{x}}: grid best %{{y:.2f}}<extra></extra>")
                fig.add_scatter(x=list(K), y=mean, mode="lines", line=dict(color=color, width=1, dash="dash"),
                                name=f"{ty}: grid best, mean over k-subsets", row=1, col=col, showlegend=col == 1,
                                legendgroup=f"{ty}m", hovertemplate=f"{ty} k=%{{x}}: mean best %{{y:.2f}}<extra></extra>")
                rep.setdefault("dim", {}).setdefault(f"t{t}", {})[ty] = {"first": first, "subset_mean": mean}
            hs = [(k, held(G, t, f"{sub[k]}-{'gamma' if ty == 'γ' else 'beta'}")) for k in K]
            hs = [(k, h) for k, h in hs if h]
            if hs:
                st = [held_stats(h, z) for _, h in hs]
                fig.add_scatter(x=[k for k, _ in hs], y=[s[0] for s in st], mode="lines+markers",
                                error_y=dict(type="data", symmetric=False, array=[s[2] - s[0] for s in st],
                                             arrayminus=[s[0] - s[1] for s in st]),
                                line=dict(color=color, width=2), marker=dict(size=9), name=f"{ty}: held-out of the pick",
                                row=1, col=col, showlegend=col == 1, legendgroup=f"{ty}h",
                                text=[f"{h['key']}, {s[3]}/{s[4]} vs z=0" for (_, h), s in zip(hs, st)],
                                hovertemplate=f"{ty} k=%{{x}} %{{text}}: held-out %{{y:.2f}}<extra></extra>")
                rep.setdefault("dim", {}).setdefault(f"t{t}", {})[f"{ty}_held"] = {k: h["rate"] for k, h in hs}
        if z:
            fig.add_hline(y=z["rate"], line=dict(color=ZERO, dash="dot"), row=1, col=col, annotation_text="z = 0 held-out",
                          annotation_position="bottom right")
    fig.update_xaxes(title_text="axes per type k (d = 2k)", type="log", tickvals=list(K))
    fig.update_yaxes(title_text="success", range=[-0.05, 1.05], row=1, col=1)
    layout(fig, height=440, title="z dimension on layers 7, 15, 23, 31: grid best and held-out vs number of axes")
    pt.save(fig, out, "dim")


def fig_combo(G, F, out, rep):
    """5 × 5 grid in the plane of the best γ and best β axis: off-axis cells from the combo run, axes from the d16 grids."""
    fig = make_subplots(1, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], horizontal_spacing=0.1)
    vals = (-1, -0.5, 0, 0.5, 1)
    for col, t in enumerate(TASKS, 1):
        d, g = G / f"grid-t{t}-combo", grid_cells(G, F, t, "d16")
        if not done(d) or g is None:
            continue
        r = np.load(d / "results.npz")
        ga, ba = (np.flatnonzero(r["Z"][1:].any(0)))
        S = np.full((5, 5), np.nan)
        S[2, 2] = g[1].mean()
        for k in range(1, len(r["Z"])):
            S[vals.index(r["Z"][k][ba] / 4), vals.index(r["Z"][k][ga] / 4)] = r["success"][r["ep_z"] == k].mean()
        for c in g[0]:
            if c["c"] in vals and c["a"] in (ga, ba):
                i = vals.index(c["c"])
                S[(i, 2) if c["a"] == ba else (2, i)] = c["s"].mean()
        fig.add_heatmap(z=S, x=[f"{v:+g}" for v in vals], y=[f"{v:+g}" for v in vals], zmin=0, zmax=1, colorscale="Blues",
                        showscale=col == 1, colorbar=dict(title="success"), row=1, col=col,
                        hovertemplate=f"γ axis {ga} c=%{{x}}, β axis {ba} c=%{{y}}: success %{{z:.2f}}<extra></extra>")
        fig.update_xaxes(title_text=f"c on γ axis {ga}", row=1, col=col)
        fig.update_yaxes(title_text=f"c on β axis {ba}", row=1, col=col)
        rep.setdefault("combo", {})[f"t{t}"] = {"axes": [int(ga), int(ba)], "S": np.round(S, 3).tolist()}
    layout(fig, height=420, title="2-D grid: best γ axis × best β axis (centre = z = 0; 8 inits per cell)")
    pt.save(fig, out, "combo")


def fig_gain_success(G, F, out, rep):
    """Post-hoc check of step size: grid success at |c| (mean over signs and axes) vs measured open-loop gain at |c|."""
    fig = make_subplots(1, len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    for col, t in enumerate(TASKS, 1):
        for p, n, v in variants():
            g, (d, cols) = grid_cells(G, F, t, v), ol(G, t, v)
            if g is None or d is None:
                continue
            axes = [a for a in range(16) if a % 8 < 4]
            xs = [0] + [gain(d, cols, axes, c) for c in (0.5, 1.0, 2.0)]
            ys = [g[1].mean()] + [np.mean([c["s"].mean() for c in g[0] if abs(c["c"]) == m]) for m in (0.5, 1.0, 2.0)]
            fig.add_scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=COLOR[p], width=1.5),
                            marker=dict(size=8, symbol=SYMBOL[n]), name=f"{v} ({LAYERS[v]})", legendgroup=v,
                            showlegend=col == 1, row=1, col=col, text=["z=0", "|c|=0.5", "|c|=1", "|c|=2"],
                            hovertemplate=f"{v} %{{text}}: gain %{{x:.3f}}, success %{{y:.2f}}<extra></extra>")
    fig.update_xaxes(title_text="measured RMS Δa of first action at |c| (open loop)")
    fig.update_yaxes(title_text="grid success, mean over axes and signs", range=[-0.05, 1.05], row=1, col=1)
    layout(fig, height=480, title="Success vs measured action change: do the variants collapse onto one curve?",
           legend=dict(orientation="v", x=1.02, y=1))
    pt.save(fig, out, "gain_success")


def table(out, name, head, rows):
    (out / f"{name}.html").write_text("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
                                      "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def fmt_held(h, z):
    if not h:
        return "–"
    p, lo, hi, w, l, pv = held_stats(h, z)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}] {h['key']}, {w}/{l} p={pv:.2g}"


def tab_variants(G, F, T, out, rep):
    rows = []
    for t in TASKS:
        z = pt.zero_held(pt.load(T, ("film8",)), t)
        for p, n, v in variants():
            g, (d, cols) = grid_cells(G, F, t, v), ol(G, t, v)
            r = [f"t{t}", v, LAYERS[v], ALPHA[n]]
            r += [f"{gain(d, cols, TYPE[ty][0], 1.0):.3f}" if d is not None else "–" for ty in TYPE]
            r += [f"{eff_rank(d, cols):.2f}" if d is not None else "–"]
            if g:
                ax = [a for a in range(16) if a % 8 < 4]
                b = axis_best(g[0], ax)
                m05 = [c["s"].mean() for c in g[0] if abs(c["c"]) == 0.5 and c["a"] in ax]
                r += [f"{g[1].mean():.2f}", f"{np.mean(m05):.2f}", f"{np.mean(np.array(m05) > g[1].mean()):.2f}",
                      f"{max(b[a] for a in b if a < 8):.2f}", f"{max(b[a] for a in b if a >= 8):.2f}"]
            else:
                r += ["–"] * 5
            name = {"s4": "d8"}.get(v, v)
            r += [fmt_held(held(G, t, f"{name}-{k}"), z) for k in ("gamma", "beta")]
            rows.append(r)
    table(out, "tab_variants", ["task", "variant", "layers", "α", "gain γ c=1", "gain β c=1", "eff. rank ∂a/∂z",
                                "z=0 grid", "mean c=±0.5", "share c=±0.5 > z=0", "best γ cell", "best β cell",
                                "held-out best γ [95% CI] pick, W/L vs z=0", "held-out best β"], rows)


def tab_dim(G, F, T, out, rep):
    rows = []
    for t in TASKS:
        z = pt.zero_held(pt.load(T, ("film8",)), t)
        d, cols = ol(G, t, "d16")
        for k, s in ((1, "d1"), (2, "d4"), (4, "d8"), (8, "d16")):
            axes = [a for a in range(16) if a % 8 < k]
            er = f"{eff_rank(d, cols, axes):.2f}" if d is not None else "–"
            erg = f"{eff_rank(d, cols, [a for a in axes if a < 8]):.2f}" if d is not None and k > 1 else "1"
            rows.append([f"t{t}", 2 * k, er, erg] + [fmt_held(held(G, t, f"{s}-{ty}"), z) for ty in ("gamma", "beta")])
            rep.setdefault("rank", {}).setdefault(f"t{t}", {})[2 * k] = er
    table(out, "tab_dim", ["task", "d", "eff. rank ∂a/∂z (all d columns)", "eff. rank, γ columns only",
                           "held-out best γ pick", "held-out best β pick"], rows)


def checks(G, F, rep):
    """z=0 episodes of every grid match the film_split grids bitwise (same seeds; pairing holds)."""
    for t in TASKS:
        ref = cells(F / f"grid-t{t}-split-in")[2] if done(F / f"grid-t{t}-split-in") else None
        runs = [d for d in G.glob(f"grid-t{t}-*") if done(d)]
        rep.setdefault("zero_identical", {})[f"t{t}"] = ref is not None and all(np.array_equal(cells(d)[2], ref) for d in runs)
        rep.setdefault("n_grids", {})[f"t{t}"] = len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("G", type=Path)
    parser.add_argument("F", type=Path)
    parser.add_argument("T", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--page", type=Path)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    checks(a.G, a.F, rep)
    fig_openloop(a.G, a.out, rep)
    fig_dose(a.G, a.F, a.out, rep)
    fig_factorial(a.G, a.F, a.T, a.out, rep)
    fig_dim(a.G, a.F, a.T, a.out, rep)
    fig_combo(a.G, a.F, a.out, rep)
    fig_gain_success(a.G, a.F, a.out, rep)
    tab_variants(a.G, a.F, a.T, a.out, rep)
    tab_dim(a.G, a.F, a.T, a.out, rep)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    if a.template:
        page(a.template, a.out, a.page)


if __name__ == "__main__":
    main()
