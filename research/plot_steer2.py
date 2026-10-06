"""Figures + tables for docs/results/smolvla_steer2_object.html (steering round 2, all plotly).

ROOT/t{T}/{grid,line,val,openloop}-{W}/ from the steer2 lines files; W in random-s0..2, pca.
Writes ROOT/report.json (numbers for the text), OUT_DIR/*.html figure/table fragments (div only) and,
with --template, the page with fragments inlined (it loads plotly.min.js next to it). Missing inputs skip a figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binomtest, wilcoxon

from research import analyze

TASKS = (0, 7, 9, 4, 2)
WS = ("random-s0", "random-s1", "random-s2", "pca")
COL = {"random-s0": "#2a78d6", "random-s1": "#5a9be6", "random-s2": "#8cbcf0", "pca": "#eb6834", "zero": "#52514e"}
DASH = {"random-s0": "solid", "random-s1": "dash", "random-s2": "dot", "pca": "solid"}
EMPTY_CLOSE = 0.01  # finger width [m] below which a closed gripper holds nothing
LAYOUT = dict(template="plotly_white", font=dict(size=12), margin=dict(l=60, r=20, t=50, b=50),
              legend=dict(orientation="h", y=-0.15))


def load(d):
    """Done episodes of a research.run output dir (+ config), or None."""
    p = Path(d) / "results.npz"
    if not p.exists():
        return None
    r = dict(np.load(p))
    m = r["done"]
    ep = {k: r[k][m] for k in ("ep_z", "ep_init", "ep_rep", "success", "steps", "state")}
    ep.update(Z=r["Z"], scale=r["scale"][ep["ep_z"]], dir=r["direction"][ep["ep_z"]],
              row_scale=r["scale"], row_dir=r["direction"])
    ep["cfg"] = json.loads((Path(d) / "provenance.json").read_text())["config"]
    ep["path"] = Path(d)
    return ep


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def zero_of(ep):
    return {k: v[ep["ep_z"] == 0] if isinstance(v, np.ndarray) and len(v) == len(ep["ep_z"]) else v
            for k, v in ep.items()}


def pairs(run, row, zero):
    """[(init, rep, success z, success z=0, steps z, steps z=0)] for z row vs z=0 of the same (init, rep)."""
    z0 = {(i, r): (s, n) for i, r, s, n, z in zip(zero["ep_init"], zero["ep_rep"], zero["success"],
                                                  zero["steps"], zero["ep_z"]) if z == 0}
    return np.array([(i, r, s, *z0[(i, r)][:1], n, z0[(i, r)][1])
                     for i, r, s, n, z in zip(run["ep_init"], run["ep_rep"], run["success"], run["steps"], run["ep_z"])
                     if z == row and (i, r) in z0], dtype=float).reshape(-1, 6)


def save(fig, out, name, **layout):
    fig.update_layout(**(LAYOUT | layout))
    (out / f"{name}.html").write_text(fig.to_html(include_plotlyjs=False, full_html=False,
                                                  config={"displaylogo": False, "responsive": True}))
    print("figure", name)


def runs(root):
    R = {}
    for t in TASKS:
        for kind in ("grid", "line", "val"):
            for w in WS:
                ep = load(root / f"t{t}" / f"{kind}-{w}")
                if ep is not None:
                    R[(t, kind, w)] = ep
    return R


def grid_zero(R, t):
    return zero_of(R[(t, "grid", "random-s0")]) if (t, "grid", "random-s0") in R else None


# ---------- figures ----------

def fig_baseline(R, out, rep):
    fig = go.Figure()
    for kind, name, sym in (("grid", "inits 0–7 × 8 noise reps (grid)", "circle"),
                            ("val", "inits 10–39 × 2 noise reps (held-out)", "diamond")):
        xs, ys, lo, hi = [], [], [], []
        for t in TASKS:
            ep = R.get((t, kind, "random-s0"))
            if ep is None:
                continue
            s = ep["success"][ep["ep_z"] == 0]
            p, a, b = wilson(s.sum(), len(s))
            xs.append(f"t{t}"), ys.append(p), lo.append(p - a), hi.append(b - p)
            rep.setdefault("z0", {}).setdefault(f"t{t}", {})[kind] = [float(p), int(len(s))]
        fig.add_scatter(x=xs, y=ys, mode="markers", name=name, marker=dict(size=10, symbol=sym),
                        error_y=dict(type="data", array=hi, arrayminus=lo))
    save(fig, out, "baseline", title="z = 0 success per task (95% Wilson CI)",
         yaxis=dict(title="success rate", range=[0, 1.05]), xaxis_title="LIBERO-Object task", height=380)


def fig_openloop(root, out, rep):
    fig = make_subplots(rows=1, cols=len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    found = False
    for c, t in enumerate(TASKS, 1):
        for w in WS:
            p = root / f"t{t}" / f"openloop-{w}" / "openloop.npz"
            if not p.exists():
                continue
            found = True
            d = np.load(p)
            x = np.linalg.norm(d["dose"][..., 0, :], axis=-1) / np.sqrt(d["dose"].shape[-1])  # [N, K, S] RMS Δa0
            med = np.median(x.reshape(-1, x.shape[-1]), 0)
            noise = d["noise_a"][..., 0, :].std(1).mean()
            fig.add_scatter(x=d["scales"], y=med, name=w, legendgroup=w, showlegend=c == 1,
                            line=dict(color=COL[w], dash=DASH[w]), row=1, col=c)
            rep.setdefault("openloop", {}).setdefault(f"t{t}", {})[w] = {
                "median_da0": dict(zip(map(float, d["scales"]), med.round(4).tolist())), "noise_sd": float(noise)}
        if found:
            fig.add_hline(y=noise, line=dict(color=COL["zero"], dash="dash"), row=1, col=c,
                          annotation_text="noise sd" if c == 1 else None)
    if not found:
        return print("skip openloop")
    fig.update_xaxes(type="log", title_text="scale c")
    fig.update_yaxes(type="log", title_text="median RMS Δa₀ (normalized)", col=1)
    save(fig, out, "openloop", title="Open-loop first-action change vs scale (no simulation; dashed = z=0 noise sd)",
         height=380)


def grid_table(ep):
    """success[dir, scale] for one grid run, scales sorted."""
    scales = np.unique(ep["scale"][ep["ep_z"] > 0])
    dirs = np.unique(ep["dir"][ep["ep_z"] > 0])
    S = np.array([[ep["success"][(ep["dir"] == k) & (ep["scale"] == c)].mean() for c in scales] for k in dirs])
    return scales, dirs, S


def fig_grid(R, out, rep):
    fig = make_subplots(rows=1, cols=len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_yaxes=True)
    for c, t in enumerate(TASKS, 1):
        z0 = grid_zero(R, t)
        if z0 is None:
            continue
        p0 = z0["success"].mean()
        for w in WS:
            ep = R.get((t, "grid", w))
            if ep is None:
                continue
            scales, dirs, S = grid_table(ep)
            x = np.insert(scales, np.searchsorted(scales, 0), 0)
            for k, row in zip(dirs, S):
                fig.add_scatter(x=x, y=np.insert(row, np.searchsorted(scales, 0), p0), mode="lines",
                                line=dict(color=COL[w], width=0.7), opacity=0.35, name=f"{w} single directions",
                                legendgroup=f"{w}-d", showlegend=bool(c == 1 and k == dirs[0]), visible="legendonly",
                                row=1, col=c)
            fig.add_scatter(x=x, y=np.insert(S.mean(0), np.searchsorted(scales, 0), p0), mode="lines+markers",
                            name=f"{w} (mean of 8 directions)", legendgroup=w, showlegend=c == 1,
                            line=dict(color=COL[w], dash=DASH[w]), row=1, col=c)
            rep.setdefault("grid", {}).setdefault(f"t{t}", {})[w] = {
                "scales": scales.tolist(), "mean": S.mean(0).round(3).tolist(), "max": S.max(0).round(3).tolist(),
                "plus_beats_minus": int(((S[:, scales > 0] - S[:, scales < 0][:, ::-1]) > 0).sum()),
                "minus_beats_plus": int(((S[:, scales > 0] - S[:, scales < 0][:, ::-1]) < 0).sum())}
        fig.add_scatter(x=[0], y=[p0], mode="markers", marker=dict(color=COL["zero"], size=10, symbol="x"),
                        name="z = 0", legendgroup="z0", showlegend=c == 1, row=1, col=c)
    fig.update_xaxes(title_text="scale c")
    fig.update_yaxes(title_text="success rate", range=[0, 1.05], col=1)
    save(fig, out, "grid", title="Grid: success vs scale", height=440)


def fig_grid_heat(R, out):
    """Per-direction success, one heatmap per W; dropdown selects the task."""
    fig = make_subplots(rows=1, cols=len(WS), subplot_titles=WS, shared_yaxes=True, horizontal_spacing=0.03)
    vis = []
    for t in TASKS:
        for c, w in enumerate(WS, 1):
            ep = R.get((t, "grid", w))
            if ep is None:
                continue
            scales, dirs, S = grid_table(ep)
            fig.add_heatmap(z=S, x=[f"{s:+g}" for s in scales], y=[f"u{k}" for k in dirs], zmin=0, zmax=1,
                            colorscale="Blues", showscale=c == 1, visible=bool(t == TASKS[0]), row=1, col=c,
                            hovertemplate="dir %{y}, c %{x}: success %{z:.2f}<extra></extra>",
                            colorbar=dict(title="success"))
            vis.append(t)
    if not vis:
        return print("skip grid_heat")
    buttons = [dict(label=f"t{t}", method="update", args=[{"visible": [bool(v == t) for v in vis]}]) for t in TASKS]
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, y=1.32, xanchor="left", yanchor="bottom",
                                        direction="right", type="buttons")])
    fig.update_xaxes(title_text="scale c")
    save(fig, out, "grid_heat", title="Per-direction grid success", height=430,
         margin=dict(l=60, r=20, t=110, b=50))


def fig_effect(R, out, rep):
    fig = make_subplots(rows=2, cols=len(TASKS), subplot_titles=[f"t{t}" for t in TASKS], shared_xaxes=True,
                        vertical_spacing=0.08)
    for c, t in enumerate(TASKS, 1):
        z0 = grid_zero(R, t)
        if z0 is None:
            continue
        noise = mfn = None
        for w in WS:
            ep = R.get((t, "grid", w))
            if ep is None:
                continue
            xs, eff, mf = [], [], []
            for s in np.unique(ep["scale"][ep["ep_z"] > 0]):
                m = ep["scale"] == s
                g = {k: ep[k][m] for k in ("ep_z", "ep_init", "ep_rep", "success", "steps", "state", "dir")}
                r = analyze.metrics(g, zero_of(R[(t, "grid", "random-s0")]))
                xs.append(s), eff.append(r["effect_rms_cm"]), mf.append(r["main_effect_frac"])
                noise, mfn = r["noise_rms_cm"], r["main_effect_frac_noise"]
                rep.setdefault("effect", {}).setdefault(f"t{t}", {}).setdefault(w, {})[f"{s:+g}"] = {
                    k: float(v) for k, v in r.items() if not k.startswith("_")}
            fig.add_scatter(x=xs, y=eff, name=w, legendgroup=w, showlegend=c == 1,
                            line=dict(color=COL[w], dash=DASH[w]), row=1, col=c)
            fig.add_scatter(x=xs, y=mf, name=w, legendgroup=w, showlegend=False,
                            line=dict(color=COL[w], dash=DASH[w]), row=2, col=c)
        if noise is not None:
            fig.add_hline(y=noise, line=dict(color=COL["zero"], dash="dash"), row=1, col=c)
            fig.add_hline(y=mfn, line=dict(color=COL["zero"], dash="dash"), row=2, col=c)
    fig.update_xaxes(title_text="scale c", row=2)
    fig.update_yaxes(title_text="eef effect, RMS cm", row=1, col=1)
    fig.update_yaxes(title_text="main-effect fraction", range=[0, 1], row=2, col=1)
    save(fig, out, "effect", height=560, title="Trajectory effect and its consistency")


def fig_line(R, out, rep):
    pairs_ = [(t, w) for t in TASKS for w in ("random-s0", "pca") if (t, "line", w) in R]
    if not pairs_:
        return print("skip line")
    fig = make_subplots(rows=2, cols=len(TASKS), subplot_titles=[f"t{t} {w}" for w in ("random-s0", "pca")
                                                                 for t in TASKS], shared_yaxes=True,
                        vertical_spacing=0.14)
    for t, w in pairs_:
        line, grid = R[(t, "line", w)], R.get((t, "grid", w))
        sel = json.loads((line["path"].parent / f"sel-{w}.json").read_text())
        row, col = (1 if w == "random-s0" else 2), TASKS.index(t) + 1
        p0 = line["success"][line["ep_z"] == 0].mean()
        for j, k in enumerate(sel["line_dirs"]):
            m = line["dir"] == j
            pts = {float(s): line["success"][m & (line["scale"] == s)].mean() for s in np.unique(line["scale"][m])}
            if grid is not None:  # same inits, rep 0: grid points on this direction join the line
                gm = grid["dir"] == k
                pts |= {float(s): grid["success"][gm & (grid["scale"] == s)].mean() for s in np.unique(grid["scale"][gm])}
            pts[0.0] = p0
            x = sorted(pts)
            name = f"u{k} (best in grid)" if j == 0 else f"u{k} (reference)"
            fig.add_scatter(x=x, y=[pts[s] for s in x], mode="lines+markers", name=name,
                            line=dict(color=COL[w], dash="solid" if j == 0 else "dot"), row=row, col=col,
                            legendgroup=f"{w}{j}", showlegend=False)
            rep.setdefault("line", {}).setdefault(f"t{t}", {}).setdefault(w, {})[f"u{k}"] = {
                f"{s:+g}": round(pts[s], 3) for s in x}
    fig.update_xaxes(title_text="scale c")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_yaxes(title_text="success", col=1)
    save(fig, out, "line", height=600, title="Line scans z = c·u_k")


def heldout(R, rep):
    rows = []
    for t in TASKS:
        z0run = R.get((t, "val", "random-s0"))
        if z0run is None:
            continue
        for w in WS:
            run = R.get((t, "val", w))
            if run is None:
                continue
            sel = json.loads((run["path"].parent / f"sel-{w}.json").read_text())["picks"][0]
            P = pairs(run, 1, z0run)
            d = P[:, 2] - P[:, 3]
            per_init = np.array([d[P[:, 0] == i].mean() for i in np.unique(P[:, 0])])
            nz = per_init[per_init != 0]
            both = (P[:, 2] == 1) & (P[:, 3] == 1)
            ds = P[both, 4] - P[both, 5]
            rows.append(dict(task=t, w=w, dir=sel["dir"], scale=sel["scale"], grid=sel["grid_success"],
                             held=float(P[:, 2].mean()), zero=float(P[:, 3].mean()), n=len(P),
                             wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                             p_sign=float(binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue) if (d != 0).any() else 1.0,
                             p_init=float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0,
                             dsteps=float(ds.mean()) if both.any() else np.nan, n_both=int(both.sum()),
                             p_steps=float(wilcoxon(ds).pvalue) if (ds != 0).sum() > 0 else 1.0))
    # Holm over all picks, on the init-level test
    order = np.argsort([r["p_init"] for r in rows])
    m, run_max = len(rows), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * rows[i]["p_init"]))
        rows[i]["p_holm"] = run_max
    rep["heldout"] = rows
    return rows


def fig_heldout(rows, out):
    if not rows:
        return print("skip heldout")
    fig = go.Figure()
    x = [f"t{r['task']} {r['w'].replace('random-', 'r')}" for r in rows]
    for key, name, color, sym in (("grid", "grid (in-sample, 8 eps)", "#9b9a95", "circle-open"),
                                  ("zero", "z = 0 held-out", COL["zero"], "x"), ("held", "pick held-out", None, "diamond")):
        y = [r[key] for r in rows]
        err = None
        if key != "grid":
            ci = [wilson(r[key] * r["n"], r["n"]) for r in rows]
            err = dict(type="data", array=[c[2] - c[0] for c in ci], arrayminus=[c[0] - c[1] for c in ci])
        fig.add_scatter(x=x, y=y, mode="markers", name=name, error_y=err,
                        marker=dict(size=10, symbol=sym, color=color or [COL[r["w"]] for r in rows]))
    save(fig, out, "heldout", title="Held-out validation (inits 10–39 × 2 noise reps)",
         yaxis=dict(title="success rate", range=[0, 1.05]), xaxis=dict(tickangle=-45), height=440)


def fig_traj3d(R, scene_dir, out, rows, n_inits=10):
    """Per task: held-out z=0 vs each W's pick (same init, noise rep 0); dropdown selects the init."""
    for t in TASKS:
        z0run = R.get((t, "val", "random-s0"))
        sp = Path(scene_dir) / f"t{t}.json"
        if z0run is None or not sp.exists():
            continue
        scene = json.loads(sp.read_text())
        target, goal = scene["objects"][0], scene["objects"][-1]
        inits = sorted(np.unique(z0run["ep_init"]))[:n_inits]
        fig, vis, anns = go.Figure(), [], {}
        for r in inits:
            def ep_of(run, row):
                i = np.flatnonzero((run["ep_z"] == row) & (run["ep_init"] == r) & (run["ep_rep"] == 0))
                return i[0] if len(i) else None
            show = bool(r == inits[0])
            traces = [("z = 0 (baseline)", z0run, 0, COL["zero"])]
            for row in (x for x in rows if x["task"] == t):
                run = R.get((t, "val", row["w"]))
                traces.append((f"{row['w']} pick: u{row['dir']}, c={row['scale']:+g}", run, 1, COL[row["w"]]))
            for name, run, zrow, color in traces:
                i = ep_of(run, zrow)
                if i is None:
                    continue
                n = run["steps"][i]
                p = run["state"][i, :n, :3] * 100
                ok = "success" if run["success"][i] else "fail"
                fig.add_scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", name=f"{name} — {ok}, {n} steps",
                                  line=dict(color=color, width=5 if zrow == 0 else 4), visible=show)
                fig.add_scatter3d(x=p[-1:, 0], y=p[-1:, 1], z=p[-1:, 2], mode="markers", showlegend=False,
                                  marker=dict(color=color, size=4, symbol="x"), visible=show,
                                  hovertext=f"{name}: episode end", hoverinfo="text")
                vis.append(r)
            p0 = z0run["state"][ep_of(z0run, 0), 0, :3] * 100
            pos = scene["inits"][str(r)]
            anns[r] = []
            for label, short, xyz, sym, color in (
                    ("end-effector start", "eef start", p0, "circle", "#0b0b0b"),
                    (f"target object start ({target})", "object", np.array(pos[target]) * 100, "diamond", "#1baf7a"),
                    (f"goal: {goal} position", "goal (basket)", np.array(pos[goal]) * 100, "square", "#c0392b")):
                fig.add_scatter3d(x=[xyz[0]], y=[xyz[1]], z=[xyz[2]], mode="markers", name=label, visible=show,
                                  marker=dict(size=7, symbol=sym, color=color))
                vis.append(r)
                anns[r].append(dict(x=xyz[0], y=xyz[1], z=xyz[2], text=short, showarrow=True, arrowhead=0, ax=0, ay=-30,
                                    font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.8)"))
        fig.add_scatter3d(x=[None], y=[None], z=[None], mode="markers", name="× = episode end (path color)",
                          marker=dict(color="#52514e", size=4, symbol="x"))
        vis.append("always")
        buttons = [dict(label=f"init {r}", method="update",
                        args=[{"visible": [bool(v == r or v == "always") for v in vis]}, {"scene.annotations": anns[r]}])
                   for r in inits]
        fig.update_layout(updatemenus=[dict(buttons=buttons, x=1, y=1, xanchor="right", direction="down")],
                          scene=dict(xaxis_title="x [cm]", yaxis_title="y [cm]", zaxis_title="z [cm]",
                                     aspectmode="data", camera=dict(eye=dict(x=1.9, y=-1.6, z=1.1)),
                                     domain=dict(x=[0, 0.64]), annotations=anns[inits[0]]))
        save(fig, out, f"traj3d_t{t}", height=600, legend=dict(orientation="v", x=0.66, y=0.5, yanchor="middle"),
             margin=dict(l=0, r=0, t=60, b=0),
             title=f"t{t}: {scene['task']}")


def grasp_table(R, rows, rep):
    """Share of episodes with an empty close (gripper closed on nothing at least once)."""
    out = []
    for t in TASKS:
        z0run = R.get((t, "val", "random-s0"))
        if z0run is None:
            continue
        def frac(run, row):
            st = run["state"][run["ep_z"] == row]
            width = st[..., 6] - st[..., 7]
            empty = (width < EMPTY_CLOSE).any(1)
            fail = ~run["success"][run["ep_z"] == row]
            return float(empty.mean()), float(empty[fail].mean()) if fail.any() else np.nan
        e0 = frac(z0run, 0)
        for r in (x for x in rows if x["task"] == t):
            out.append(dict(task=t, w=r["w"], zero_all=e0[0], zero_fail=e0[1],
                            pick_all=frac(R[(t, "val", r["w"])], 1)[0]))
    rep["empty_close"] = out


def table(out, name, head, rows):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    (out / f"{name}.html").write_text(f"<table><tr>{h}</tr>{b}</table>")


def tables(rep, out):
    g = rep.get("grid", {})
    rows = []
    for t in TASKS:
        for w in WS:
            if w in g.get(f"t{t}", {}):
                x = g[f"t{t}"][w]
                z0 = rep["z0"][f"t{t}"]["grid"][0]
                rows.append([f"t{t}", w, f"{z0:.2f}"] + [f"{m:.2f}" for m in x["mean"]] + [f"{max(x['max']):.2f}",
                            f"{x['plus_beats_minus']} / {x['minus_beats_plus']}"])
    if rows:
        sc = g[f"t{TASKS[0]}"]["random-s0"]["scales"]
        table(out, "tab_grid", ["task", "W", "z=0"] + [f"c={c:+g}" for c in sc] + ["best single z", "+c &gt; −c / −c &gt; +c"],
              rows)
    rows = [[f"t{r['task']}", r["w"], f"u{r['dir']}, {r['scale']:+g}", f"{r['grid']:.2f}", f"{r['zero']:.2f}",
             f"<b>{r['held']:.2f}</b>", f"{r['wins']}/{r['losses']}", f"{r['p_sign']:.1g}", f"{r['p_init']:.1g}",
             f"{r['p_holm']:.1g}", f"{r['dsteps']:+.1f} ({r['n_both']}, p={r['p_steps']:.1g})"]
            for r in rep.get("heldout", [])]
    if rows:
        table(out, "tab_heldout", ["task", "W", "pick (u, c)", "grid (8 eps)", "z=0 held-out", "pick held-out",
                                   "wins/losses", "sign p", "init-level p", "Holm p", "Δsteps both-success (n, p)"], rows)
    rows = [[f"t{e['task']}", e["w"], f"{e['zero_all']:.2f}", f"{e['zero_fail']:.2f}", f"{e['pick_all']:.2f}"]
            for e in rep.get("empty_close", [])]
    if rows:
        table(out, "tab_empty", ["task", "W", "z=0: all eps", "z=0: failed eps", "pick: all eps"], rows)


def page(template, figs, dest):
    """Inline <!--fig:name--> fragments from figs into template; write dest."""
    import re
    html = re.sub(r"<!--fig:(\w+)-->", lambda m: (figs / f"{m.group(1)}.html").read_text()
                  if (figs / f"{m.group(1)}.html").exists() else f"<p>(missing {m.group(1)})</p>", template.read_text())
    dest.write_text(html)
    print("page", dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=Path, required=True, help="dir with research.scene t{T}.json")
    parser.add_argument("--out", type=Path, required=True, help="figure fragment dir")
    parser.add_argument("--template", type=Path, help="page with <!--fig:name--> markers")
    parser.add_argument("--page", type=Path, help="output page (needs --template)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R, rep = runs(args.root), {}
    fig_baseline(R, args.out, rep)
    fig_openloop(args.root, args.out, rep)
    fig_grid(R, args.out, rep)
    fig_grid_heat(R, args.out)
    fig_effect(R, args.out, rep)
    fig_line(R, args.out, rep)
    rows = heldout(R, rep)
    fig_heldout(rows, args.out)
    grasp_table(R, rows, rep)
    fig_traj3d(R, args.scene, args.out, rows)
    tables(rep, args.out)
    (args.root / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    if args.template:
        page(args.template, args.out, args.page)
    print(f"numbers in {args.root / 'report.json'}")


if __name__ == "__main__":
    main()
