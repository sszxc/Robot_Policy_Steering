"""Figures for docs/results/smolvla_steer_object.html (z -> FiLM steering on LIBERO-Object).

Each figure is skipped with a note if its inputs are missing.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

TASKS = (0, 9, 2)
GRID = {0: "pair-64646851-t0", 9: "pair-64646852-t9", 2: "pair-64646853-t2"}
WS = ("random", "pca")
COL = {"random": "#2a78d6", "pca": "#eb6834", "zero": "#52514e"}
TASK_COL = {0: "#1baf7a", 9: "#4a3aa7", 2: "#e87ba4"}
MARK = {"random": "o", "pca": "s"}
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 2,
})


def load(d):
    """Done episodes of a research.run output dir, or None."""
    p = Path(d) / "results.npz"
    if not p.exists():
        return None
    r = dict(np.load(p))
    m = r["done"]
    ep = {k: r[k][m] for k in ("ep_z", "ep_init", "ep_rep", "success", "steps", "state")}
    ep.update(Z=r["Z"], scale=r["scale"][ep["ep_z"]], dir=r["direction"][ep["ep_z"]])
    return ep


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def paired(run, row, zero_run=None):
    """(success at z row, success at z=0) over (init, rep) pairs present in both."""
    zr = zero_run if zero_run is not None else run
    z0 = {(i, r): s for i, r, s, z in zip(zr["ep_init"], zr["ep_rep"], zr["success"], zr["ep_z"]) if z == 0}
    pairs = [(s, z0[(i, r)]) for i, r, s, z in zip(run["ep_init"], run["ep_rep"], run["success"], run["ep_z"])
             if z == row and (i, r) in z0]
    return np.array(pairs, dtype=float).reshape(-1, 2)


def skip(name, why):
    print(f"skip {name}: {why}")


def save(fig, out, name):
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def openloop_a0(d):
    """Per-state |Δa0| [N, K, S] and noise deviation of a0 [N, R], both / sqrt(7)."""
    A = d["base"].shape[-1]
    dose = np.linalg.norm(d["dose"][..., 0, :], axis=-1) / np.sqrt(A)
    na = d["noise_a"][:, :, 0]
    noise = np.linalg.norm(na - na.mean(1, keepdims=True), axis=-1) / np.sqrt(A)
    return dose, noise


def fig_openloop_dose(O, out):
    files = {(t, w): O / f"openloop-t{t}-{w}/openloop.npz" for t in TASKS for w in WS}
    if not all(f.exists() for f in files.values()):
        return skip("openloop_dose.png", "missing openloop npz")
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, t in zip(axs, TASKS):
        for w in WS:
            d = np.load(files[t, w])
            dose, noise = openloop_a0(d)
            x = dose.reshape(-1, dose.shape[-1])
            q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0)
            ax.fill_between(d["scales"], q25, q75, color=COL[w], alpha=0.18, lw=0)
            ax.plot(d["scales"], q50, "-o", color=COL[w], ms=4, label=w)
        ax.axhline(np.median(noise), color=COL["zero"], ls="--", lw=1.2, label="z=0 noise (median)")
        ax.set(xscale="log", yscale="log", title=f"task {t}", xlabel="scale c (z = c·u_k)")
        ax.set_xticks(d["scales"], [f"{c:g}" for c in d["scales"]])
        ax.minorticks_off()
        ax.set_yticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
    axs[0].set_ylabel("|Δa0| (normalized action units)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Open-loop first-action change (median, 25–75% band over states × directions)",
                 fontsize=9.5, y=1.13)
    save(fig, out, "openloop_dose.png")


def grid_runs(O):
    runs = {}
    for t in TASKS:
        for w in WS:
            runs[t, w] = load(O / "hpc" / GRID[t] / w)
    return runs


def dir_success(ep):
    """{(scale, dir): success rate} over non-zero z rows."""
    out = {}
    for c, k in sorted(set(zip(ep["scale"], ep["dir"]))):
        if k >= 0:
            out[c, k] = ep["success"][(ep["scale"] == c) & (ep["dir"] == k)].mean()
    return out


def fig_grid_success(O, out):
    runs = grid_runs(O)
    if any(v is None for v in runs.values()):
        return skip("grid_success.png", "missing grid results")
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, t in zip(axs, TASKS):
        r0 = runs[t, "random"]
        s0 = r0["success"][r0["ep_z"] == 0].mean()
        ax.axhline(s0, color=COL["zero"], ls="--", lw=1.2, label="z=0")
        for w in WS:
            s = dir_success(runs[t, w])
            cs = sorted({c for c, _ in s} | {0.0})  # c=0 is z=0 for every direction
            M = np.array([[s0 if c == 0 else s[c, k] for c in cs] for k in range(8)])
            for row in M:
                ax.plot(cs, row, color=COL[w], lw=0.7, alpha=0.25)
            ax.plot(cs, M.mean(0), "-o", color=COL[w], ms=4, label=f"{w} (mean of 8 dirs)")
        ax.plot(0, s0, "o", color=COL["zero"], ms=6, zorder=5)
        ax.set(title=f"task {t}", xlabel="scale c", ylim=(-0.03, 1.03))
        ax.set_xticks(cs, [f"{c:g}".replace("0.5", ".5") for c in cs])
    axs[0].set_ylabel("success rate")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "grid_success.png")


def fig_effect_vs_noise(O, out):
    p = O / "hpc/grid_metrics.json"
    if not p.exists():
        return skip("effect_vs_noise.png", "missing grid_metrics.json")
    g = json.loads(p.read_text())
    fig, axs = plt.subplots(2, 3, figsize=(9, 5), sharex=True, sharey="row")
    for j, t in enumerate(TASKS):
        m = g[str(t)]
        for w in WS:
            keys = sorted((k for k in m if k.startswith(w + ":")), key=lambda k: float(k.split(":")[1]))
            cs = [float(k.split(":")[1]) for k in keys]
            # c=0: effect is exactly 0 (same noise, same z); main-effect fraction is 0/0
            e = [m[k]["effect_rms_cm"] for k in keys]
            i0 = int(np.searchsorted(cs, 0))
            axs[0, j].plot(cs[:i0] + [0] + cs[i0:], e[:i0] + [0] + e[i0:], "-o", ms=4, color=COL[w], label=w)
            axs[1, j].plot(cs, [m[k]["main_effect_frac"] for k in keys], "-o", ms=4, color=COL[w], label=w)
        axs[0, j].plot(0, 0, "o", color=COL["zero"], ms=6, zorder=5)
        axs[1, j].annotate("c=0:\nundefined", (0, 0.86), ha="center", fontsize=7, color=COL["zero"])
        any_m = next(iter(m.values()))
        axs[0, j].axhline(any_m["noise_rms_cm"], color=COL["zero"], ls="--", lw=1.2, label="z=0 noise")
        axs[1, j].axhline(any_m["main_effect_frac_noise"], color=COL["zero"], ls="--", lw=1.2, label="z=0 noise")
        axs[0, j].set_title(f"task {t}")
        axs[1, j].set_xlabel("scale c")
        ticks = sorted(cs + [0.0])
        axs[1, j].set_xticks(ticks, [f"{c:g}".replace("0.5", ".5") for c in ticks])
    axs[0, 0].set(ylabel="eef effect RMS (cm)", ylim=(0, None))
    axs[1, 0].set(ylabel="main-effect fraction", ylim=(0, 1))
    axs[0, 0].legend(loc="lower left")
    save(fig, out, "effect_vs_noise.png")


def fig_line(O, out):
    runs = {w: load(O / f"line-t0-{w}") for w in WS}
    if any(v is None for v in runs.values()):
        return skip("line_t0.png", "missing line-t0 results")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for w, ep in runs.items():
        p0 = ep["success"][ep["ep_z"] == 0].mean()
        for k, ls in zip(sorted(set(ep["dir"]) - {-1}), ("-", "--")):
            cs = np.array(sorted(set(ep["scale"][ep["dir"] == k]) | {0.0}))
            ys = [p0 if c == 0 else ep["success"][(ep["dir"] == k) & (ep["scale"] == c)].mean() for c in cs]
            ax.plot(cs, ys, ls, marker="o", ms=3.5, color=COL[w], label=f"{w} u{k}")
        ax.plot(0, p0, "o", ms=8, mfc="white", mec=COL[w], mew=2, zorder=5)
    ax.plot([], [], "o", ms=8, mfc="white", mec=INK2, mew=2, label="z=0")
    ax.set(xlabel="signed scale s (z = s·u_k)", ylabel="success rate (8 inits)", ylim=(-0.03, 1.03),
           title="task 0: success along two directions")
    ax.legend(loc="upper left", ncol=1)
    save(fig, out, "line_t0.png")


def fig_plane(O, out):
    ep, abp = load(O / "plane-t0-random"), O / "plane-t0-ab.npy"
    if ep is None or not abp.exists():
        return skip("plane_t0.png", "missing plane results")
    ab = np.vstack([[0, 0], np.load(abp)])
    vals = [-1, 0, 0.5, 1, 2]
    S, N = np.full((5, 5), np.nan), np.zeros((5, 5), int)
    for row in range(len(ab)):
        m = ep["ep_z"] == row
        i, j = vals.index(ab[row, 1]), vals.index(ab[row, 0])  # row = b, col = a
        N[i, j] = m.sum()
        if m.any():
            S[i, j] = ep["success"][m].mean()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.grid(False)
    im = ax.imshow(np.ma.masked_invalid(S), cmap="Blues", vmin=0, vmax=1, origin="lower")
    for i in range(5):
        for j in range(5):
            if N[i, j] == 0:
                ax.text(j, i, "–", ha="center", va="center", color=INK2)
            else:
                ax.text(j, i, f"{S[i, j]:.2f}\nn={N[i, j]}", ha="center", va="center", fontsize=7.5,
                        color="white" if S[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), [f"{v:g}" for v in vals])
    ax.set_yticks(range(5), [f"{v:g}" for v in vals])
    ax.set(xlabel="a (coefficient of u0)", ylabel="b (coefficient of u4)",
           title=f"task 0, random W: success on z = a·u0 + b·u4\n({N.sum()} episodes, inits 0–7)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="success rate")
    save(fig, out, "plane_t0.png")


def fig_openloop_vs_closedloop(O, out):
    runs = grid_runs(O)
    files = {(t, w): O / f"openloop-t{t}-{w}/openloop.npz" for t in TASKS for w in WS}
    if any(v is None for v in runs.values()) or not all(f.exists() for f in files.values()):
        return skip("openloop_vs_closedloop.png", "missing grid or openloop data")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    X, Y = [], []
    for t in TASKS:
        for w in WS:
            d = np.load(files[t, w])
            chunk = np.sqrt((d["dose"] ** 2).mean(axis=(-1, -2)))  # [N, K, S]
            med = np.median(chunk, axis=0)
            s = dir_success(runs[t, w])
            x, y = [], []
            for j, c in enumerate(d["scales"]):
                for k in range(med.shape[0]):
                    if (c, k) in s:
                        x.append(med[k, j]), y.append(s[c, k])
            ax.scatter(x, y, s=22, marker=MARK[w], color=TASK_COL[t], alpha=0.75, edgecolor=SURF, lw=0.5)
            X += x
            Y += y
    rho = spearmanr(X, Y)[0]
    for t in TASKS:
        ax.scatter([], [], s=22, color=TASK_COL[t], label=f"task {t}")
    for w in WS:
        ax.scatter([], [], s=22, marker=MARK[w], facecolor="none", edgecolor=INK2, label=w)
    ax.set(xscale="log", xlabel="open-loop median chunk |Δa| (normalized)", ylabel="closed-loop success",
           ylim=(-0.03, 1.03), title="per direction, scales 0.5, 1, 2, 4")
    ax.text(0.97, 0.95, f"Spearman ρ = {rho:.2f} (n={len(X)})", transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, out, "openloop_vs_closedloop.png")


def point(ax, x, succ, color, label=None):
    p, lo, hi = wilson(succ.sum(), len(succ))
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=6, color=color, capsize=3, lw=1.5, label=label)
    ax.text(x, hi + 0.03, f"{p:.2f}\nn={len(succ)}", ha="center", va="bottom", fontsize=7, color=INK2)


def fig_heldout(O, out):
    seeds = [load(O / d) for d in ("val-t0-line-random", "val-t0-u0-seed2000", "val-t0-u0-seed3000")]
    seeds = [s for s in seeds if s is not None]
    val = {(t, w): load(O / f"val/t{t}-{w}") for t in (0, 9) for w in WS}
    sel_p = O / "val/selection.json"
    sel = json.loads(sel_p.read_text()) if sel_p.exists() else {}
    trans = {t: load(O / f"val/transfer-t{t}") for t in range(10)}
    trans = {t: e for t, e in trans.items() if e is not None}
    if not seeds and not any(v is not None for v in val.values()) and not trans:
        return skip("heldout.png", "no held-out results")
    groups = []  # (label, [(name, success array, color)])
    if seeds:
        pr = np.vstack([paired(s, 1) for s in seeds])
        groups.append(("t0 line pick\n(3 seeds)", [("z=0", pr[:, 1], COL["zero"]), ("u0", pr[:, 0], COL["random"])]))
    for t in (0, 9):
        zr = val[t, "random"]
        if zr is None:
            continue
        pts = [("z=0", zr["success"][zr["ep_z"] == 0], COL["zero"])]
        for w in WS:
            ep = val[t, w]
            if ep is None:
                continue
            for row, s in enumerate(sel.get(f"t{t}-{w}", [{}, {}])[:2], start=1):
                pr = paired(ep, row, zr)
                name = f"{s.get('scale', 0):g}·u{s.get('dir', '?')}" if s else f"z{row}"
                pts.append((f"{w}\n{name}", pr[:, 0], COL[w]))
        groups.append((f"t{t} grid picks", pts))
    ncol = 2 if trans else 1
    n_pts = sum(len(p) for _, p in groups)
    fig, axs = plt.subplots(1, ncol, figsize=(max(3.2, 0.75 * n_pts + 1) + (4 if trans else 0), 3.4),
                            squeeze=False, gridspec_kw={"width_ratios": [max(1, n_pts), 6][:ncol]})
    ax = axs[0, 0]
    x, ticks, names = 0, [], []
    for label, pts in groups:
        for name, succ, c in pts:
            if len(succ):
                point(ax, x, succ, c)
            ticks.append(x), names.append(name)
            x += 1
        ax.text(x - (len(pts) + 1) / 2, -0.17, label, ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())
        x += 0.6
    ax.set_xticks(ticks, names, fontsize=7.5)
    ax.set(ylabel="success rate (95% Wilson CI)", ylim=(0, 1.15), xlim=(-0.6, x - 0.9),
           title="held-out inits 10–49")
    if trans:
        ax = axs[0, 1]
        rng = np.random.default_rng(0)
        ts = sorted(trans)
        for off, (row, name, mk, fc) in zip((-0.15, 0.15), ((1, "u0", "o", COL["random"]), (2, "u4", "D", SURF))):
            for t in ts:
                pr = paired(trans[t], row)
                if not len(pr):
                    continue
                d = pr[:, 0] - pr[:, 1]
                bs = rng.choice(d, (2000, len(d))).mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                ax.errorbar(t + off, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt=mk, ms=5, mfc=fc,
                            color=COL["random"], capsize=2, lw=1.3, label=name if t == ts[0] else None)
        ax.axhline(0, color=INK2, lw=1)
        ax.set(xticks=ts, xlabel="task", ylabel="success(z) − success(z=0)",
               title="transfer: u0 / u4 at scale 1, random W (bootstrap 95% CI)")
        ax.legend(loc="best", ncol=2)
    save(fig, out, "heldout.png")


PLOTLY_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<script src="../plotly.min.js"></script>
<style>body{{margin:0;font-family:system-ui,sans-serif;background:#fcfcfb}} .p{{height:480px}}</style>
</head><body>{divs}<script>{scripts}</script></body></html>"""


def path3d(ep, i, name, color, **kw):
    """Plotly traces for one episode: eef path (cm) plus an end marker (diamond = success, x = fail)."""
    xyz = np.round(100 * ep["state"][i, : ep["steps"][i], :3].astype(float), 2)
    ok = bool(ep["success"][i])
    line = dict(type="scatter3d", mode="lines", x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(),
                name=f"{name} ({'ok' if ok else 'fail'})", line=dict(color=color, width=5), legendgroup=name, **kw)
    end = dict(type="scatter3d", mode="markers", x=xyz[-1:, 0].tolist(), y=xyz[-1:, 1].tolist(),
               z=xyz[-1:, 2].tolist(), showlegend=False,
               marker=dict(color=color, size=6, symbol="diamond" if ok else "x"), legendgroup=name, **kw)
    return [line, end]


def fig_traj3d(O, out):
    """Interactive 3D eef paths: t0 line scan along u0 (init 0) and held-out z=0 vs the random-W pick."""
    figs = []
    ep = load(O / "line-t0-random")
    if ep is not None:
        scales = sorted(set(ep["scale"][ep["dir"] == 0]))
        traces = []
        for c in [0.0] + scales:
            m = (ep["ep_init"] == 0) & (ep["ep_rep"] == 0)
            m &= (ep["ep_z"] == 0) if c == 0 else ((ep["dir"] == 0) & (ep["scale"] == c))
            i = np.flatnonzero(m)
            if len(i):
                cmap = plt.cm.Reds if c < 0 else plt.cm.Blues  # darker = larger |s|, floor keeps it visible
                col = COL["zero"] if c == 0 else matplotlib.colors.to_hex(cmap(0.35 + 0.65 * abs(c) / 4))
                traces += path3d(ep, i[0], "z=0" if c == 0 else f"s={c:+g}", col)
        figs.append(("task 0, init 0: z = s·u0 (random W); red s<0, blue s>0; click legend to toggle", traces, []))
    v = load(O / "val/t0-random")
    if v is not None:
        inits = sorted(set(v["ep_init"]))[:8]
        traces = []
        for n, r in enumerate(inits):
            for row, name, col in ((0, "z=0", COL["zero"]), (1, "random d4, s=+1", COL["random"])):
                i = np.flatnonzero((v["ep_init"] == r) & (v["ep_rep"] == 0) & (v["ep_z"] == row))
                if len(i):
                    traces += path3d(v, i[0], name, col, visible=n == 0)
        # each init adds 4 traces (line + end marker for z=0 and the pick)
        per = len(traces) // len(inits)
        buttons = [dict(label=f"init {r}", method="update",
                        args=[{"visible": [k // per == n for k in range(len(traces))]}]) for n, r in enumerate(inits)]
        menus = [dict(buttons=buttons, x=0, y=1.08, xanchor="left", direction="down")]
        figs.append(("task 0 held-out (Sol, rep 0): z=0 vs grid pick, same flow noise; pick an init", traces, menus))
    if not figs:
        return skip("traj_t0_3d.html", "missing line-t0-random and val/t0-random")
    divs, scripts = "", ""
    for n, (title, traces, menus) in enumerate(figs):
        layout = dict(title=dict(text=title, font=dict(size=13)), updatemenus=menus, paper_bgcolor="#fcfcfb",
                      margin=dict(l=0, r=0, t=60, b=0), legend=dict(font=dict(size=11)),
                      scene=dict(aspectmode="data", xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title="z (cm)"))
        divs += f'<div id="f{n}" class="p"></div>'
        scripts += f"Plotly.newPlot('f{n}', {json.dumps(traces)}, {json.dumps(layout)});\n"
    js = out.parent / "plotly.min.js"  # shared by all reports in docs/results; works offline
    if not js.exists():
        from plotly.offline import get_plotlyjs
        js.write_text(get_plotlyjs())
    (out / "traj_t0_3d.html").write_text(PLOTLY_PAGE.format(title="eef paths", divs=divs, scripts=scripts))
    print("wrote", out / "traj_t0_3d.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/libero/smolvla/steer")
    ap.add_argument("--out", default="docs/results/smolvla_steer_object")
    a = ap.parse_args()
    O, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_openloop_dose, fig_grid_success, fig_effect_vs_noise, fig_line, fig_plane,
              fig_openloop_vs_closedloop, fig_heldout, fig_traj3d):
        f(O, out)


if __name__ == "__main__":
    main()
