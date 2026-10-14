"""Figures + tables for docs/results/smolvla_icl_object.html (z optimization on fixed inits, VLM vs CEM, all plotly).

Each ROOT holds research.optimize output dirs ROOT/*/*/{provenance.json, chains/i{init}-r{rep}.jsonl, vlm/...};
task and method come from provenance. Chains may be incomplete or still running. Writes OUT_DIR/*.html fragments
(div only) and, with --template, the page with <!--fig:name--> fragments inlined (it loads plotly.min.js next to it).
"""
import argparse
import base64
import json
import re
import textwrap
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TASKS = (0, 7, 9, 4)
METHODS = ("vlm", "vlm-perm", "llm-perm", "cem")
COL = {"vlm": "#eb6834", "vlm-perm": "#9b4dca", "llm-perm": "#1f9e74", "cem": "#2a78d6", "zero": "#52514e"}
SYM = {0: "circle", 7: "diamond", 9: "square", 4: "triangle-up"}
LAYOUT = dict(template="plotly_white", font=dict(size=12), margin=dict(l=60, r=20, t=50, b=50),
              legend=dict(orientation="h", y=-0.15))


def load(roots):
    """{(task, method, init, rep): chain dict} from all roots; later roots override."""
    C = {}
    for root in roots:
        for prov in sorted(root.glob("*/*/provenance.json")):
            cfg = json.loads(prov.read_text())["config"]
            cfg["method"] = ("llm" if cfg.get("text_only") else cfg["method"]) + ("-perm" if cfg.get("perm") else "")
            for f in sorted((prov.parent / "chains").glob("i*-r*.jsonl")):
                cands = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                if not cands:
                    continue
                init, rep = map(int, re.match(r"i(\d+)-r(\d+)", f.stem).groups())
                rate = np.array([c["rate"] for c in cands])
                best = max(cands, key=lambda c: (c["rate"], -c["mean_steps"]))
                C[(cfg["task_id"], cfg["method"], init, rep)] = dict(
                    task=cfg["task_id"], method=cfg["method"], init=init, rep=rep, cands=cands, rate=rate,
                    best=best, K=cfg["k"], budget=cfg["budget"], dir=prov.parent, name=f.stem,
                    solved=bool(rate.max() == 1), finished=bool(rate.max() == 1 or len(cands) >= cfg["budget"]))
    return C


def first_hit(ch, thr):
    """First candidate index with rate >= thr, else None."""
    i = np.flatnonzero(ch["rate"] >= thr - 1e-9)
    return int(i[0]) if len(i) else None


def best_so_far(ch):
    """Best rate after each candidate; a solved chain is carried at 1.0 to the budget."""
    b = np.maximum.accumulate(ch["rate"])
    return np.concatenate([b, np.ones(ch["budget"] - len(b))]) if ch["solved"] else b


def horizon(ch, thr=1.0):
    """Number of indices with a known outcome: all of them once the chain reached thr or ended."""
    return ch["budget"] if first_hit(ch, thr) is not None or ch["finished"] else len(ch["cands"])


def label(ch):
    return f"t{ch['task']} i{ch['init']} r{ch['rep']} {ch['method']}"


def save(fig, out, name, **layout):
    fig.update_layout(**(LAYOUT | layout))
    (out / f"{name}.html").write_text(fig.to_html(include_plotlyjs=False, full_html=False,
                                                  config={"displaylogo": False, "responsive": True}))
    print("figure", name)


def table(out, name, head, rows):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    (out / f"{name}.html").write_text(f"<table><tr>{h}</tr>{b}</table>")


# ---------- figures ----------

def fig_best(C, out):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"t{t}" for t in TASKS], shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.1)
    shown = set()  # legend entry on the first subplot where a method appears
    for n, t in enumerate(TASKS):
        r, c = n // 2 + 1, n % 2 + 1
        for m in METHODS:
            chs = [ch for ch in C.values() if ch["task"] == t and ch["method"] == m]
            curves = [best_so_far(ch) for ch in chs]
            for ch, y in zip(chs, curves):
                fig.add_scatter(x=np.arange(len(y)), y=y, mode="lines", line=dict(color=COL[m], width=1),
                                opacity=0.35, name=label(ch), legendgroup=m + "_chain", showlegend=False,
                                hovertemplate=label(ch) + " cand %{x}: best %{y:.3f}<extra></extra>", row=r, col=c)
                fig.add_scatter(x=np.arange(len(ch["rate"])), y=ch["rate"], mode="lines",
                                line=dict(color=COL[m], width=1, dash="dash"), opacity=0.35, name=m.upper() + " raw",
                                legendgroup=m + "_raw", showlegend=m not in shown and ch is chs[0],
                                hovertemplate=label(ch) + " cand %{x}: rate %{y:.3f}<extra></extra>", row=r, col=c)
            if curves:  # mean only where every chain of the method is known
                L = min(map(len, curves))
                fig.add_scatter(x=np.arange(L), y=np.mean([y[:L] for y in curves], 0), mode="lines+markers",
                                line=dict(color=COL[m], width=3), marker=dict(size=4), name=m.upper(),
                                legendgroup=m, showlegend=m not in shown,
                                hovertemplate=m.upper() + f" cand %{{x}}: mean best %{{y:.3f}} ({len(curves)} chains)<extra></extra>",
                                row=r, col=c)
                shown.add(m)
    fig.update_xaxes(range=[-0.5, 29.5], dtick=5)
    fig.update_xaxes(title_text="candidates evaluated (0 = z=0)", row=2)
    fig.update_yaxes(range=[-0.03, 1.05], dtick=0.25)
    fig.update_yaxes(title_text="best success so far", col=1)
    save(fig, out, "best", title="Best-so-far per chain (thin), raw rate per candidate (dashed), per-method mean of best-so-far (bold)", height=620)


def fig_solved(C, out):
    thrs = (6 / 8, 1.0)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["≥ 6/8", "8/8"], shared_yaxes=True)
    for c, thr in enumerate(thrs, 1):
        for m in METHODS:
            chs = [ch for ch in C.values() if ch["method"] == m]
            if not chs:
                continue
            hits = [first_hit(ch, thr) for ch in chs]
            xs = np.arange(min(horizon(ch, thr) for ch in chs))  # only where every chain is known
            ys = [np.mean([h is not None and h <= x for h in hits]) for x in xs]
            fig.add_scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=COL[m], width=2.5, shape="hv"),
                            marker=dict(size=4), name=m.upper(), legendgroup=m, showlegend=c == 1,
                            hovertemplate=m.upper() + f" cand %{{x}}: %{{y:.2f}} of {len(chs)} chains<extra></extra>",
                            row=1, col=c)
    fig.update_xaxes(range=[-0.5, 29.5], dtick=5, title_text="candidates evaluated (0 = z=0)")
    fig.update_yaxes(range=[-0.03, 1.05], dtick=0.25)
    fig.update_yaxes(title_text="fraction of chains reached", col=1)
    save(fig, out, "solved", title="Fraction of chains that reached the threshold, pooled over tasks", height=400)


def fig_explore(C, out):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["success rate of each candidate", "‖z‖ of each candidate"])
    for m, dx in zip(METHODS, (-0.3, -0.1, 0.1, 0.3)):
        chs = [ch for ch in C.values() if ch["method"] == m]
        if not chs:
            continue
        x = np.concatenate([np.arange(len(ch["cands"])) + dx for ch in chs])
        rate = np.concatenate([ch["rate"] for ch in chs])
        norm = np.concatenate([[np.linalg.norm(c["z"]) for c in ch["cands"]] for ch in chs])
        txt = sum([[label(ch)] * len(ch["cands"]) for ch in chs], [])
        for col, y in ((1, rate), (2, norm)):
            fig.add_scatter(x=x, y=y, mode="markers", marker=dict(color=COL[m], size=6, opacity=0.55), name=m.upper(),
                            legendgroup=m, showlegend=col == 1, text=txt,
                            hovertemplate="%{text} cand %{x:.0f}: %{y:.3f}<extra></extra>", row=1, col=col)
    fig.update_xaxes(range=[-0.5, 29.5], dtick=5, title_text="candidate index (0 = z=0)")
    fig.update_yaxes(range=[-0.05, 1.05], dtick=0.25, title_text="success rate", col=1)
    fig.update_yaxes(rangemode="tozero", title_text="‖z‖", col=2)
    save(fig, out, "explore", title="Every evaluated candidate (methods shifted left to right in legend order)", height=400)


def vlm_attempts(C):
    """Failed attempt counts from VLM response logs: API error, no <param> block, other (duplicate / incomplete)."""
    n = {"API error": 0, "no &lt;param&gt; block": 0, "duplicate or incomplete z": 0}
    for ch in C.values():
        if not ch["method"].startswith(("vlm", "llm")):
            continue
        accepted = {c["i"] for c in ch["cands"] if c["source"].startswith("vlm")}
        for f in sorted((ch["dir"] / "vlm" / ch["name"]).glob("response_*.txt")):
            if int(f.stem.split("_")[1]) >= len(ch["cands"]):  # proposal still being evaluated
                continue
            parts = re.split(r"^--- retry \d+ temp [\d.]+", f.read_text(), flags=re.M)[1:]
            ok_last = int(f.stem.split("_")[1]) in accepted
            for j, p in enumerate(parts):
                if ok_last and j == len(parts) - 1:
                    continue
                if p.lstrip().startswith(": error"):
                    n["API error"] += 1
                elif not re.search(r"<param\b", p.split("--- content ---")[-1], flags=re.I):
                    n["no &lt;param&gt; block"] += 1
                else:
                    n["duplicate or incomplete z"] += 1
    return n


def fig_vlm(C, out):
    calls = [(ch["task"], c) for ch in C.values() if ch["method"].startswith(("vlm", "llm")) for c in ch["cands"] if c["source"] != "zero"]
    if not calls:
        return print("skip vlm")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["latency per proposal [s]", "completion tokens per call"])
    for t in TASKS:
        cs = [c for tt, c in calls if tt == t]
        if not cs:
            continue
        fig.add_box(y=[c.get("latency", np.nan) for c in cs], name=f"t{t}", marker_color=COL["vlm"], boxpoints="all",
                    jitter=0.4, showlegend=False, row=1, col=1)
        fig.add_box(y=[c["usage"].get("completion_tokens", np.nan) for c in cs if "usage" in c], name=f"t{t}",
                    marker_color=COL["vlm"], boxpoints="all", jitter=0.4, showlegend=False, row=1, col=2)
    src = {}
    for _, c in calls:
        src[c["source"]] = src.get(c["source"], 0) + 1
    rows = [["proposal source", k, v] for k, v in sorted(src.items())] + [
        ["failed attempt", k, v] for k, v in vlm_attempts(C).items()]
    table(out, "tab_vlm", ["kind", "", "count"], rows)
    save(fig, out, "vlm_diag", title=f"VLM / LLM proposals ({len(calls)} incl. fallbacks)", height=400)


def load_heldout(root, C):
    """Attach research.heldout rates to chains (ch["held"]); returns {(task, init): z=0 held-out rate}."""
    by_path = {(ch["dir"] / "chains" / f"{ch['name']}.jsonl").resolve(): ch for ch in C.values()}
    Z = {}
    for f in sorted(root.glob("*/results.jsonl")):
        task = json.loads((f.parent / "provenance.json").read_text())["config"]["task_id"]
        for row in map(json.loads, f.read_text().splitlines()):
            if row["method"] == "zero":
                Z[(task, row["init"])] = row["rate"]
            elif Path(row["key"]) in by_path:
                by_path[Path(row["key"])]["held"] = row["rate"]
    return Z


def fig_heldout(C, Z, out):
    chs = [ch for ch in C.values() if "held" in ch]
    if not chs:
        return print("skip heldout")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        subplot_titles=["best z: 8 optimization seeds vs 32 held-out seeds", "held-out rate per task"])
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=COL["zero"], dash="dash", width=1),
                    showlegend=False, hoverinfo="skip", row=1, col=1)
    rng = np.random.default_rng(0)
    for n, m in enumerate(METHODS):
        cm = [ch for ch in chs if ch["method"] == m]
        if not cm:
            continue
        fig.add_scatter(x=[ch["best"]["rate"] + rng.uniform(-0.02, 0.02) for ch in cm], y=[ch["held"] for ch in cm],
                        mode="markers", marker=dict(color=COL[m], symbol=[SYM[ch["task"]] for ch in cm], size=9, opacity=0.8),
                        name=m.upper(), legendgroup=m, text=[label(ch) for ch in cm],
                        hovertemplate="%{text}<br>train %{x:.2f}, held-out %{y:.3f}<extra></extra>", row=1, col=1)
        fig.add_box(x=[f"t{ch['task']}" for ch in cm], y=[ch["held"] for ch in cm], name=m.upper(), legendgroup=m,
                    marker_color=COL[m], boxpoints="all", jitter=0.4, pointpos=0, showlegend=False, text=[label(ch) for ch in cm],
                    hovertemplate="%{text}: held-out %{y:.3f}<extra></extra>", row=1, col=2)
    zt = sorted(Z.items(), key=lambda kv: TASKS.index(kv[0][0]))
    fig.add_scatter(x=[f"t{t}" for (t, _), _ in zt], y=[r for _, r in zt], mode="markers", name="z = 0",
                    marker=dict(color=COL["zero"], symbol="line-ew-open", size=40, line=dict(width=3)),
                    hovertemplate="%{x} z=0: held-out %{y:.3f}<extra></extra>", row=1, col=2)
    fig.update_layout(boxmode="group")
    fig.update_xaxes(range=[-0.05, 1.05], dtick=0.25, title_text="rate on the chain's 8 seeds", row=1, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=[f"t{t}" for t in TASKS], row=1, col=2)
    fig.update_yaxes(range=[-0.03, 1.05], dtick=0.25, title_text="rate on 32 held-out seeds", col=1)
    save(fig, out, "heldout", title="Held-out re-evaluation of each chain's best z", height=480)

    rows = []
    for t in TASKS + (None,):
        for m in METHODS:
            cm = [ch for ch in chs if ch["method"] == m and (t is None or ch["task"] == t)]
            if not cm:
                continue
            d = np.array([ch["held"] - Z[(ch["task"], ch["init"])] for ch in cm])
            sv = [ch["held"] for ch in cm if ch["solved"]]
            rows.append([f"t{t}" if t is not None else "<b>all</b>", m.upper(), len(cm),
                         f"{np.mean([ch['best']['rate'] for ch in cm]):.3f}", f"{np.mean([ch['held'] for ch in cm]):.3f}",
                         f"{np.mean(sv):.3f} (n={len(sv)})" if sv else "–", f"{d.mean():+.3f}",
                         f"{(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}"])
    rows += [[f"t{t}", "z = 0", 1, "", f"{r:.3f}", "", "", ""] for (t, _), r in zt]
    table(out, "tab_heldout", ["task", "method", "chains", "mean train rate (best z)", "mean held-out",
                               "held-out of 8/8 chains", "mean Δ vs z=0 held-out", "better / worse / equal than z=0"], rows)


def fig_steps(C, out):
    fig = go.Figure()
    T = 280
    fig.add_scatter(x=[0, T], y=[0, T], mode="lines", line=dict(color=COL["zero"], dash="dash", width=1),
                    showlegend=False, hoverinfo="skip")
    for m in METHODS:
        for t in TASKS:
            chs = [ch for ch in C.values() if ch["method"] == m and ch["task"] == t]
            if not chs:
                continue
            fig.add_scatter(x=[ch["cands"][0]["mean_steps"] for ch in chs], y=[ch["best"]["mean_steps"] for ch in chs],
                            mode="markers", marker=dict(color=COL[m], symbol=SYM[t], size=10, opacity=0.8),
                            name=f"{m.upper()} t{t}", legendgroup=m, text=[label(ch) for ch in chs],
                            customdata=[[ch["rate"][0], ch["best"]["rate"]] for ch in chs],
                            hovertemplate="%{text}<br>z=0: %{x:.0f} steps (rate %{customdata[0]:.3f})<br>"
                                          "best: %{y:.0f} steps (rate %{customdata[1]:.3f})<extra></extra>")
    fig.update_xaxes(range=[100, T + 5], title_text="z=0: mean steps of successes")
    fig.update_yaxes(range=[100, T + 5], title_text="best candidate: mean steps of successes")
    save(fig, out, "steps", title="Best candidate vs z=0", height=560, width=600)


# ---------- tables ----------

def tables(C, out):
    def fmt(i):
        return "–" if i is None else str(i)

    rows = []
    for (t, m, init, rep), ch in sorted(C.items(), key=lambda kv: (TASKS.index(kv[0][0]), kv[0][2], kv[0][3], kv[0][1])):
        K, n = ch["K"], len(ch["cands"])
        status = f"solved at {n - 1}" if ch["solved"] else ("budget used" if ch["finished"] else f"running ({n})")
        rows.append([f"t{t}", init, rep, m.upper(), status, f"{round(ch['rate'][0] * K)}/{K}",
                     fmt(first_hit(ch, 6 / 8)), fmt(first_hit(ch, 1.0)), f"{round(ch['best']['rate'] * K)}/{K}",
                     f"{np.linalg.norm(ch['best']['z']):.2f}", f"{ch['best']['mean_steps']:.0f}"])
    table(out, "tab_chains", ["task", "init", "rep", "method", "status (candidates)", "z=0", "first ≥6/8",
                              "first 8/8", "best", "best ‖z‖", "best mean steps"], rows)

    rows = []
    for t in TASKS + (None,):
        for m in METHODS:
            chs = [ch for ch in C.values() if ch["method"] == m and (t is None or ch["task"] == t)]
            if not chs:
                continue
            row = [f"t{t}" if t is not None else "<b>all</b>", m.upper(), len(chs), sum(ch["finished"] for ch in chs)]
            for thr in (6 / 8, 1.0):  # chains with a known outcome only; not reached within budget = inf
                h = [first_hit(ch, thr) for ch in chs]
                known = [x if x is not None else np.inf for ch, x in zip(chs, h) if x is not None or ch["finished"]]
                med = np.median(known) if known else None
                row += [f"{sum(x is not None for x in h)}/{len(chs)}",
                        "–" if med is None else ("&gt;29" if np.isinf(med) else f"{med:g}") + f" (n={len(known)})"]
            rows.append(row)
    table(out, "tab_median", ["task", "method", "chains", "finished", "reached ≥6/8", "median first ≥6/8",
                              "reached 8/8", "median first 8/8"], rows)
    print("table tab_chains, tab_median")


# ---------- VLM examples ----------

def vlm_examples(C, out, n_lines=8, width=110):
    """Excerpts of the final answer (not reasoning) for the first, best and last proposal of the longest VLM chain."""
    chs = [ch for ch in C.values() if ch["method"] == "vlm" and (ch["dir"] / "vlm" / ch["name"]).exists()]
    if not chs:
        return print("skip vlm examples")
    ch = max(chs, key=lambda ch: (len(ch["cands"]), ch["best"]["rate"]))
    d = ch["dir"] / "vlm" / ch["name"]
    picks = sorted({i for i in (1, ch["best"]["i"], len(ch["cands"]) - 1) if i > 0 and (d / f"response_{i:03d}.txt").exists()})
    html = []
    for i in picks:
        text = (d / f"response_{i:03d}.txt").read_text().split("--- content ---")[-1]
        text = re.sub(r"<param\b.*?(</param\s*>|$)", "", text, flags=re.S | re.I).strip()
        lines = sum([textwrap.wrap(para, width) or [""] for para in text.splitlines()], [])
        exc = "\n".join(lines[:n_lines]) + (" …" if len(lines) > n_lines else "")
        c = ch["cands"][i]
        z = ", ".join(f"{v:.1f}" for v in c["z"])
        exc = exc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html.append(f"<p><b>{label(ch)}, candidate {i}</b> → z = [{z}], result {round(c['rate'] * ch['K'])}/{ch['K']}"
                    f"</p><pre class=\"vlm\">{exc}</pre>")
    (out / "vlm_text.html").write_text("\n".join(html))
    print("fragment vlm_text")

    imgs = sorted(d.glob("image_*_0.jpg"))
    pick = d / f"image_{ch['best']['i']:03d}_0.jpg"
    img = pick if pick.exists() else (imgs[0] if imgs else None)
    if img is None:
        return print("skip vlm image")
    im = cv2.imread(str(img))
    im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
    b64 = base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()).decode()
    n = int(img.stem.split("_")[1])
    (out / "vlm_image.html").write_text(
        f'<img src="data:image/jpeg;base64,{b64}" alt="VLM input image" style="max-width:100%;height:auto">'
        f"<figcaption>{label(ch)}: first image attached to the prompt for candidate {n} (one rollout of candidate {n - 1}"
        f", {ch['cands'][n - 1]['rate'] * ch['K']:.0f}/{ch['K']} successes; shown at half resolution).</figcaption>")
    print("fragment vlm_image")


def summary(C):
    for t in TASKS:
        for m in METHODS:
            chs = [ch for ch in C.values() if ch["task"] == t and ch["method"] == m]
            if chs:
                print(f"t{t} {m}: {len(chs)} chains, cands {[len(ch['cands']) for ch in chs]}, "
                      f"z0 {[round(ch['rate'][0] * 8) for ch in chs]}, best {[round(ch['best']['rate'] * 8) for ch in chs]}, "
                      f"first>=6/8 {[first_hit(ch, 0.75) for ch in chs]}, first 8/8 {[first_hit(ch, 1.0) for ch in chs]}")


def page(template, figs, dest):
    """Inline <!--fig:name--> fragments from figs into template; write dest."""
    html = re.sub(r"<!--fig:(\w+)-->", lambda m: (figs / f"{m.group(1)}.html").read_text()
                  if (figs / f"{m.group(1)}.html").exists() else f"<p>(missing {m.group(1)})</p>", template.read_text())
    dest.write_text(html)
    print("page", dest, f"{len(html) / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True, help="figure fragment dir")
    parser.add_argument("--template", type=Path, help="page with <!--fig:name--> markers")
    parser.add_argument("--page", type=Path, help="output page (needs --template)")
    parser.add_argument("--heldout", type=Path, help="research.heldout root (*/results.jsonl)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    C = load(args.roots)
    print(f"{len(C)} chains")
    fig_best(C, args.out)
    fig_solved(C, args.out)
    fig_explore(C, args.out)
    fig_vlm(C, args.out)
    fig_steps(C, args.out)
    if args.heldout:
        fig_heldout(C, load_heldout(args.heldout, C), args.out)
    tables(C, args.out)
    vlm_examples(C, args.out)
    summary(C)
    if args.template:
        page(args.template, args.out, args.page)


if __name__ == "__main__":
    main()
