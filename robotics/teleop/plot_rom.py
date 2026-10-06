"""Per-joint distributions of raw mapping angles (mapping.angles) for ROM recordings, one HTML page.

Grey band = raw angles that land inside the robot joint range under the chosen mapping.CALIB preset.
Usage: python -m robotics.teleop.plot_rom [files.npz ...] [--calib full] [--out docs/results/teleop_rom_<calib>.html]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from robotics.teleop import mapping

DEFAULT = ["assets/teleop/20261006_rom_palm_down.npz", "assets/teleop/20261006_rom_palm_side.npz"]
PLOTLY_JS = "docs/results/plotly.min.js"
MIN_CONF = 0.5  # drop frames Leap is unsure about (mostly hand entering the view)
COLORS = ["#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc2"]
ROWS = ["thumb", "index", "middle", "ring", "pinky"]
NAMES = ["cmc", "yaw", "mcp", "ip"] + ["roll", "mcp", "pip", "dip"] * 4


def load(path):
    r = np.load(path)
    keep = r["confidence"] > MIN_CONF
    q = np.array([mapping.angles(k) for k in r["kp"][keep].astype(np.float64)])
    return Path(path).stem.split("rom_")[-1], q, keep.mean()


def band(calib):
    """Raw-angle interval per joint that maps into [LO, HI]."""
    g, o = mapping.CALIB[calib]
    a, b = (mapping.LO - o) / g, (mapping.HI - o) / g
    return np.minimum(a, b), np.maximum(a, b)


def figure(sets, calib):
    lo, hi = band(calib)
    fig = make_subplots(rows=5, cols=4, vertical_spacing=0.06, horizontal_spacing=0.05,
                        subplot_titles=[f"{ROWS[j // 4]} {NAMES[j]}" for j in range(20)])
    for j in range(20):
        row, col = j // 4 + 1, j % 4 + 1
        allq = np.concatenate([q[:, j] for _, q, _ in sets])
        x0, x1 = min(allq.min(), lo[j]) - 0.05, max(allq.max(), hi[j]) + 0.05
        bins = np.linspace(x0, x1, 61)
        for i, (name, q, _) in enumerate(sets):
            h, e = np.histogram(q[:, j], bins, density=True)
            p2, p98 = np.percentile(q[:, j], [2, 98])
            fig.add_trace(go.Scatter(
                x=e[:-1], y=h, mode="lines", line=dict(color=COLORS[i], width=2, shape="hv"),
                fill="tozeroy", fillcolor=COLORS[i] + "26", name=name, legendgroup=name, showlegend=j == 0,
                hovertemplate=f"{name}<br>%{{x:.2f}} rad<br>density %{{y:.2f}}<br>p2..p98 {p2:.2f}..{p98:.2f}<extra></extra>",
            ), row=row, col=col)
        fig.add_vrect(x0=lo[j], x1=hi[j], fillcolor="#9a9893", opacity=0.12, layer="below",
                      line=dict(color="#6f6e69", width=1, dash="dash"), row=row, col=col)
        fig.update_xaxes(range=[x0, x1], row=row, col=col)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#ebeae6", zeroline=True, zerolinecolor="#c3c2bd")
    fig.update_layout(template="plotly_white", height=1250, font=dict(size=12),
                      margin=dict(l=30, r=20, t=60, b=40), legend=dict(orientation="h", y=1.04, x=0),
                      hovermode="closest")
    fig.update_annotations(font_size=12)
    return fig


def table(sets, calib):
    lo, hi = band(calib)
    g, o = mapping.CALIB[calib]
    head = "".join(f"<th colspan=3>{n}</th>" for n, _, _ in sets)
    sub = "<th>p2..p98</th><th>out lo / hi</th><th>robot range used</th>" * len(sets)
    rows = []
    for j in range(20):
        cells = ""
        for _, q, _ in sets:
            x = q[:, j]
            p2, p98 = np.percentile(x, [2, 98])
            used = (np.clip(p98, lo[j], hi[j]) - np.clip(p2, lo[j], hi[j])) / (hi[j] - lo[j])
            cells += (f"<td>{p2:.2f}..{p98:.2f}</td><td>{np.mean(x < lo[j]):.0%} / {np.mean(x > hi[j]):.0%}</td>"
                      f"<td>{used:.0%}</td>")
        rows.append(f"<tr><td>{ROWS[j // 4]} {NAMES[j]}</td><td>{mapping.LO[j]:.2f}..{mapping.HI[j]:.2f}</td>"
                    f"<td>{g[j]:.2f} / {o[j]:.2f}</td>{cells}</tr>")
    return (f"<table><thead><tr><th rowspan=2>joint</th><th rowspan=2>robot LO..HI</th>"
            f"<th rowspan=2>GAIN / OFFSET</th>{head}</tr><tr>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=DEFAULT)
    ap.add_argument("--calib", choices=list(mapping.CALIB), default=mapping.DEFAULT_CALIB)
    ap.add_argument("--out", help="default docs/results/teleop_rom_<calib>.html")
    args = ap.parse_args()

    args.out = args.out or f"docs/results/teleop_rom_{args.calib}.html"
    sets = [load(f) for f in args.files]
    js = os.path.relpath(PLOTLY_JS, Path(args.out).parent)
    meta = ", ".join(f"{n}: {len(q)} frames ({k:.1%} kept)" for n, q, k in sets)
    div = figure(sets, args.calib).to_html(include_plotlyjs=False, full_html=False, config={"displaylogo": False, "responsive": True})
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Teleop ROM</title>
<script src="{js}"></script>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#1f1e1c;background:#fff}}
table{{border-collapse:collapse;font-size:13px;margin-top:16px}}th,td{{border-bottom:1px solid #e3e2de;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.note{{color:#5f5e5a;font-size:14px}}</style></head><body>
<h1>Teleop range of motion &middot; calib <code>{args.calib}</code></h1>
<p class="note">Raw mapping angles (rad, <code>mapping.angles</code>, before GAIN/OFFSET/clip); density per recording.
Grey band = raw angles that reach the robot joint range under this preset&#39;s GAIN/OFFSET. Frames with Leap confidence
&le; {MIN_CONF} dropped. {meta}.</p>
{div}
<h2>Per joint</h2>
<p class="note">out lo / hi = frames clipped at the robot limit; robot range used = share of [LO, HI] covered by p2..p98.</p>
{table(sets, args.calib)}
</body></html>"""
    Path(args.out).write_text(html)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
