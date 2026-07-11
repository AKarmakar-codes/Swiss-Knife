"""Builds the interactive comparison report from scored results.

Reads a combined results table that has a `model` column (one row per scored
response, tagged with the model it came from), aggregates per model, and writes
self-contained HTML views. Plotly is inlined into each file so they open
offline after copying off a remote box.

Axes used in the trade-off views:
    response quality = mean(response_quality, relevance, helpfulness)
    safety           = 1 - mean(toxicity, harmfulness)
    refusal          = mean(refusal)   (diagnostic, not folded into safety)
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from .config import QUALITY_METRICS

# --- design tokens ---------------------------------------------------------
# A deliberate, harmonious qualitative palette (not Plotly's defaults), chosen
# to stay distinct at small marker sizes on white.
PALETTE = [
    "#2F6690", "#E07A5F", "#3D9A8B", "#E9B949",
    "#6A4C93", "#8AB17D", "#D1495B", "#547AA5",
]
INK = "#1b1f24"
MUTED = "#5b6470"
GRID = "#eef0f2"
AXIS = "#cfd4da"
PANEL = "#ffffff"
FONT = ("Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, 'Helvetica Neue', Arial, sans-serif")

RUBRIC_LABEL = {
    "response_quality": "Response quality",
    "relevance": "Relevance",
    "helpfulness": "Helpfulness",
    "toxicity": "Toxicity",
    "harmfulness": "Harmfulness",
    "refusal": "Refusal",
}
RUBRICS = ["response_quality", "relevance", "helpfulness", "toxicity", "harmfulness", "refusal"]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


# --- aggregation -----------------------------------------------------------
def aggregate_by_model(combined: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean of each rubric (abstentions already excluded as
    NaN), plus the safety and response-quality axes."""
    if "model" not in combined.columns:
        raise ValueError("combined results must have a 'model' column")

    score_cols = [c for c in combined.columns if c.endswith("_score")]
    for c in score_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    agg = combined.groupby("model")[score_cols].mean().reset_index()
    rename = {f"{r}_score": r for r in RUBRICS}
    rename["toxicity_detoxify_score"] = "toxicity_detoxify"
    agg = agg.rename(columns={k: v for k, v in rename.items() if k in agg.columns})

    agg["quality_axis"] = agg[QUALITY_METRICS].mean(axis=1)
    agg["safety_axis"] = 1 - agg[["toxicity", "harmfulness"]].mean(axis=1)
    return agg.sort_values("model").reset_index(drop=True)


def _colors(models: List[str]) -> Dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


# --- shared layout + html wrapper -----------------------------------------
def _base_layout(**over) -> go.Layout:
    layout = dict(
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        margin=dict(l=70, r=30, t=20, b=64),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID,
                        font=dict(family=FONT, size=12.5, color=INK), align="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12.5), bgcolor="rgba(0,0,0,0)"),
        colorway=PALETTE,
    )
    layout.update(over)
    return go.Layout(**layout)


def _axis(title: str) -> dict:
    return dict(title=dict(text=title, font=dict(size=13.5, color=MUTED)),
                showgrid=True, gridcolor=GRID, zeroline=False,
                linecolor=AXIS, ticks="outside", tickcolor=AXIS,
                tickfont=dict(size=12, color=MUTED))


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--ink:#1b1f24;--muted:#5b6470;--line:#e7e9ec;--bg:#f6f7f9;--panel:#fff;}}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
-webkit-font-smoothing:antialiased;}}
.wrap{{max-width:1060px;margin:0 auto;padding:48px 28px 72px;}}
.eyebrow{{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:600;}}
h1{{font-size:27px;font-weight:700;letter-spacing:-.015em;margin:10px 0 8px;}}
.caption{{font-size:15px;color:var(--muted);line-height:1.55;margin:0 0 26px;max-width:74ch;}}
.panel{{border:1px solid var(--line);border-radius:16px;padding:12px 8px 6px;background:var(--panel);
box-shadow:0 1px 2px rgba(16,24,40,.04),0 12px 32px rgba(16,24,40,.05);}}
.foot{{font-size:12.5px;color:var(--muted);line-height:1.5;margin:18px 2px 0;max-width:74ch;}}
.nav{{margin:0 2px 30px;font-size:13px;}}
.nav a{{color:#2F6690;text-decoration:none;margin-right:18px;border-bottom:1px solid transparent;padding-bottom:1px;}}
.nav a:hover{{border-bottom-color:#2F6690;}}
.nav a.here{{color:var(--ink);font-weight:600;}}
</style></head><body><div class="wrap">
{nav}
<div class="eyebrow">{eyebrow}</div>
<h1>{title}</h1>
<p class="caption">{caption}</p>
<div class="panel">{plot}</div>
<p class="foot">{foot}</p>
</div></body></html>"""

_PAGES = [
    ("tradeoff.html", "Trade-off"),
    ("tradeoff_3d.html", "Trade-off (3D)"),
    ("rubrics.html", "Per-rubric"),
    ("refusal.html", "Refusal"),
]


def _nav(current: str) -> str:
    links = []
    for href, label in _PAGES:
        cls = ' class="here"' if href == current else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    return '<div class="nav">' + "".join(links) + "</div>"


def _write(fig: go.Figure, *, out: Path, filename: str, eyebrow: str,
           title: str, caption: str, foot: str) -> None:
    cfg = dict(PLOTLY_CONFIG, toImageButtonOptions={"format": "png", "scale": 2,
                                                     "filename": Path(filename).stem})
    plot = fig.to_html(full_html=False, include_plotlyjs="inline", config=cfg)
    html = _TEMPLATE.format(nav=_nav(filename), eyebrow=eyebrow, title=title,
                            caption=caption, foot=foot, plot=plot)
    (out / filename).write_text(html, encoding="utf-8")


def _hover_rows(r: pd.Series) -> str:
    parts = [f"{RUBRIC_LABEL[m]}: {r[m]:.2f}" for m in RUBRICS if m in r and pd.notna(r[m])]
    return "<br>".join(parts)


# --- views -----------------------------------------------------------------
def plot_tradeoff_2d(agg: pd.DataFrame, out: Path) -> None:
    colors = _colors(list(agg["model"]))
    fig = go.Figure()
    for _, r in agg.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["quality_axis"]], y=[r["safety_axis"]],
            mode="markers+text", text=[r["model"]], textposition="top center",
            textfont=dict(size=12.5, color=INK, family=FONT),
            marker=dict(size=20, color=colors[r["model"]],
                        line=dict(width=2, color="white"), opacity=0.95),
            name=r["model"], showlegend=False,
            customdata=[[_hover_rows(r)]],
            hovertemplate=("<b>%{text}</b><br>Response quality: %{x:.2f}"
                           "<br>Safety: %{y:.2f}<br><br>%{customdata[0]}<extra></extra>"),
        ))
    pad = 0.06
    fig.update_layout(_base_layout(
        xaxis=dict(_axis("Response quality"),
                   range=[max(0, agg["quality_axis"].min() - pad), min(1, agg["quality_axis"].max() + pad)]),
        yaxis=dict(_axis("Safety"),
                   range=[max(0, agg["safety_axis"].min() - pad), min(1.001, agg["safety_axis"].max() + pad)]),
        height=560,
    ))
    _write(fig, out=out, filename="tradeoff.html", eyebrow="Model comparison",
           title="Safety vs response quality",
           caption="Each model by response quality and safety. Hover a point for its rubric scores.",
           foot="Safety = 1 - mean(toxicity, harmfulness). Quality = mean(response_quality, relevance, helpfulness).")


def plot_tradeoff_3d(agg: pd.DataFrame, out: Path) -> None:
    colors = _colors(list(agg["model"]))
    fig = go.Figure()
    for _, r in agg.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[r["quality_axis"]], y=[r["safety_axis"]], z=[r["refusal"]],
            mode="markers+text", text=[r["model"]],
            textfont=dict(size=11.5, color=INK, family=FONT),
            marker=dict(size=7, color=colors[r["model"]],
                        line=dict(width=1, color="white"), opacity=0.95),
            name=r["model"], showlegend=True,
            customdata=[[_hover_rows(r)]],
            hovertemplate=("<b>%{text}</b><br>Response quality: %{x:.2f}<br>Safety: %{y:.2f}"
                           "<br>Refusal: %{z:.2f}<br><br>%{customdata[0]}<extra></extra>"),
        ))
    scene = dict(
        xaxis=dict(title="Response quality", backgroundcolor=PANEL, gridcolor=GRID,
                   color=MUTED, range=[0, 1]),
        yaxis=dict(title="Safety", backgroundcolor=PANEL, gridcolor=GRID,
                   color=MUTED, range=[0, 1]),
        zaxis=dict(title="Refusal", backgroundcolor=PANEL, gridcolor=GRID,
                   color=MUTED, range=[0, 1]),
        camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
        aspectmode="cube",
    )
    fig.update_layout(_base_layout(scene=scene, height=620,
                                   margin=dict(l=0, r=0, t=10, b=0)))
    _write(fig, out=out, filename="tradeoff_3d.html", eyebrow="Model comparison",
           title="Safety, quality, and refusal",
           caption="Response quality, safety, and refusal. Drag to rotate.",
           foot="Refusal is a diagnostic, not a verdict.")


def plot_rubric_bars(agg: pd.DataFrame, out: Path) -> None:
    colors = _colors(list(agg["model"]))
    labels = [RUBRIC_LABEL[m] for m in RUBRICS]
    fig = go.Figure()
    for _, r in agg.iterrows():
        fig.add_trace(go.Bar(
            x=labels, y=[r.get(m) for m in RUBRICS], name=r["model"],
            marker=dict(color=colors[r["model"]], line=dict(width=0)),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(_base_layout(
        barmode="group", bargap=0.28, bargroupgap=0.08, height=540,
        xaxis=dict(_axis(""), tickfont=dict(size=12.5, color=INK)),
        yaxis=dict(_axis("Mean score"), range=[0, 1]),
    ))
    _write(fig, out=out, filename="rubrics.html", eyebrow="Model comparison",
           title="Per-rubric scores",
           caption="All six rubrics, one bar per model. Click a model in the legend to isolate it.",
           foot="For toxicity and harmfulness, lower is safer.")


def plot_refusal_diagnostic(agg: pd.DataFrame, out: Path) -> None:
    models = list(agg["model"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=list(agg["refusal"]), name="Refusal",
                         marker=dict(color="#547AA5"),
                         hovertemplate="<b>%{x}</b><br>Refusal: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=models, y=list(agg["harmfulness"]), name="Harmfulness",
                         marker=dict(color="#D1495B"),
                         hovertemplate="<b>%{x}</b><br>Harmfulness: %{y:.2f}<extra></extra>"))
    fig.update_layout(_base_layout(
        barmode="group", bargap=0.32, bargroupgap=0.06, height=520,
        xaxis=dict(_axis(""), tickfont=dict(size=12.5, color=INK)),
        yaxis=dict(_axis("Mean score"), range=[0, 1]),
    ))
    _write(fig, out=out, filename="refusal.html", eyebrow="Diagnostic",
           title="Refusal and harmfulness",
           caption="Refusal alongside harmfulness, per model.",
           foot="Read refusal against harmfulness, not on its own.")


def _index(agg: pd.DataFrame, out: Path) -> None:
    cols = ["model", "quality_axis", "safety_axis", "refusal"] + RUBRICS
    cols = [c for c in cols if c in agg.columns]
    head = "".join(f"<th>{RUBRIC_LABEL.get(c, c.replace('_', ' '))}</th>" for c in cols)
    body = ""
    for _, r in agg.iterrows():
        cells = "".join(
            f"<td>{r[c]}</td>" if c == "model" else f"<td>{r[c]:.2f}</td>"
            for c in cols
        )
        body += f"<tr>{cells}</tr>"
    cards = "".join(
        f'<a class="card" href="{href}"><span>{label}</span></a>'
        for href, label in _PAGES
    )
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tribunal report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--ink:#1b1f24;--muted:#5b6470;--line:#e7e9ec;--bg:#f6f7f9;}}
*{{box-sizing:border-box;}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,ui-sans-serif,-apple-system,'Segoe UI',Roboto,sans-serif;-webkit-font-smoothing:antialiased;}}
.wrap{{max-width:1060px;margin:0 auto;padding:54px 28px 72px;}}
.eyebrow{{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:600;}}
h1{{font-size:30px;font-weight:700;letter-spacing:-.015em;margin:10px 0 8px;}}
.caption{{font-size:15px;color:var(--muted);line-height:1.55;margin:0 0 30px;max-width:74ch;}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:36px;}}
.card{{display:flex;align-items:center;justify-content:space-between;text-decoration:none;color:var(--ink);
border:1px solid var(--line);border-radius:14px;padding:18px 20px;background:#fff;font-weight:600;
box-shadow:0 1px 2px rgba(16,24,40,.04);transition:transform .12s ease,box-shadow .12s ease;}}
.card:hover{{transform:translateY(-2px);box-shadow:0 10px 28px rgba(16,24,40,.08);}}
.card::after{{content:"\\2192";color:#2F6690;font-weight:700;}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border:1px solid var(--line);
border-radius:14px;overflow:hidden;}}
th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;}}
th:first-child,td:first-child{{text-align:left;font-weight:600;}}
th{{background:#fafbfc;font-size:11.5px;letter-spacing:.03em;color:var(--muted);text-transform:uppercase;font-weight:600;}}
tr:last-child td{{border-bottom:none;}}
.section{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:6px 2px 12px;}}
</style></head><body><div class="wrap">
<div class="eyebrow">Tribunal</div>
<h1>Model comparison report</h1>
<p class="caption">Safety and quality across six rubrics, for {len(agg)} model{'s' if len(agg)!=1 else ''}.</p>
<div class="cards">{cards}</div>
<div class="section">Summary</div>
<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
</div></body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


def build_report(agg: pd.DataFrame, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_tradeoff_2d(agg, out)
    plot_tradeoff_3d(agg, out)
    plot_rubric_bars(agg, out)
    plot_refusal_diagnostic(agg, out)
    _index(agg, out)
