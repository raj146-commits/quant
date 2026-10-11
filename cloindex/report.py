"""Output in the layout Flat Rock publishes: headline statistics, a
year x quarter return grid and the quarterly index level series."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


def qlabel(ts: pd.Timestamp) -> str:
    return f"{ts.quarter}Q{ts.year}"


def _ann(rets: pd.Series) -> float:
    rets = rets.dropna()
    if rets.empty:
        return np.nan
    return float((1 + rets).prod() ** (4 / len(rets)) - 1)


def summary_stats(index: pd.DataFrame) -> dict:
    df = index.sort_values("period_end").reset_index(drop=True)
    r = df["total_return"].iloc[1:]
    last = df.iloc[-1]
    worst = r.idxmin() if r.notna().any() else None
    lvl = df["index_level"]
    dd = lvl / lvl.cummax() - 1

    def trailing(n):
        tail = r.dropna().iloc[-n:]
        if len(tail) < n:
            return np.nan
        cum = float((1 + tail).prod() - 1)
        return cum if n == 4 else (1 + cum) ** (4 / n) - 1

    return {
        "As of": qlabel(last["period_end"]),
        "Index Level": float(last["index_level"]),
        "Quarterly Return": float(last["total_return"]),
        "One Year Return": trailing(4),
        "Three Year Return (ann.)": trailing(12),
        "Five Year Return (ann.)": trailing(20),
        f"Annualized Return since inception on {df['period_end'].iloc[0]:%m/%d/%Y}": _ann(r),
        "Annualized Volatility": float(r.std() * 2) if r.notna().sum() > 1 else np.nan,
        "Maximum Quarterly Drawdown": float(r.min()) if worst is not None else np.nan,
        "Maximum Quarterly Drawdown Period": qlabel(df.loc[worst, "period_end"]) if worst is not None else "",
        "Maximum Peak-to-Trough Drawdown": float(dd.min()),
        "CLO AUM at Fair Value": float(last["clo_aum_fair_value"]),
        "Number of CLOs": int(last["num_clos"]),
        "Firms Providing Marks": int(last["num_firms"]),
    }


def quarterly_table(index: pd.DataFrame) -> pd.DataFrame:
    df = index.dropna(subset=["total_return"]).copy()
    df["Year"] = df["period_end"].dt.year
    df["Q"] = "Q" + df["period_end"].dt.quarter.astype(str)
    grid = df.pivot(index="Year", columns="Q", values="total_return").reindex(columns=["Q1", "Q2", "Q3", "Q4"])
    grid["Year Total"] = df.groupby("Year")["total_return"].apply(lambda x: (1 + x).prod() - 1)
    return grid


def _fmt(v, kind="pct"):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "usd":
        return f"${v:,.0f}"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:,.2f}"


def _kind(label: str) -> str:
    if label in ("Index Level",):
        return "num"
    if label == "CLO AUM at Fair Value":
        return "usd"
    if label in ("Number of CLOs", "Firms Providing Marks"):
        return "int"
    return "pct"


def _svg_chart(index: pd.DataFrame, w=760, h=260, pad=40) -> str:
    df = index.sort_values("period_end")
    y = df["index_level"].to_numpy(float)
    if len(y) < 2:
        return ""
    lo, hi = y.min() * 0.95, y.max() * 1.05
    xs = pad + np.arange(len(y)) * (w - 2 * pad) / (len(y) - 1)
    ys = h - pad - (y - lo) / (hi - lo) * (h - 2 * pad)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    ticks = "".join(
        f'<text x="{xs[i]:.1f}" y="{h - pad + 16}" text-anchor="middle">{df["period_end"].iloc[i].year}</text>'
        for i in range(0, len(y), max(1, len(y) // 8)))
    yt = "".join(
        f'<text x="{pad - 6}" y="{h - pad - (v - lo) / (hi - lo) * (h - 2 * pad) + 4:.1f}" text-anchor="end">{v:.0f}</text>'
        for v in np.linspace(lo, hi, 5))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Index level">'
            f'<polyline fill="none" stroke="currentColor" stroke-width="2" points="{pts}"/>'
            f'<g font-size="11" fill="currentColor" opacity=".7">{ticks}{yt}</g></svg>')


def to_html(result, title="CLO Equity Returns Index (public-filings replication)") -> str:
    idx = result.index
    stats = summary_stats(idx)
    grid = quarterly_table(idx)
    stat_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{_fmt(v, _kind(k))}</td></tr>" for k, v in stats.items())
    grid_rows = "".join(
        f"<tr><th>{yr}</th>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for yr, row in grid.iterrows())
    series_rows = "".join(
        f"<tr><th>{qlabel(r.period_end)}</th><td>{r.index_level:.2f}</td><td>{_fmt(r.total_return)}</td>"
        f"<td>{_fmt(r.price_return)}</td><td>{_fmt(r.clo_aum_fair_value, 'usd')}</td><td>{r.num_clos}</td>"
        f"<td>{r.num_firms}</td><td>{_fmt(r.distribution_coverage)}</td></tr>"
        for r in idx.sort_values("period_end", ascending=False).itertuples())
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLO Equity Index</title><style>
body{{font-family:system-ui,sans-serif;margin:24px auto;max-width:880px;padding:0 16px;color:#1d2433;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid #e3e6ec;text-align:right}} th:first-child{{text-align:left}}
thead th{{background:#f4f6f9}} .note{{font-size:13px;color:#5b6475}} svg{{width:100%;height:auto}}
</style></head><body><h1>{html.escape(title)}</h1>
<p class="note">Unlevered, gross-of-fee total return of US CLO equity tranches held by 1940 Act funds that disclose
holdings and fair values to the SEC. Asset-weighted by beginning-of-quarter fair value; base 100 at
{idx['period_end'].iloc[0]:%m/%d/%Y}. Distributions via {', '.join(result.config.distribution_methods)}.</p>
<h2>Summary</h2><table>{stat_rows}</table>
<h2>Index level</h2>{_svg_chart(idx)}
<h2>Quarterly returns</h2><table><thead><tr><th>Year</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Year</th></tr></thead>{grid_rows}</table>
<h2>Quarterly detail</h2><table><thead><tr><th>Quarter</th><th>Level</th><th>Total</th><th>Price</th>
<th>CLO AUM at FV</th><th>CLOs</th><th>Firms</th><th>Dist. coverage</th></tr></thead>{series_rows}</table>
</body></html>"""


def write_report(result, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "index": out / "index_levels.csv",
        "quarterly_table": out / "quarterly_returns.csv",
        "summary": out / "summary.csv",
        "positions": out / "position_returns.csv",
        "eligibility": out / "fund_eligibility.csv",
        "html": out / "report.html",
    }
    result.index.to_csv(paths["index"], index=False)
    quarterly_table(result.index).to_csv(paths["quarterly_table"])
    pd.Series(summary_stats(result.index)).to_csv(paths["summary"], header=["value"])
    result.positions.to_csv(paths["positions"], index=False)
    result.eligibility.to_csv(paths["eligibility"], index=False)
    paths["html"].write_text(to_html(result))
    return paths
