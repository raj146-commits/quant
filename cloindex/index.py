"""Index engine: position returns -> asset-weighted quarterly index."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import distributions as dist_mod
from .classify import add_keys, filter_us_clo_equity
from .schema import normalise_holdings


@dataclass
class IndexConfig:
    base_date: str = "2014-09-30"
    base_value: float = 100.0
    # Fund eligibility: CLO equity FMV / total investments (needs fund_info).
    min_clo_equity_share: float = 0.25
    # Filing must be accepted within this many days of the valuation date.
    max_filing_lag_days: int = 90
    # Methods tried in order for each position's quarterly cash.
    distribution_methods: tuple[str, ...] = ("deal", "cost_basis")
    # If False, positions with no distribution estimate are dropped from the
    # total-return calculation (they still count in the price return).
    allow_missing_distributions: bool = False
    # Include positions that left the portfolio when deal-level data shows
    # their final payout (redemptions / calls). FMV_end is taken as 0.
    include_exits_with_payout: bool = True
    # Optional symmetric cap on single-position quarterly returns, e.g. 2.0.
    return_cap: float | None = None
    holdings_are_equity_only: bool = False


@dataclass
class IndexResult:
    index: pd.DataFrame          # one row per quarter
    positions: pd.DataFrame      # one row per position-quarter used
    holdings: pd.DataFrame       # cleaned eligible holdings
    eligibility: pd.DataFrame    # fund-quarter eligibility decisions
    config: IndexConfig = field(default_factory=IndexConfig)


def _quarter_end(ts: pd.Series) -> pd.Series:
    return (ts + pd.offsets.QuarterEnd(0)).dt.normalize()


def _aggregate(h: pd.DataFrame) -> pd.DataFrame:
    """Combine duplicate lines (same fund, quarter, position)."""
    h = h.assign(_yw=h["effective_yield"] * h["cost"])
    g = h.groupby(["fund", "period_end", "key"], as_index=False, dropna=False)
    out = g.agg(
        filing_date=("filing_date", "max"), firm=("firm", "first"), form=("form", "first"), issuer=("issuer", "first"),
        tranche=("tranche", "first"), cusip=("cusip", "first"), isin=("isin", "first"),
        par=("par", lambda x: x.sum(min_count=1)), cost=("cost", lambda x: x.sum(min_count=1)),
        fair_value=("fair_value", "sum"), _yw=("_yw", lambda x: x.sum(min_count=1)),
        ownership_pct=("ownership_pct", lambda x: x.sum(min_count=1)),
    )
    out["effective_yield"] = out["_yw"] / out["cost"]
    return out.drop(columns="_yw")


def eligibility(h: pd.DataFrame, fund_info: pd.DataFrame | None, cfg: IndexConfig) -> pd.DataFrame:
    fq = h.groupby(["fund", "period_end"], as_index=False).agg(
        clo_equity_fv=("fair_value", "sum"), filing_date=("filing_date", "max"))
    fq["filing_lag_days"] = (fq["filing_date"] - fq["period_end"]).dt.days
    fq["timely"] = fq["filing_lag_days"].isna() | (fq["filing_lag_days"] <= cfg.max_filing_lag_days)
    if fund_info is not None and not fund_info.empty:
        fi = fund_info.copy()
        fi["period_end"] = pd.to_datetime(fi["period_end"]).dt.normalize()
        fi = fi.groupby(["fund", "period_end"], as_index=False)["total_investments"].max()
        fq = fq.merge(fi, on=["fund", "period_end"], how="left")
        fq["clo_equity_share"] = fq["clo_equity_fv"] / fq["total_investments"]
        fq["significant"] = fq["clo_equity_share"].isna() | (fq["clo_equity_share"] >= cfg.min_clo_equity_share)
    else:
        fq["total_investments"] = np.nan
        fq["clo_equity_share"] = np.nan
        fq["significant"] = True
    fq["eligible"] = fq["timely"] & fq["significant"]
    return fq


def position_returns(h: pd.DataFrame, dist: pd.DataFrame | None, cfg: IndexConfig) -> pd.DataFrame:
    """Match each position to the same fund's prior-quarter holding."""
    cur = h.copy()
    cur["period_begin"] = (cur["period_end"] - pd.offsets.QuarterEnd(1)).dt.normalize()
    prev = h.rename(columns={c: f"{c}_begin" for c in
                             ["par", "cost", "fair_value", "effective_yield", "ownership_pct"]})
    prev = prev[["fund", "key", "period_end", "par_begin", "cost_begin", "fair_value_begin",
                 "effective_yield_begin", "ownership_pct_begin"]].rename(columns={"period_end": "period_begin"})
    pairs = cur.merge(prev, on=["fund", "key", "period_begin"], how="inner")
    pairs["status"] = "held"

    if cfg.include_exits_with_payout and dist is not None and not dist.empty:
        # prior-quarter holdings missing this quarter (sold / called / redeemed)
        nxt = prev.copy()
        nxt["period_end"] = (nxt["period_begin"] + pd.offsets.QuarterEnd(1)).dt.normalize()
        present = cur[["fund", "key", "period_end"]].assign(_p=1)
        funds_reporting = cur[["fund", "period_end"]].drop_duplicates()
        exits = nxt.merge(present, on=["fund", "key", "period_end"], how="left")
        exits = exits[exits["_p"].isna()].drop(columns="_p")
        exits = exits.merge(funds_reporting, on=["fund", "period_end"])  # fund did report
        meta = h[["fund", "key", "period_end", "issuer", "tranche", "cusip", "isin", "form"]].rename(
            columns={"period_end": "period_begin"})
        exits = exits.merge(meta, on=["fund", "key", "period_begin"], how="left")
        exits = exits.assign(par=0.0, cost=0.0, fair_value=0.0, effective_yield=np.nan,
                             ownership_pct=np.nan, status="exit")
        pairs = pd.concat([pairs, exits], ignore_index=True)

    pairs = pairs[pairs["fair_value_begin"] > 0].reset_index(drop=True)

    # Restate ending value to beginning units so purchases / partial sales
    # (assumed at the reported mark) do not show up as performance.
    scale = pairs["par_begin"] / pairs["par"]
    ok = scale.replace([np.inf, -np.inf], np.nan).notna() & (pairs["status"] == "held")
    pairs["units_changed"] = ok & ~np.isclose(scale.fillna(1), 1.0)
    pairs["fv_end_adj"] = np.where(ok, pairs["fair_value"] * scale, pairs["fair_value"])

    pairs["distribution"] = np.nan
    pairs["dist_method"] = pd.NA
    for method in cfg.distribution_methods:
        if method == "deal":
            est = dist_mod.deal_level(pairs, dist)
        elif method == "cost_basis":
            est = dist_mod.estimate_from_cost_basis(pairs).where(pairs["status"] == "held")
        else:
            raise ValueError(f"unknown distribution method {method}")
        fill = pairs["distribution"].isna() & est.notna() & np.isfinite(est)
        pairs.loc[fill, "distribution"] = est[fill]
        pairs.loc[fill, "dist_method"] = method

    # Exits are only usable when the payout is known.
    pairs = pairs[(pairs["status"] == "held") | pairs["distribution"].notna()].reset_index(drop=True)

    pairs["price_return"] = pairs["fv_end_adj"] / pairs["fair_value_begin"] - 1
    pairs["total_return"] = (pairs["fv_end_adj"] + pairs["distribution"]) / pairs["fair_value_begin"] - 1
    if cfg.allow_missing_distributions:
        pairs["total_return"] = pairs["total_return"].fillna(pairs["price_return"])
        pairs["dist_method"] = pairs["dist_method"].fillna("none")
    if cfg.return_cap is not None:
        for c in ("price_return", "total_return"):
            pairs[c] = pairs[c].clip(-1.0, cfg.return_cap)
    return pairs


def build_index(holdings: pd.DataFrame, distributions: pd.DataFrame | None = None,
                fund_info: pd.DataFrame | None = None, config: IndexConfig | None = None) -> IndexResult:
    cfg = config or IndexConfig()
    h = normalise_holdings(holdings)
    h["period_end"] = _quarter_end(h["period_end"])
    h = filter_us_clo_equity(h, already_equity=cfg.holdings_are_equity_only)
    h = add_keys(h)
    h = _aggregate(h)

    elig = eligibility(h, fund_info, cfg)
    ok = elig.loc[elig["eligible"], ["fund", "period_end"]]
    h = h.merge(ok, on=["fund", "period_end"]).sort_values(["period_end", "fund", "key"]).reset_index(drop=True)

    base = pd.Timestamp(cfg.base_date)
    pos = position_returns(h, distributions, cfg)
    pos = pos[pos["period_end"] > base]

    quarters = pd.date_range(base, h["period_end"].max(), freq="QE") if len(h) else pd.DatetimeIndex([base])
    rows = []
    level_tr = level_pr = cfg.base_value
    for q in quarters:
        snap = h[h["period_end"] == q]
        p = pos[pos["period_end"] == q]
        tr = p[p["total_return"].notna()]
        w_tr, w_pr = tr["fair_value_begin"], p["fair_value_begin"]
        r_tr = float((tr["total_return"] * w_tr).sum() / w_tr.sum()) if len(tr) else np.nan
        r_pr = float((p["price_return"] * w_pr).sum() / w_pr.sum()) if len(p) else np.nan
        if q > base:
            level_tr = level_tr * (1 + r_tr) if np.isfinite(r_tr) else level_tr
            level_pr = level_pr * (1 + r_pr) if np.isfinite(r_pr) else level_pr
        rows.append({
            "period_end": q,
            "total_return": r_tr if q > base else np.nan,
            "price_return": r_pr if q > base else np.nan,
            "index_level": level_tr,
            "price_index_level": level_pr,
            "clo_aum_fair_value": float(snap["fair_value"].sum()),
            "num_clos": int(snap["key"].nunique()),
            "num_firms": int(snap["firm"].nunique()),
            "num_positions_in_return": int(len(tr)),
            "distribution_coverage": float(w_tr.sum() / w_pr.sum()) if w_pr.sum() else np.nan,
            "cash_yield": float(tr["distribution"].sum() / w_tr.sum()) if len(tr) else np.nan,
        })
    idx = pd.DataFrame(rows)
    return IndexResult(index=idx, positions=pos.reset_index(drop=True), holdings=h, eligibility=elig, config=cfg)
