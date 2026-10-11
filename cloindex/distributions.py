"""Cash distributions received by each position during a quarter.

Two public-data methods, used in priority order:

``deal``        Flat Rock's approach: the fund's ownership share of the equity
                tranche times the total cash the tranche received
                (trustee / payment-date reports), i.e.
                    dist = (par_held / tranche_size) * tranche_distribution
                or, equivalently, distribution_per_unit * par_held.

``cost_basis``  An accounting identity on the schedule of investments. Funds
                book CLO equity income with the effective-yield method, so
                    cost_end = cost_begin + income - cash
                    cash     = eff_yield * cost_begin * days/365 + (cost_begin - cost_end)
                computed per unit of par to strip out purchases / sales. It
                is an estimate: yields reset during the quarter and trades at
                prices away from cost add noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def deal_level(pairs: pd.DataFrame, dist: pd.DataFrame | None) -> pd.Series:
    """Distribution per position from deal-level data (NaN when unavailable)."""
    out = pd.Series(np.nan, index=pairs.index)
    if dist is None or dist.empty or pairs.empty:
        return out
    d = dist.copy()
    d["period_end"] = pd.to_datetime(d["period_end"]).dt.normalize()
    d["key"] = d["key"].astype(str).str.upper()
    for c in ("tranche_distribution", "tranche_size", "distribution_per_unit"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d.groupby(["key", "period_end"], as_index=False).agg(
        tranche_distribution=("tranche_distribution", lambda x: x.sum(min_count=1)),
        tranche_size=("tranche_size", "max"),
        distribution_per_unit=("distribution_per_unit", lambda x: x.sum(min_count=1)),
    )
    m = pairs[["key", "period_end", "par_begin", "ownership_pct_begin"]].reset_index().merge(
        d, on=["key", "period_end"], how="left").set_index("index")

    share = pd.to_numeric(m["ownership_pct_begin"], errors="coerce")
    share = share.where(share.notna(), m["par_begin"] / m["tranche_size"])
    by_share = share * m["tranche_distribution"]
    by_unit = m["distribution_per_unit"] * m["par_begin"]
    res = by_unit.where(by_unit.notna(), by_share)
    return res.reindex(pairs.index)


def estimate_from_cost_basis(pairs: pd.DataFrame) -> pd.Series:
    days = (pairs["period_end"] - pairs["period_begin"]).dt.days
    units = pairs["par_begin"]
    cpu_b = pairs["cost_begin"] / units
    cpu_e = pairs["cost"] / pairs["par"]
    income = pairs["effective_yield_begin"] * cpu_b * days / 365.0
    cash = (income + (cpu_b - cpu_e)) * units
    return cash.clip(lower=0)
