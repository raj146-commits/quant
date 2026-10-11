"""Canonical tabular schemas shared by the parsers and the index engine."""

import pandas as pd

# One row per (fund, valuation date, CLO equity position).
HOLDINGS_COLUMNS = [
    "fund",            # short fund identifier, e.g. "OXLC"
    "firm",            # fund manager (for the "firms providing marks" count)
    "period_end",      # valuation date (calendar quarter end)
    "filing_date",     # date the filing was accepted by EDGAR
    "form",            # 10-Q, 10-K, N-CSR, NPORT-P ...
    "issuer",          # CLO name as reported
    "tranche",         # tranche description (Subordinated Notes, Income Notes ...)
    "cusip",
    "isin",
    "currency",
    "par",             # units / principal held
    "cost",            # amortized cost (if disclosed)
    "fair_value",      # reported fair market value, USD
    "effective_yield", # disclosed effective yield, decimal (if disclosed)
    "ownership_pct",   # fund share of the equity tranche, decimal (if known)
    "source_url",
]

# Deal-level equity distributions (e.g. from trustee reports), one row per
# (position key, valuation quarter). Either `tranche_distribution` together
# with `tranche_size` (or a holdings-level `ownership_pct`), or
# `distribution_per_unit`, must be given.
DISTRIBUTION_COLUMNS = [
    "key",                    # cusip / isin / normalised issuer name
    "period_end",             # quarter in which the cash was received
    "tranche_distribution",   # total cash paid to the whole equity tranche
    "tranche_size",           # total par of the equity tranche
    "distribution_per_unit",  # cash per unit of par
]

NUMERIC_HOLDING_COLS = ["par", "cost", "fair_value", "effective_yield", "ownership_pct"]


def empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(columns=HOLDINGS_COLUMNS)


def normalise_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns, coerce types and drop rows without a fair value."""
    df = df.copy()
    for c in HOLDINGS_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in NUMERIC_HOLDING_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    for c in ["fund", "firm", "issuer", "tranche", "cusip", "isin", "currency", "form"]:
        df[c] = df[c].astype("string").str.strip()
    df["currency"] = df["currency"].fillna("USD").str.upper()
    df["firm"] = df["firm"].fillna(df["fund"])
    df = df[df["fair_value"].notna()]
    return df[HOLDINGS_COLUMNS].reset_index(drop=True)
