"""Identify US CLO equity positions and build stable position keys."""

import re

import pandas as pd

# Descriptions CLO managers / funds use for the residual (equity) tranche.
_EQUITY_PAT = re.compile(
    r"\b(subordinated\s+notes?|sub(?:ordinated)?\.?\s+notes?|income\s+notes?|"
    r"preference\s+shares?|preferred\s+(?:shares?|securities|stock|notes?)|"
    r"class\s+m\s+notes?|residual|clo\s+equity|equity(?:\s+(?:notes?|tranche|interests?))?|"
    r"membership\s+interests?|certificates?)\b",
    re.I,
)
# Things that look like equity but are not CLO equity tranches.
_EXCLUDE_PAT = re.compile(
    r"\b(warehouse|loan\s+accumulation|accumulation\s+facility|"
    r"class\s+[a-f]-?\d?(?:-[a-z])?\s+(?:notes?|loans?)|senior|mezzanine|"
    r"term\s+loan|revolver|common\s+stock|money\s+market)\b",
    re.I,
)
# Non-US (mostly European) CLO issuer vehicles.
_EURO_PAT = re.compile(r"\b(dac|d\.a\.c\.|designated activity company|b\.v\.|bv|plc|s\.a\.)\b", re.I)
# Tokens removed when normalising issuer names into matching keys.
_NAME_NOISE = re.compile(
    r"\b(ltd|limited|llc|l\.l\.c|inc|corp|co|lp|the|clo|cdo|clos|subordinated|sub|notes?|"
    r"income|preferred|preference|shares?|equity|securities|class|m)\b|[^a-z0-9 ]",
    re.I,
)


def is_clo_equity(issuer: str | None, tranche: str | None = None) -> bool:
    text = " ".join(str(x) for x in (issuer, tranche) if isinstance(x, str))
    if not text:
        return False
    return bool(_EQUITY_PAT.search(text)) and not _EXCLUDE_PAT.search(text)


def is_us_clo(issuer: str | None, currency: str | None = "USD") -> bool:
    if currency and str(currency).upper() not in ("USD", "<NA>", "NAN", ""):
        return False
    return not (isinstance(issuer, str) and _EURO_PAT.search(issuer))


def normalise_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    s = _NAME_NOISE.sub(" ", name.lower())
    # "2014-1" and "2014 1" should match; roman numerals stay as-is
    s = re.sub(r"[-/]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def position_key(row: pd.Series) -> str:
    """CUSIP > ISIN > normalised issuer name."""
    for col in ("cusip", "isin"):
        v = row.get(col)
        if isinstance(v, str) and v.strip() and v.strip().upper() not in ("N/A", "NONE", "000000000"):
            return v.strip().upper()
    return "NAME:" + normalise_name(row.get("issuer"))


def add_keys(holdings: pd.DataFrame) -> pd.DataFrame:
    h = holdings.copy()
    h["key"] = h.apply(position_key, axis=1) if len(h) else pd.Series(dtype=str)
    return h


def filter_us_clo_equity(holdings: pd.DataFrame, already_equity: bool = False) -> pd.DataFrame:
    """Keep US CLO equity rows. Set `already_equity` when the parser has
    already selected equity tranches (e.g. a curated CSV)."""
    if holdings.empty:
        return holdings
    mask = holdings.apply(lambda r: is_us_clo(r["issuer"], r["currency"]), axis=1)
    if not already_equity:
        mask &= holdings.apply(lambda r: is_clo_equity(r["issuer"], r["tranche"]), axis=1)
    return holdings[mask].reset_index(drop=True)
