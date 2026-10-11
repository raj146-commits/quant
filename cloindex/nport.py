"""Parse Form N-PORT (NPORT-P) XML into the canonical holdings schema.

N-PORT reports, per holding: name, title, CUSIP / ISIN, balance (units),
currency, value in USD and asset / issuer category. It does not report cost
or income, so N-PORT supplies marks and units; distributions must come from
another source (see distributions.py).
"""

from __future__ import annotations

import pandas as pd
from lxml import etree

from .classify import is_clo_equity

# asset categories CLO equity is usually filed under
_CLO_ASSET_CATS = {"ABS-CBDO", "ABS-O", "ABS-OTHER", "EC", "EP", "DBT", "OTHER"}


def _text(el, path: str) -> str | None:
    found = el.xpath(path)
    if not found:
        return None
    v = found[0]
    return (v if isinstance(v, str) else v.text or "").strip() or None


def parse_nport(xml: bytes, fund: str, filing_date=None, source_url: str | None = None,
                equity_only: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (holdings, fund_info). fund_info has total investments/net assets
    used for the eligibility test."""
    root = etree.fromstring(xml)
    # Strip namespaces so xpath stays readable across N-PORT schema versions.
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    period = _text(root, "//genInfo/repPdDate/text()")
    info = {
        "fund": fund,
        "period_end": pd.Timestamp(period) if period else None,
        "total_assets": pd.to_numeric(_text(root, "//fundInfo/totAssets/text()"), errors="coerce"),
        "net_assets": pd.to_numeric(_text(root, "//fundInfo/netAssets/text()"), errors="coerce"),
    }

    rows, total_value = [], 0.0
    for sec in root.iter("invstOrSec"):
        value = pd.to_numeric(_text(sec, "valUSD/text()"), errors="coerce")
        if pd.notna(value):
            total_value += float(value)
        name, title = _text(sec, "name/text()"), _text(sec, "title/text()")
        asset_cat = (_text(sec, "assetCat/text()") or _text(sec, "assetConditional/@desc") or "").upper()
        if equity_only and not (is_clo_equity(name, title) and (not asset_cat or asset_cat in _CLO_ASSET_CATS)):
            continue
        isin = _text(sec, "identifiers/isin/@value")
        rows.append({
            "fund": fund,
            "period_end": info["period_end"],
            "filing_date": filing_date,
            "form": "NPORT-P",
            "issuer": name,
            "tranche": title,
            "cusip": _text(sec, "cusip/text()"),
            "isin": isin,
            "currency": _text(sec, "curCd/text()") or "USD",
            "par": pd.to_numeric(_text(sec, "balance/text()"), errors="coerce"),
            "cost": None,
            "fair_value": value,
            "effective_yield": None,
            "ownership_pct": None,
            "source_url": source_url,
        })
    info["total_investments"] = total_value
    return pd.DataFrame(rows), info
