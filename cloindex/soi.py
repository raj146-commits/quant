"""Best-effort parser for HTML Schedules of Investments (10-Q, 10-K, N-CSR).

Listed CLO equity funds (e.g. those filing 10-Q / 10-K) disclose, for each CLO
equity position, principal amount, amortized cost, fair value and the
effective yield used for income recognition. Those cost / yield columns let
us estimate cash distributions (see distributions.estimate_from_cost_basis).

EDGAR HTML tables are irregular (spacer cells, "$" in separate cells,
headers spanning several columns), so this parser:

1. expands colspans into a grid,
2. finds a header row containing "fair value" plus "cost" or "principal",
3. for every later row reads each target field from the first numeric cell
   inside that header's column span,
4. forward-fills the issuer column (issuer names often sit on their own row).

Always review parsed output against the filing - layouts differ by fund and
year. Curated CSVs in the canonical schema can be fed to the engine directly.
"""

from __future__ import annotations

import re

import pandas as pd
from lxml import html as lxml_html

from .classify import is_clo_equity

_FIELDS = {
    "tranche": re.compile(r"(investment\s+type|type\s+of\s+investment|^investment$|class|tranche|security\s+type)", re.I),
    "issuer": re.compile(r"^(issuer|company|investment|portfolio\s+(company|investment)|security|description)", re.I),
    "par": re.compile(r"\b(principal|par|shares|units|face)\b", re.I),
    "cost": re.compile(r"(amortized\s+)?cost", re.I),
    "fair_value": re.compile(r"fair\s+value|market\s+value", re.I),
    "effective_yield": re.compile(r"(effective|estimated)\s+yield|yield", re.I),
    "cusip": re.compile(r"cusip", re.I),
}
_NUM = re.compile(r"^\(?-?\$?\s*[\d,]+(\.\d+)?\s*%?\)?$")


def _num(s: str) -> float | None:
    s = s.replace("—", "").replace("–", "").replace("$", "").replace("\xa0", " ").strip()
    if not s or not _NUM.match(s.replace(" ", "")):
        return None
    neg = s.startswith("(") and s.endswith(")")
    pct = s.endswith("%") or s.endswith("%)")
    v = float(re.sub(r"[^\d.]", "", s) or "nan")
    v = -v if neg else v
    return v / 100 if pct else v


def _grid(table) -> list[list[str]]:
    rows = []
    for tr in table.xpath(".//tr"):
        row = []
        for td in tr.xpath("./td|./th"):
            text = " ".join(td.text_content().split())
            span = int(td.get("colspan", "1") or 1)
            row.extend([text] + [""] * (span - 1))
        rows.append(row)
    return rows


def _header_map(row: list[str]) -> dict[str, tuple[int, int]] | None:
    starts = [(i, c) for i, c in enumerate(row) if c]
    if not any(_FIELDS["fair_value"].search(c) for _, c in starts):
        return None
    if not any(_FIELDS["cost"].search(c) or _FIELDS["par"].search(c) for _, c in starts):
        return None
    out: dict[str, tuple[int, int]] = {}
    for n, (i, c) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(row) + 20
        for field, pat in _FIELDS.items():
            # "amortized cost" must not be read as "cost" of fair value etc.
            if field not in out and pat.search(c):
                if field == "cost" and _FIELDS["fair_value"].search(c):
                    continue
                if field == "par" and _FIELDS["fair_value"].search(c):
                    continue
                out[field] = (i, end)
                break
    if "issuer" not in out:
        out["issuer"] = (0, starts[0][0] if starts and starts[0][0] > 0 else 1)
    return out


def parse_schedule_of_investments(doc: bytes | str, fund: str, period_end, filing_date=None,
                                  form: str | None = None, source_url: str | None = None,
                                  equity_only: bool = True) -> pd.DataFrame:
    tree = lxml_html.fromstring(doc)
    records = []
    for table in tree.xpath("//table"):
        grid = _grid(table)
        hdr, issuer = None, None
        for row in grid:
            m = _header_map(row)
            if m:
                hdr = m
                continue
            if hdr is None or not any(row):
                continue

            def cell(field, numeric=True):
                if field not in hdr:
                    return None
                a, b = hdr[field]
                for c in row[a:b]:
                    if not c or c in ("$", "%"):
                        continue
                    if numeric:
                        v = _num(c)
                        if v is not None:
                            return v
                    else:
                        return c
                return None

            name = cell("issuer", numeric=False)
            fv = cell("fair_value")
            if name and fv is None and not re.match(r"^(total|subtotal)", name, re.I):
                issuer = name  # issuer heading row
                continue
            if fv is None or (name and re.match(r"^(total|subtotal)", name, re.I)):
                continue
            tranche = cell("tranche", numeric=False)
            iss = name if name and not _looks_like_tranche(name) else issuer
            if name and _looks_like_tranche(name) and not tranche:
                tranche = name
            records.append({
                "fund": fund, "period_end": pd.Timestamp(period_end), "filing_date": filing_date,
                "form": form, "issuer": iss, "tranche": tranche,
                "cusip": cell("cusip", numeric=False), "isin": None, "currency": "USD",
                "par": cell("par"), "cost": cell("cost"), "fair_value": fv,
                "effective_yield": cell("effective_yield"), "ownership_pct": None,
                "source_url": source_url,
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # SOIs usually state values in whole dollars; yields > 1 were not % signed
    if "effective_yield" in df:
        ey = pd.to_numeric(df["effective_yield"], errors="coerce")
        df["effective_yield"] = ey.where(ey.isna() | (ey <= 1.5), ey / 100)
    if equity_only:
        df = df[df.apply(lambda r: is_clo_equity(r["issuer"], r["tranche"]), axis=1)]
    return df.reset_index(drop=True)


def _looks_like_tranche(s: str) -> bool:
    return bool(re.match(r"^(subordinated|income|preferred|preference|class\s|cl\s|equity)", s, re.I))
