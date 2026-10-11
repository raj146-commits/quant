"""Download and parse holdings for the fund universe into the canonical schema."""

from __future__ import annotations

import logging

import pandas as pd

from .classify import is_clo_equity
from .edgar import EdgarClient, Filing
from .nport import parse_nport
from .soi import parse_schedule_of_investments
from .universe import Fund

log = logging.getLogger(__name__)


def _quarter_end(ts) -> pd.Timestamp | None:
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    qe = (ts + pd.offsets.QuarterEnd(0)).normalize()
    # only accept filings whose report date is a calendar quarter end
    return qe if qe == ts.normalize() else None


def fetch_fund(client: EdgarClient, fund: Fund, start: str = "2014-09-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    cik = fund.cik or client.cik_for_ticker(fund.ticker)
    start_ts = pd.Timestamp(start)
    frames, infos = [], []
    seen: set[pd.Timestamp] = set()
    for source in fund.sources:
        if source == "soi":
            filings = client.filings(cik, fund.soi_forms)
        elif source == "nport":
            filings = client.filings(cik, ("NPORT-P", "NPORT-P/A"))
        else:
            raise ValueError(source)
        latest: dict[pd.Timestamp, Filing] = {}
        for f in filings:
            q = _quarter_end(f.report_date)
            if q is None or q < start_ts or q in seen:
                continue
            latest[q] = f  # filings are date-sorted, so amendments win
        for q, f in sorted(latest.items()):
            try:
                if source == "nport":
                    h, info = parse_nport(client.nport_xml(f), fund.fund, f.filing_date, f.url)
                    infos.append(info)
                else:
                    doc = client.get(f.url)
                    allrows = parse_schedule_of_investments(doc, fund.fund, q, f.filing_date, f.form, f.url,
                                                            equity_only=False)
                    if allrows.empty:
                        log.warning("%s %s %s: no schedule of investments parsed", fund.fund, f.form, q.date())
                        continue
                    h = allrows[allrows.apply(lambda r: is_clo_equity(r["issuer"], r["tranche"]), axis=1)]
                    infos.append({"fund": fund.fund, "period_end": q,
                                  "total_investments": allrows["fair_value"].sum()})
            except Exception as exc:  # keep going; one bad filing should not stop the run
                log.warning("%s %s %s failed: %s", fund.fund, f.form, f.accession, exc)
                continue
            if not h.empty:
                h = h.assign(period_end=q)
                frames.append(h)
                seen.add(q)
    holdings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return holdings, pd.DataFrame(infos)


def fetch_universe(client: EdgarClient, funds: list[Fund], start: str = "2014-09-30"):
    hs, infos = [], []
    for fund in funds:
        if fund.cik is None and fund.ticker is None:
            log.warning("%s: no CIK or ticker configured, skipped", fund.fund)
            continue
        log.info("fetching %s", fund.fund)
        h, i = fetch_fund(client, fund, start)
        if not h.empty:
            h["firm"] = fund.firm
        hs.append(h)
        infos.append(i)
    return (pd.concat(hs, ignore_index=True) if hs else pd.DataFrame(),
            pd.concat(infos, ignore_index=True) if infos else pd.DataFrame())
