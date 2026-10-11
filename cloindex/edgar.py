"""Minimal SEC EDGAR client with on-disk caching and fair-access rate limiting.

SEC requires a descriptive User-Agent with contact details and at most
10 requests per second: https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"
TICKER_FILES = [
    "https://www.sec.gov/files/company_tickers_exchange.json",
    "https://www.sec.gov/files/company_tickers_mf.json",
]


@dataclass
class Filing:
    cik: int
    form: str
    accession: str
    filing_date: pd.Timestamp
    report_date: pd.Timestamp | None
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document)


class EdgarClient:
    def __init__(self, user_agent: str | None = None, cache_dir: str | Path = ".cache/edgar",
                 min_interval: float = 0.12):
        ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise ValueError(
                "SEC requires a User-Agent with contact info: pass user_agent= or set "
                "SEC_USER_AGENT='Your Name your@email.com'"
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0

    # -- transport -------------------------------------------------------
    def get(self, url: str, use_cache: bool = True) -> bytes:
        path = self.cache_dir / hashlib.sha1(url.encode()).hexdigest()
        if use_cache and path.exists():
            return path.read_bytes()
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.session.get(url, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            path.write_bytes(r.content)
            return r.content
        r.raise_for_status()
        return r.content

    def get_json(self, url: str, use_cache: bool = True) -> dict:
        return json.loads(self.get(url, use_cache))

    # -- lookups ---------------------------------------------------------
    def cik_for_ticker(self, ticker: str) -> int:
        t = ticker.upper()
        for url in TICKER_FILES:
            data = self.get_json(url)
            fields, rows = data["fields"], data["data"]
            ti, ci = fields.index("ticker"), fields.index("cik")
            for row in rows:
                if str(row[ti]).upper() == t:
                    return int(row[ci])
        raise KeyError(f"ticker {ticker} not found on EDGAR; give the CIK explicitly")

    def filings(self, cik: int, forms: tuple[str, ...]) -> list[Filing]:
        """All filings of the given form types, including older paginated ones."""
        sub = self.get_json(SUBMISSIONS_URL.format(cik=cik), use_cache=False)
        blocks = [sub["filings"]["recent"]]
        for f in sub["filings"].get("files", []):
            blocks.append(self.get_json("https://data.sec.gov/submissions/" + f["name"]))
        out: list[Filing] = []
        wanted = {f.upper() for f in forms}
        for b in blocks:
            for i, form in enumerate(b["form"]):
                if form.upper() not in wanted:
                    continue
                rd = b.get("reportDate", [None] * len(b["form"]))[i]
                out.append(Filing(
                    cik=cik,
                    form=form,
                    accession=b["accessionNumber"][i],
                    filing_date=pd.Timestamp(b["filingDate"][i]),
                    report_date=pd.Timestamp(rd) if rd else None,
                    primary_document=b["primaryDocument"][i],
                ))
        return sorted(out, key=lambda f: f.filing_date)

    def filing_documents(self, filing: Filing) -> list[str]:
        idx = self.get_json(INDEX_URL.format(cik=filing.cik, acc=filing.accession.replace("-", "")))
        return [item["name"] for item in idx["directory"]["item"]]

    def nport_xml(self, filing: Filing) -> bytes:
        """N-PORT filings carry the structured data in primary_doc.xml."""
        doc = "primary_doc.xml"
        url = ARCHIVE_URL.format(cik=filing.cik, acc=filing.accession.replace("-", ""), doc=doc)
        return self.get(url)
