"""Candidate fund universe.

Flat Rock's index uses the CLO equity positions of 1940 Act funds that keep
a significant CLO equity allocation and file holdings with the SEC on time
(its site lists the count as "firms providing marks"; it does not name them).
The list below is the set of public CLO equity funds we know of. These are
candidates, not a confirmed match to Flat Rock's constituents: check each
fund's filings, form types and CLO equity share before relying on it. CIKs
are resolved from tickers at run time unless given.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fund:
    fund: str                  # short id used in output
    name: str
    firm: str                  # manager, for the "firms providing marks" count
    ticker: str | None = None
    cik: int | None = None
    # Sources tried in order. "soi" = HTML schedule of investments in the
    # forms listed in soi_forms; "nport" = NPORT-P XML.
    sources: tuple[str, ...] = ("soi", "nport")
    soi_forms: tuple[str, ...] = ("10-Q", "10-K", "N-CSR", "N-CSRS")


CANDIDATE_FUNDS: list[Fund] = [
    Fund("ECC", "Eagle Point Credit Company Inc.", "Eagle Point Credit Management", ticker="ECC"),
    Fund("EIC", "Eagle Point Income Company Inc.", "Eagle Point Credit Management", ticker="EIC"),
    Fund("OXLC", "Oxford Lane Capital Corp.", "Oxford Lane Management", ticker="OXLC"),
    Fund("OCCI", "OFS Credit Company, Inc.", "OFS Capital Management", ticker="OCCI"),
    Fund("CCIF", "Carlyle Credit Income Fund", "Carlyle Global Credit Investment Management", ticker="CCIF"),
    Fund("FROPX", "Flat Rock Opportunity Fund", "Flat Rock Global", ticker="FROPX"),
    # Non-listed funds: supply the CIK from EDGAR company search.
    Fund("PRIS", "Priority Income Fund, Inc.", "Prospect Capital Management", cik=None),
]
