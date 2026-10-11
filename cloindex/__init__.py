"""CLO equity returns index built from public SEC filings.

Replicates the construction of the Flat Rock CLO Equity Returns Index:

* universe   - CLO equity positions held by 1940-Act funds with a significant
               CLO equity allocation that file holdings with the SEC on time
* valuation  - calendar quarter ends, reported fair market value (FMV)
* weighting  - beginning-of-quarter FMV (asset weighted)
* return     - (FMV_end + ownership share x tranche distributions) / FMV_begin - 1
* base       - 100 at 2014-09-30
"""

from .index import IndexResult, build_index, position_returns
from .report import summary_stats, quarterly_table, write_report

__all__ = [
    "IndexResult",
    "build_index",
    "position_returns",
    "summary_stats",
    "quarterly_table",
    "write_report",
]
