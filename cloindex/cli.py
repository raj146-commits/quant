"""Command line entry point.

    cloindex fetch --user-agent "Name email" --out data/
    cloindex build --holdings data/holdings.csv [--fund-info data/fund_info.csv]
                   [--distributions data/distributions.csv] --out output/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .index import IndexConfig, build_index
from .report import summary_stats, write_report


def _fetch(args):
    from .edgar import EdgarClient
    from .pipeline import fetch_universe
    from .universe import CANDIDATE_FUNDS

    funds = [f for f in CANDIDATE_FUNDS if not args.funds or f.fund in args.funds]
    client = EdgarClient(args.user_agent, cache_dir=args.cache)
    holdings, info = fetch_universe(client, funds, start=args.start)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    holdings.to_csv(out / "holdings.csv", index=False)
    info.to_csv(out / "fund_info.csv", index=False)
    print(f"wrote {len(holdings)} holdings rows for {holdings['fund'].nunique() if len(holdings) else 0} funds to {out}")


def _build(args):
    holdings = pd.read_csv(args.holdings)
    info = pd.read_csv(args.fund_info) if args.fund_info else None
    dist = pd.read_csv(args.distributions) if args.distributions else None
    cfg = IndexConfig(
        base_date=args.base_date,
        min_clo_equity_share=args.min_share,
        max_filing_lag_days=args.max_lag,
        distribution_methods=tuple(args.methods.split(",")),
        allow_missing_distributions=args.allow_missing_distributions,
        holdings_are_equity_only=args.equity_only,
    )
    res = build_index(holdings, dist, info, cfg)
    paths = write_report(res, args.out)
    for k, v in summary_stats(res.index).items():
        print(f"{k:>55}: {v:.4f}" if isinstance(v, float) else f"{k:>55}: {v}")
    print(f"report: {paths['html']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cloindex")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download holdings from SEC EDGAR")
    f.add_argument("--user-agent", help="'Name email' (or set SEC_USER_AGENT)")
    f.add_argument("--funds", nargs="*", help="subset of fund ids from universe.py")
    f.add_argument("--start", default="2014-09-30")
    f.add_argument("--cache", default=".cache/edgar")
    f.add_argument("--out", default="data")
    f.set_defaults(func=_fetch)

    b = sub.add_parser("build", help="compute the index from holdings")
    b.add_argument("--holdings", required=True)
    b.add_argument("--fund-info")
    b.add_argument("--distributions")
    b.add_argument("--base-date", default="2014-09-30")
    b.add_argument("--min-share", type=float, default=0.25)
    b.add_argument("--max-lag", type=int, default=90)
    b.add_argument("--methods", default="deal,cost_basis")
    b.add_argument("--allow-missing-distributions", action="store_true")
    b.add_argument("--equity-only", action="store_true", help="holdings already curated to CLO equity")
    b.add_argument("--out", default="output")
    b.set_defaults(func=_build)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
