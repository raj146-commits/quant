# cloindex: CLO equity returns index from public SEC filings

This package rebuilds a quarterly **US CLO equity total-return index** using the construction method of the
[Flat Rock CLO Equity Returns Index](https://flatrockglobal.com/clo-equity-index/), with publicly available
data only.

## Methodology (mirrors Flat Rock's published overview)

| Rule | Flat Rock | This implementation |
|---|---|---|
| Objective | Unlevered, gross-of-fee performance of US CLO equity tranches | Same. Returns are computed on the underlying positions, not on fund NAVs, so fund leverage and fees are excluded |
| Universe | CLO equity held by 1940 Act funds with a significant CLO equity allocation that disclose holdings and fair values to the SEC | `universe.py` lists the candidate funds. Rows are kept only if they look like CLO equity (sub/income notes, preferred shares) and are US deals (Euro DAC/BV vehicles and non-USD deals are dropped). Warehouses and debt tranches are excluded |
| Timeliness | 10-Q filed within 75 days (10-K within 90) of the valuation date | `max_filing_lag_days` (default 90) |
| Significant allocation | Not quantified | `min_clo_equity_share` of total investments (default 25%) |
| Valuation dates | Calendar quarter ends | Same |
| Weighting | Reported fair market value | Beginning-of-quarter FMV |
| Position return | `(FMV_end + ownership share × tranche distributions) / FMV_begin − 1` | Same. Ending FMV is restated to the beginning-of-quarter unit count, so purchases and sales do not count as performance |
| Base | 100 at 9/30/2014, published on a ~75-day lag | Same (configurable) |

Flat Rock's methodology works the formula through an example: Ares XLIX, 7% of the tranche,
`(1,686,669 + 7% × 1,891,838) / 1,704,805 − 1 ≈ +7%`. That example is a unit test (`tests/test_index.py`).

### Distributions

SEC filings report marks, but they do not report the cash each position received. The engine tries two
methods for each position, in order:

1. **`deal`**: the fund's ownership share times the cash paid to the whole equity tranche. You supply a
   `distributions.csv` with columns `key, period_end, tranche_distribution, tranche_size` or
   `distribution_per_unit`. The data usually comes from trustee or payment-date reports. This is the
   approach Flat Rock describes.
2. **`cost_basis`**: an accounting identity. Funds book CLO equity income at an effective yield, so
   `cash = eff_yield × cost_begin × days/365 + (cost_begin − cost_end)`, computed per unit. This works from
   10-Q, 10-K and N-CSR schedules of investments that disclose amortized cost and effective yield. It is
   an estimate.

Positions with no distribution estimate are left out of the total return. They still count in the
separately published price return, and `distribution_coverage` reports the share of FMV covered.
`--allow-missing-distributions` treats missing cash as zero instead. Called or redeemed deals enter with
FMV_end = 0 when deal-level data shows their final payout.

## Data sources

| Source | Gives | Module |
|---|---|---|
| EDGAR submissions API (`data.sec.gov/submissions`) | filing list per fund | `edgar.py` |
| Form N-PORT (`primary_doc.xml`) | CUSIP/ISIN, units, USD fair value, asset category, fund total investments | `nport.py` |
| 10-Q / 10-K / N-CSR schedule of investments (HTML) | par, amortized cost, fair value, effective yield | `soi.py` (best-effort table parser: check its output against the filing) |
| Trustee / payment-date reports (optional) | tranche-level cash distributions and tranche size | `distributions.csv` |

Positions are matched across quarters on CUSIP, then ISIN, then a normalised issuer name.

## Usage

```bash
pip install -e .[test]
export SEC_USER_AGENT="Your Name you@example.com"   # required by SEC fair-access rules

cloindex -v fetch --out data/                        # all candidate funds, 2014-09-30 onward
cloindex build --holdings data/holdings.csv --fund-info data/fund_info.csv \
               [--distributions data/distributions.csv] --out output/
```

`output/` contains files laid out like Flat Rock's publication:

- `summary.csv`: index level, quarterly / 1Y / 3Y / 5Y returns, annualized return since inception,
  volatility, maximum quarterly drawdown and its quarter, CLO AUM at fair value, number of CLOs, and
  firms providing marks.
- `quarterly_returns.csv`: a year × Q1–Q4 grid plus the calendar-year return.
- `index_levels.csv`: per quarter, total- and price-return levels, coverage and counts.
- `position_returns.csv` and `fund_eligibility.csv`: the audit trail.
- `report.html`: all of the above as one page.

Holdings you curated yourself, in the schema in `schema.py`, can go straight to `build` (`--equity-only`
skips the classifier).

## Known differences from the Flat Rock index

- Flat Rock does not name its constituent funds, so `universe.py` is a candidate list to verify.
- Flat Rock has deal-level payment data for every position. Without it, the `cost_basis` estimate adds
  noise, and funds that publish no cost or yield (N-PORT-only quarters) need `distributions.csv`.
- HTML schedules of investments differ by fund and by year. Spot-check parsed holdings before using the
  results.

## Tests

```bash
python -m pytest -q
```
