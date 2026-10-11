import numpy as np
import pandas as pd
import pytest

from cloindex.classify import is_clo_equity, is_us_clo, normalise_name
from cloindex.index import IndexConfig, build_index
from cloindex.nport import parse_nport
from cloindex.report import quarterly_table, summary_stats, to_html
from cloindex.soi import parse_schedule_of_investments


def H(fund, q, issuer, par, fv, cost=None, ey=None, cusip=None, filed=None, **kw):
    q = pd.Timestamp(q)
    return dict(fund=fund, period_end=q, filing_date=filed or q + pd.Timedelta(days=60), form="10-Q",
                issuer=issuer, tranche="Subordinated Notes", cusip=cusip, par=par, fair_value=fv,
                cost=cost, effective_yield=ey, **kw)


def test_methodology_worked_example():
    """Flat Rock's methodology example: Ares XLIX, 7% of the tranche,
    FMV 1,704,805 -> 1,686,669 with 1,891,838 distributed to the tranche."""
    hold = pd.DataFrame([
        H("F", "2014-09-30", "Ares XLIX CLO Ltd", 7_000_000, 1_704_805, cusip="ARES49SUB"),
        H("F", "2014-12-31", "Ares XLIX CLO Ltd", 7_000_000, 1_686_669, cusip="ARES49SUB"),
    ])
    dist = pd.DataFrame([dict(key="ARES49SUB", period_end="2014-12-31",
                              tranche_distribution=1_891_838, tranche_size=100_000_000)])
    res = build_index(hold, dist)
    r = res.index.set_index("period_end").loc["2014-12-31", "total_return"]
    expected = (1_686_669 + 0.07 * 1_891_838) / 1_704_805 - 1
    assert r == pytest.approx(expected)
    assert round(r, 2) == 0.07
    assert res.index["index_level"].iloc[-1] == pytest.approx(100 * (1 + expected))


def test_asset_weighting_and_unit_adjustment():
    hold = pd.DataFrame([
        H("A", "2015-03-31", "Alpha CLO 2014-1", 10e6, 6e6, cusip="X1"),
        H("A", "2015-06-30", "Alpha CLO 2014-1", 20e6, 13.2e6, cusip="X1"),   # doubled units at the mark
        H("B", "2015-03-31", "Beta CLO", 5e6, 2e6, cusip="X2"),
        H("B", "2015-06-30", "Beta CLO", 5e6, 1.8e6, cusip="X2"),
    ])
    dist = pd.DataFrame([
        dict(key="X1", period_end="2015-06-30", distribution_per_unit=0.04),
        dict(key="X2", period_end="2015-06-30", distribution_per_unit=0.02),
    ])
    res = build_index(hold, dist, config=IndexConfig(base_date="2015-03-31"))
    p = res.positions.set_index("key")
    # 13.2m restated to 10m units = 6.6m; +0.4m cash on 6m -> 16.67%
    assert p.loc["X1", "total_return"] == pytest.approx((6.6e6 + 0.4e6) / 6e6 - 1)
    assert p.loc["X2", "total_return"] == pytest.approx((1.8e6 + 0.1e6) / 2e6 - 1)
    exp = (6e6 * p.loc["X1", "total_return"] + 2e6 * p.loc["X2", "total_return"]) / 8e6
    assert res.index["total_return"].iloc[1] == pytest.approx(exp)
    assert res.index["num_firms"].iloc[1] == 2
    assert res.index["clo_aum_fair_value"].iloc[1] == pytest.approx(15e6)


def test_cost_basis_distribution_estimate():
    # 20% yield on cost 8m for a 90-day quarter = 394,521 income; cost fell 0.3m
    q0, q1 = pd.Timestamp("2016-03-31"), pd.Timestamp("2016-06-30")
    hold = pd.DataFrame([
        H("A", q0, "Gamma CLO", 10e6, 7.5e6, cost=8e6, ey=0.20, cusip="G1"),
        H("A", q1, "Gamma CLO", 10e6, 7.4e6, cost=7.7e6, ey=0.19, cusip="G1"),
    ])
    res = build_index(hold, config=IndexConfig(base_date=str(q0.date())))
    p = res.positions.iloc[0]
    cash = 0.20 * 8e6 * (q1 - q0).days / 365 + 0.3e6
    assert p["dist_method"] == "cost_basis"
    assert p["distribution"] == pytest.approx(cash)
    assert p["total_return"] == pytest.approx((7.4e6 + cash) / 7.5e6 - 1)


def test_eligibility_filters():
    q0, q1 = "2015-03-31", "2015-06-30"
    hold = pd.DataFrame([
        H("LATE", q0, "Late CLO", 1e6, 1e6, cusip="L"),
        H("LATE", q1, "Late CLO", 1e6, 0.5e6, cusip="L", filed=pd.Timestamp(q1) + pd.Timedelta(days=120)),
        H("SMALL", q0, "Small CLO", 1e6, 1e6, cusip="S"),
        H("SMALL", q1, "Small CLO", 1e6, 0.5e6, cusip="S"),
        H("OK", q0, "Good CLO", 1e6, 1e6, cusip="G"),
        H("OK", q1, "Good CLO", 1e6, 1.1e6, cusip="G"),
        H("OK", q1, "Euro CLO 2015 DAC", 1e6, 9e6, cusip="E", currency="EUR"),
    ])
    info = pd.DataFrame([dict(fund="SMALL", period_end=q, total_investments=100e6) for q in (q0, q1)])
    res = build_index(hold, fund_info=info, config=IndexConfig(base_date=q0, allow_missing_distributions=True))
    assert set(res.positions["fund"]) == {"OK"}
    assert res.index["total_return"].iloc[1] == pytest.approx(0.10)
    assert res.index["num_clos"].iloc[1] == 1  # euro deal excluded


def test_missing_distribution_excluded_by_default():
    hold = pd.DataFrame([H("A", "2015-03-31", "Z CLO", 1, 1, cusip="Z"), H("A", "2015-06-30", "Z CLO", 1, 1, cusip="Z")])
    res = build_index(hold, config=IndexConfig(base_date="2015-03-31"))
    assert np.isnan(res.index["total_return"].iloc[1])
    assert res.index["price_return"].iloc[1] == 0
    assert res.index["distribution_coverage"].iloc[1] == 0


def test_called_deal_with_final_payout():
    hold = pd.DataFrame([
        H("A", "2015-03-31", "Called CLO", 1e6, 0.5e6, cusip="C"),
        H("A", "2015-03-31", "Other CLO", 1e6, 0.5e6, cusip="O"),
        H("A", "2015-06-30", "Other CLO", 1e6, 0.5e6, cusip="O"),
    ])
    dist = pd.DataFrame([dict(key="C", period_end="2015-06-30", distribution_per_unit=0.55),
                         dict(key="O", period_end="2015-06-30", distribution_per_unit=0.0)])
    res = build_index(hold, dist, config=IndexConfig(base_date="2015-03-31"))
    p = res.positions.set_index("key")
    assert p.loc["C", "status"] == "exit"
    assert p.loc["C", "total_return"] == pytest.approx(0.10)
    assert res.index["total_return"].iloc[1] == pytest.approx(0.05)


def test_report_layout():
    rng = np.random.default_rng(0)
    qs = pd.date_range("2014-09-30", "2017-12-31", freq="QE")
    rows = []
    fv = 1e6
    for i, q in enumerate(qs):
        rows.append(H("A", q, "R CLO", 1e6, fv, cusip="R"))
        fv *= 1 + rng.normal(0, 0.05)
    res = build_index(pd.DataFrame(rows), config=IndexConfig(allow_missing_distributions=True))
    grid = quarterly_table(res.index)
    assert list(grid.columns) == ["Q1", "Q2", "Q3", "Q4", "Year Total"]
    assert grid.index.tolist() == [2014, 2015, 2016, 2017]
    s = summary_stats(res.index)
    assert s["As of"] == "4Q2017"
    assert s["Maximum Quarterly Drawdown"] == pytest.approx(res.index["total_return"].min())
    assert "<svg" in to_html(res)


def test_classifier():
    assert is_clo_equity("Octagon Investment Partners 18-R, Ltd.", "Subordinated Notes")
    assert is_clo_equity("Madison Park Funding XIV", "Income Notes")
    assert not is_clo_equity("Madison Park Funding XIV", "Class E Notes")
    assert not is_clo_equity("Some CLO Warehouse", "Subordinated Notes")
    assert not is_us_clo("Harvest CLO XX DAC")
    assert normalise_name("Ares XLIX CLO, Ltd.") == normalise_name("ARES XLIX CLO LTD")


NPORT = b"""<?xml version="1.0"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
 <formData>
  <genInfo><repPdDate>2023-03-31</repPdDate></genInfo>
  <fundInfo><totAssets>1000000</totAssets><netAssets>900000</netAssets></fundInfo>
  <invstOrSecs>
   <invstOrSec><name>Ares XLIX CLO Ltd</name><title>Subordinated Notes</title><cusip>04016AAA1</cusip>
     <identifiers><isin value="US04016AAA10"/></identifiers><balance>500000</balance><curCd>USD</curCd>
     <valUSD>300000</valUSD><assetCat>ABS-CBDO</assetCat></invstOrSec>
   <invstOrSec><name>Ares XLIX CLO Ltd</name><title>Class E Notes</title><cusip>04016AAB9</cusip>
     <balance>500000</balance><curCd>USD</curCd><valUSD>480000</valUSD><assetCat>ABS-CBDO</assetCat></invstOrSec>
  </invstOrSecs>
 </formData>
</edgarSubmission>"""


def test_parse_nport():
    h, info = parse_nport(NPORT, "F", "2023-05-30")
    assert len(h) == 1
    r = h.iloc[0]
    assert (r["cusip"], r["isin"], r["par"], r["fair_value"]) == ("04016AAA1", "US04016AAA10", 500000, 300000)
    assert info["total_investments"] == 780000
    assert info["period_end"] == pd.Timestamp("2023-03-31")


SOI = """<html><body><table>
<tr><td colspan="2">Company</td><td>Investment</td><td></td><td colspan="2">Principal Amount</td>
    <td colspan="2">Amortized Cost</td><td colspan="2">Fair Value</td><td>Effective Yield</td></tr>
<tr><td colspan="2">CLO Equity</td></tr>
<tr><td colspan="2">Ares XLIX CLO Ltd.</td><td>Subordinated Notes</td><td></td><td>$</td><td>7,000,000</td>
    <td>$</td><td>2,100,500</td><td>$</td><td>1,686,669</td><td>18.5%</td></tr>
<tr><td colspan="2">Madison Park Funding XIV</td><td>Class E Notes</td><td></td><td>$</td><td>1,000,000</td>
    <td>$</td><td>990,000</td><td>$</td><td>950,000</td><td>9.1%</td></tr>
<tr><td colspan="2">Total CLO Equity</td><td></td><td></td><td></td><td></td><td></td><td>3,090,500</td>
    <td></td><td>2,636,669</td><td></td></tr>
</table></body></html>"""


def test_parse_soi():
    all_rows = parse_schedule_of_investments(SOI, "F", "2014-12-31", equity_only=False)
    assert len(all_rows) == 2
    eq = parse_schedule_of_investments(SOI, "F", "2014-12-31")
    assert len(eq) == 1
    r = eq.iloc[0]
    assert r["issuer"] == "Ares XLIX CLO Ltd."
    assert r["tranche"] == "Subordinated Notes"
    assert (r["par"], r["cost"], r["fair_value"]) == (7_000_000, 2_100_500, 1_686_669)
    assert r["effective_yield"] == pytest.approx(0.185)
