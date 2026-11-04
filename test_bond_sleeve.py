"""Tests for the scored bond sleeve. Run:  python -m pytest -q test_bond_sleeve.py"""
import json, os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bond_sleeve as bs
import laura_strategy as ls
from funnel_v2 import CASH, MarketData
from laura_strategy import Plan, decide, lifecycle_backtest

HERE = os.path.dirname(os.path.abspath(__file__))
SETTLE = "2026-10-23"
STUB = lambda data, asof, cfg, universe: SimpleNamespace(weights={"AAPL": 0.6, "MSFT": 0.4}, status="TARGET MET", mc={"p_hit": 0.9})
PHASE_YEAR = {"growth": 1, "pre_gift_y5": 5, "pre_gift": 6, "post_gift": 7, "reserve": 5}


@pytest.fixture(scope="module")
def data():
    return MarketData(os.path.join(HERE, "data_us"))


@pytest.fixture(scope="module")
def df():
    return bs.score_bonds()


def row(df, name):
    return df[df.bond == name].iloc[0]


# ---------------------------------------------------------------- 1. anchors from the workbook
def test_workbook_anchors(df):
    r = row(df, "U.S. 4.500% Feb 2036")
    assert r.mod_dur == pytest.approx(7.43, abs=0.02)
    assert r.carry_1y == pytest.approx(0.0493, abs=0.0002)
    assert r.ret_1y == pytest.approx(0.0516, abs=0.0002)
    assert r.excess_per_dur_bp == pytest.approx(9.7, abs=0.1)
    old = row(df, "U.S. 3.125% May 2048")
    assert old.excess_per_dur_bp == pytest.approx(5.0, abs=0.1)
    usd = df[(df.currency == "USD") & df.excess_per_dur_bp.notna()]
    assert old.key in set(usd.nsmallest(3, "excess_per_dur_bp").key)


# ---------------------------------------------------------------- 2. sanity of every row
def test_durations_and_no_nan(df):
    assert (df.mod_dur < df.mac_dur).all()
    always = ["default_risk", "years_left", "clean", "accrued", "dirty", "mac_dur", "mod_dur", "dv01_100k",
              "carry_1y", "ret_1y", "ret_per_dur_bp"]
    vals = df[always].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    alive = df.maturity > bs.edate(SETTLE, 12)
    assert np.isfinite(df.loc[alive, ["price_1y", "coupons_1y"]].to_numpy(dtype=float)).all()
    assert df.loc[~alive, "carry_1y"].equals(df.loc[~alive, "ytm"])          # matures within the year: carry = YTM
    usd = df.currency == "USD"
    assert np.isfinite(df.loc[usd & (df.years_left >= 1), "excess_per_dur_bp"]).all()


# ---------------------------------------------------------------- 3. score formulas on a hand-built group
def test_scores_by_hand(tmp_path):
    # three bonds in one bucket (same terms); C beats B beats A on return, return per duration and default risk,
    # roll-down order B > C > A. One lone bond in another bucket.
    rows = [dict(stocktrak_no=n, bond=b, currency="USD", maturity=m, coupon=0.04, freq=2, ytm=y, cds_bp=cds,
                 convexity=1, rolldown=roll, breakeven_bp=None)
            for n, b, m, y, cds, roll in [(1, "A", "2040-02-15", 0.050, 30, 0.0010), (2, "B", "2040-02-15", 0.051, 20, 0.0015),
                                           (3, "C", "2040-02-15", 0.052, 10, 0.0012), (4, "LONE", "2028-02-15", 0.045, 10, 0.0)]]
    pd.DataFrame(rows).to_csv(tmp_path / "bonds.csv", index=False)
    s = bs.BondSettings()
    d = bs.score_bonds(s, str(tmp_path / "bonds.csv")).set_index("bond")
    # A: last on everything -> 0. B: 0.4*.5 + 0.3*.5 + 0.15*1 + 0.15*.5 = 57.5. C: 0.4 + 0.3 + 0.15*.5 + 0.15 = 92.5
    assert d.loc[["A", "B", "C"], "within_score"].tolist() == pytest.approx([0.0, 57.5, 92.5])
    assert d.loc[["A", "B", "C"], "within_rank"].tolist() == [3, 2, 1]
    assert d.loc["LONE", "within_score"] == 50 and d.loc["LONE", "within_rank"] == 1 and d.loc["LONE", "group_within"] == 1
    # across maturities: percentile of excess return per duration among all four (same currency)
    aa = d.excess_per_dur_bp
    want = {b: 100 * (aa < aa[b]).sum() / 3 for b in d.index}
    assert d.across_score.to_dict() == pytest.approx(want)
    assert sorted(d.across_rank) == [1, 2, 3, 4]
    # weights live in BondSettings: change one, the score changes
    d2 = bs.score_bonds(bs.BondSettings(w_ret_per_dur=0.0, w_ret=0.7), str(tmp_path / "bonds.csv")).set_index("bond")
    assert d2.loc["B", "within_score"] == pytest.approx(100 * (0.7 * 0.5 + 0.15 * 1 + 0.15 * 0.5))


def test_lone_across_group_scores_50(tmp_path):
    r = dict(stocktrak_no=1, bond="X", currency="USD", maturity="2035-02-15", coupon=0.04, freq=2, ytm=0.05, cds_bp=30,
             convexity=1, rolldown=0.001, breakeven_bp=None)
    pd.DataFrame([r]).to_csv(tmp_path / "b.csv", index=False)
    d = bs.score_bonds(bs.BondSettings(), str(tmp_path / "b.csv"))
    assert d.across_score.iloc[0] == 50 and d.within_score.iloc[0] == 50


# ---------------------------------------------------------------- 4. currencies
def test_foreign_bonds_scored_within_but_not_across(df):
    f = df[df.currency.isin(["GBP", "EUR"])]
    assert f.within_score.notna().all() and f.across_score.isna().all()


@pytest.mark.parametrize("phase", list(PHASE_YEAR))
def test_non_usd_never_selected(phase):
    rep = bs.bond_sleeve(SETTLE, Plan(bond_mode="scored"), phase, PHASE_YEAR[phase])
    usd = set(bs.score_bonds().query("currency == 'USD'").key)
    assert set(rep["weights"]) <= usd | {CASH, "IEF"}


# ---------------------------------------------------------------- 5. sleeve rules
@pytest.mark.parametrize("phase", ["growth", "pre_gift_y5", "pre_gift", "post_gift"])
@pytest.mark.parametrize("asof", [SETTLE, "2029-06-30"])
def test_sleeve_weights_cap_duration_and_maturity(phase, asof):
    plan = Plan(bond_mode="scored")
    rep = bs.bond_sleeve(asof, plan, phase, PHASE_YEAR[phase])
    assert sum(rep["weights"].values()) == pytest.approx(1.0)
    assert rep["limit"] == max(1.0, min(plan.phase_end(phase) - PHASE_YEAR[phase], plan.bond_max_duration))
    assert rep["wavg_dur"] <= rep["limit"] + 1e-9
    for r in rep["rows"]:
        assert r["mod_dur"] <= rep["limit"] and r["maturity"] > pd.Timestamp(asof)
        assert (pd.Timestamp(r["maturity"]) - pd.Timestamp(asof)).days / 365.25 >= 0.5
    if len(rep["rows"]) * plan.bond_cap >= 1:
        assert max(rep["weights"].values()) <= plan.bond_cap + 1e-9
    else:
        assert rep["log"] or not rep["rows"]                                   # cap relaxed -> logged


def test_growth_picks_today():
    rep = bs.bond_sleeve(SETTLE, Plan(bond_mode="scored"), "growth", 1)
    assert [r["bond"] for r in rep["rows"]] == ["U.S. 5.250% Feb 2029", "U.S. 5.375% Feb 2031"]
    assert {s["bucket"] for s in rep["skipped"]} == {"7-10", "Over 10"}


def test_nothing_eligible_falls_back():
    p = Plan(bond_mode="scored")
    pre = bs.bond_sleeve(SETTLE, p, "pre_gift", 6)                   # 1-year limit: every bucket's best is longer
    assert pre["weights"] == {CASH: 1.0} and "T-bills" in pre["status"]
    late = bs.bond_sleeve("2049-01-31", p, "growth", 4)              # every bond has matured
    assert late["weights"] == {"IEF": 1.0}                            # return-seeking phase: no T-bills -> IEF


# ---------------------------------------------------------------- 6. reserve ladder
def test_reserve_ladder(df):
    plan = Plan(bond_mode="scored")
    lad = bs.reserve_ladder(SETTLE, plan)
    usd_years = set(df[df.currency == "USD"].maturity.dt.year)
    want_gap = [y for y in range(2033, 2043) if y - 1 not in usd_years]
    assert lad["gap_years"] == want_gap == [2033, 2034, 2035, 2036]
    assert lad["gross_gap"] == 200_000 and 0 < lad["net_gap"] <= 200_000
    for r in lad["rows"]:
        assert r["coverage"] >= 1.0 - 1e-9, r
        if not r["gap"]:
            assert pd.Timestamp(r["maturity"]).year == r["year"] - 1 and r["face"] % 1000 == 0
    keys = set(df[df.currency == "USD"].key)
    assert set(lad["weights"]) <= keys | {CASH}
    assert lad["gap_bond"]["bond"] == "U.S. 5.375% Feb 2031"
    assert sum(lad["weights"].values()) == pytest.approx(1.0)


def test_ladder_without_a_gap_bond_holds_tbills():
    lad = bs.reserve_ladder("2031-06-30", Plan(bond_mode="scored"))      # Feb 2031 bond already matured
    assert lad["gap_bond"] is None and lad["cash_needed"] > 0 and CASH in lad["weights"]
    assert all(r["coverage"] >= 1 - 1e-9 for r in lad["rows"])


# ---------------------------------------------------------------- 7. backward compatibility
def test_ief_mode_is_default_and_unchanged(data):
    assert Plan().bond_mode == "ief"
    for ph, yr in PHASE_YEAR.items():
        a = decide(data, "2013-12-31", Plan(), ph, yr, selector=STUB)["weights"]
        b = decide(data, "2013-12-31", Plan(bond_mode="ief"), ph, yr, selector=STUB)["weights"]
        assert a == b and list(a) == list(b)
        assert not any(k.startswith("UST_") for k in a)


def test_scored_mode_cannot_be_backtested(data):
    with pytest.raises(ValueError, match="live-only|no price history"):
        lifecycle_backtest(data, "2007-12-31", Plan(bond_mode="scored"), verbose=False)
    with pytest.raises(ValueError):
        Plan(bond_mode="bonds")


# ---------------------------------------------------------------- 8. scored mode keeps the bond share and 70/30 split
def test_scored_bond_share_equals_mix_at_every_level(data, monkeypatch):
    seen = []
    orig = ls._decide_mix
    def spy(data, asof, plan, phase, ph, *a):
        d = orig(data, asof, plan, phase, ph, *a)
        seen.append((phase, dict(ph.mix), d))
        return d
    monkeypatch.setattr(ls, "_decide_mix", spy)
    plan = Plan(bond_mode="scored")
    for ph, yr in PHASE_YEAR.items():
        decide(data, "2026-09-30", plan, ph, yr, selector=STUB)
    levels = {(p, round(m["bonds"], 6)) for p, m, _ in seen}
    assert {b for p, b in levels if p == "growth"} == set(plan.phases["growth"].bond_choices)
    for p, m, d in seen:
        w = d["weights"]
        assert sum(w.values()) == pytest.approx(1.0)
        rep = d["info"].get("bonds")
        if m["bonds"] == 0:
            assert rep is None and not any(k.startswith("UST_") for k in w)
            continue
        assert rep["share"] == pytest.approx(m["bonds"])
        assert sum(m["bonds"] * x for x in rep["weights"].values()) == pytest.approx(m["bonds"])
        ust = sum(x for k, x in w.items() if k.startswith("UST_"))
        assert ust == pytest.approx(m["bonds"] * sum(x for k, x in rep["weights"].items() if k.startswith("UST_")))
        assert "IEF" not in w or rep["weights"].get("IEF")
        eq = m["domestic"] + m["foreign"]
        assert m["foreign"] == pytest.approx(0.30 * eq)


# ---------------------------------------------------------------- 9. stale data
def test_stale_warning():
    s = bs.BondSettings.load()
    assert bs.stale_warning("2026-10-31", s) is None
    assert "stale" in bs.stale_warning("2026-11-30", s)
    assert "stale" in bs.bond_sleeve("2026-11-30", Plan(bond_mode="scored"), "growth", 1)["warning"]


def test_build_reads_inputs_only(tmp_path):
    import build_approved_bonds as b
    rows, s = b.read_workbook()
    assert len(rows) == 60 and s["settle"] == SETTLE and s["rf_usd"] == pytest.approx(0.0444)
    csv = pd.read_csv(bs.BONDS)
    assert list(rows.bond) == list(csv.bond) and np.allclose(rows.ytm, csv.ytm) and list(rows.maturity) == list(csv.maturity)
    assert json.load(open(os.path.join(HERE, "bond_settings.json")))["recovery"] == 0.4
