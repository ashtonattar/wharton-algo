"""Tests for the country-ETF foreign sleeve. Run:  python -m pytest -q test_country_sleeve.py"""
import json, os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import country_sleeve as cs
import laura_strategy as ls
from funnel_v2 import CASH, MarketData
from laura_strategy import Plan, decide, default_phases, lifecycle_backtest, v3_phases

HERE = os.path.dirname(os.path.abspath(__file__))
ASOF = "2026-09-30"
STUB = lambda data, asof, cfg, universe: SimpleNamespace(weights={"AAPL": 0.6, "MSFT": 0.4}, status="TARGET MET", mc={"p_hit": 0.9})


@pytest.fixture(scope="module")
def data():
    return MarketData(os.path.join(HERE, "data_us"))


@pytest.fixture
def approved(monkeypatch):
    """Swap the approved ETF list for a test one: approved(rows) with rows (ticker, country)."""
    def use(rows):
        cols = ["ticker", "name", "issuer", "target_country", "market_tracking", "notes", "expense_ratio"]
        df = pd.DataFrame([(t, t, "", c, "Y", "", er) for t, c, er in rows], columns=cols)
        monkeypatch.setattr(cs, "load_approved", lambda path=None: df)
    return use


def foreign_weight(dec):
    return sum(dec["weights"].get(t, 0) for t in dec["info"]["sleeves"]["foreign"]["weights"])


# ---------------------------------------------------------------- 1. screener parser
def test_parser_reads_left_table():
    s = cs.load_country_scores()
    assert len(s) == 138 and list(s.columns[:3]) == ["country", "iso3", "score"]
    assert s.set_index("country").loc["Canada", "score"] == pytest.approx(14.1507)
    assert s.set_index("country").loc["Australia", "gdp_per_capita"] == pytest.approx(75_648)
    assert "Türkiye" in set(s.country) and s.score.notna().all()


def test_top_22_matches_right_hand_summary():
    top = cs.ranked_countries(cs.load_country_scores(), Plan())
    summ = cs.load_screener_summary()
    assert list(top.country) == list(summ.country)
    assert np.allclose(top.score, summ.score)


def test_missing_score_fails_loudly(tmp_path):
    src = cs._read_rows(cs.SCREENER)
    src[1][15] = ""                                         # first country's Score blank
    f = tmp_path / "bad.csv"
    pd.DataFrame(src).to_csv(f, header=False, index=False)
    with pytest.raises(ValueError, match="Score missing"):
        cs.load_country_scores(str(f))


def test_us_is_never_picked(approved):
    approved([("SPY", "United States", 0.09), ("BBCA", "Canada", 0.19), ("EWA", "Australia", 0.5)])
    rep = cs.country_sleeve(os.path.join(HERE, "data_us"), ASOF, Plan(), 0.40)
    assert "United States" not in {r["country"] for r in rep["rows"]} | {r["country"] for r in rep["skipped"]}
    assert "SPY" not in rep["weights"]


# ---------------------------------------------------------------- 2. weights
def test_weights_sum_to_one_capped_and_in_score_order():
    sc = {"A": 14.0, "B": 13.0, "C": 11.0, "D": 10.0, "E": 9.0}
    w, left = cs.capped_score_weights(sc, 0.40)
    assert sum(w.values()) == pytest.approx(1.0) and left == 0
    assert list(w) == sorted(sc, key=sc.get, reverse=True)
    assert all(w[a] >= w[b] for a, b in zip("ABCD", "BCDE"))
    assert w["A"] == pytest.approx(14 / 57)
    big, left = cs.capped_score_weights({"A": 60.0, "B": 10.0, "C": 8.0, "D": 2.0}, 0.40)
    assert big["A"] == pytest.approx(0.40) and max(big.values()) <= 0.40 + 1e-12 and sum(big.values()) == pytest.approx(1.0)
    assert big["B"] / big["C"] == pytest.approx(10 / 8)                # excess spread in proportion to score
    two, left = cs.capped_score_weights({"A": 14.0, "B": 13.0}, 0.40)  # two countries can't fill 100% at a 40% cap
    assert two == {"A": 0.40, "B": 0.40} and left == pytest.approx(0.20)


def test_live_sleeve_weights(data):
    dec = decide(data, ASOF, Plan(), "growth", 1, selector=STUB)
    f = dec["info"]["sleeves"]["foreign"]
    rep = f["country"]
    assert sum(f["weights"].values()) == pytest.approx(1.0)
    assert all(r["weight"] <= 0.40 + 1e-9 for r in rep["rows"])
    assert [r["country"] for r in rep["rows"]] == ["Canada", "Australia"]          # the only top-22 countries with an approved ETF
    assert f["weights"] == pytest.approx({"BBCA": 0.40, "EWA": 0.40, "VEA": 0.20})


# ---------------------------------------------------------------- 3. 70/30 equity split everywhere
def test_equity_split_70_30_in_every_phase_and_bond_level(data, monkeypatch):
    seen = []
    orig = ls._decide_mix
    def spy(data, asof, plan, phase, ph, *a):
        seen.append((phase, ph.mix))
        return orig(data, asof, plan, phase, ph, *a)
    monkeypatch.setattr(ls, "_decide_mix", spy)
    plan = Plan()
    for ph_name, yr in (("growth", 1), ("pre_gift_y5", 5), ("pre_gift", 6), ("reserve", 5), ("post_gift", 7)):
        decide(data, "2013-12-31", plan, ph_name, yr, selector=STUB)
    assert {p for p, _ in seen} == set(plan.phases)
    levels = {(p, round(m["bonds"], 6)) for p, m in seen}
    assert {b for p, b in levels if p == "growth"} == set(plan.phases["growth"].bond_choices)       # every ladder rung tried
    assert {b for p, b in levels if p == "post_gift"} == set(plan.phases["post_gift"].bond_choices)
    for p, m in seen:
        eq = m["domestic"] + m["foreign"]
        assert m["foreign"] == pytest.approx(0.30 * eq) and m["domestic"] == pytest.approx(0.70 * eq)
        assert sum(m.values()) == pytest.approx(1.0)
        assert m["cash"] == plan.phases[p].mix["cash"]                           # cash share untouched


def test_equity_split_in_holdings(data):
    dec = decide(data, "2013-12-31", Plan(), "growth", 1, selector=STUB)
    dom = dec["weights"]["AAPL"] + dec["weights"]["MSFT"]
    assert foreign_weight(dec) / (dom + foreign_weight(dec)) == pytest.approx(0.30)


# ---------------------------------------------------------------- 4. no look-ahead
def test_etf_pick_ignores_prices_after_asof(approved, monkeypatch):
    approved([("BBCA", "Canada", 0.19), ("EWW", "Canada", 0.50)])               # two candidates for one country
    px = cs.etf_prices(os.path.join(HERE, "data_us")).copy()
    asof = pd.Timestamp("2024-06-30")
    first, _ = cs.pick_etf("Canada", asof, cs.load_approved(), px)
    other = "EWW" if first["ticker"] == "BBCA" else "BBCA"
    scrambled = px.copy()
    after = scrambled.index > asof
    scrambled.loc[after, first["ticker"]] *= 0.01                               # the winner collapses later ...
    scrambled.loc[after, other] *= np.linspace(1, 50, after.sum())              # ... and the loser soars
    again, _ = cs.pick_etf("Canada", asof, cs.load_approved(), scrambled)
    assert again["ticker"] == first["ticker"] and again["ret_12m"] == pytest.approx(first["ret_12m"])
    monkeypatch.setattr(cs, "etf_prices", lambda folder: scrambled)
    a = cs.country_sleeve("x", asof, Plan(), 0.40)
    monkeypatch.setattr(cs, "etf_prices", lambda folder: px)
    b = cs.country_sleeve("x", asof, Plan(), 0.40)
    assert a["weights"] == b["weights"]


def test_etf_needs_12_months_of_history(approved):
    approved([("BBCA", "Canada", 0.19), ("EWA", "Australia", 0.5)])
    rep = cs.country_sleeve(os.path.join(HERE, "data_us"), "2019-03-31", Plan(), 0.40)   # BBCA listed Aug 2018
    assert [r["country"] for r in rep["rows"]] == ["Australia"]
    assert any(s["country"] == "Canada" and "months of prices" in s["reason"] for s in rep["skipped"])


def test_lookahead_warning():
    assert "look-ahead" in cs.lookahead_warning(Plan(), "2006-12-31")
    assert cs.lookahead_warning(Plan(), "2026-09-30") is None
    assert cs.lookahead_warning(Plan(foreign_mode="index"), "2006-12-31") is None


# ---------------------------------------------------------------- 5. fallbacks
def test_three_countries_are_reweighted(approved):
    approved([("BBCA", "Canada", 0.19), ("EWA", "Australia", 0.5), ("EWW", "Sweden", 0.5)])
    rep = cs.country_sleeve(os.path.join(HERE, "data_us"), ASOF, Plan(), 0.40)
    assert [r["country"] for r in rep["rows"]] == ["Canada", "Australia", "Sweden"]
    tot = 14.1507 + 13.8388 + 11.4812
    assert rep["weights"] == pytest.approx({"BBCA": 14.1507 / tot, "EWA": 13.8388 / tot, "EWW": 11.4812 / tot}, abs=1e-4)
    assert rep["filler"] == 0 and sum(rep["weights"].values()) == pytest.approx(1.0)


def test_no_country_qualifies_falls_back_to_index(data, approved):
    approved([])
    g = decide(data, ASOF, Plan(), "growth", 1, selector=STUB)
    assert g["info"]["sleeves"]["foreign"]["weights"] == {"VEA": 1.0}
    old = decide(data, ASOF, Plan(fallback_holdings={"domestic": "SPY", "foreign": "EFA"}), "growth", 1, selector=STUB)
    assert old["info"]["sleeves"]["foreign"]["weights"] == {"EFA": 1.0}
    r = decide(data, ASOF, Plan(), "reserve", 5, selector=STUB)                 # security phase: T-bills
    assert r["info"]["sleeves"]["foreign"]["weights"] == {CASH: 1.0}


def test_m1_downtrend_sends_foreign_sleeve_to_tbills(data):
    assert ls.market_up(data, "2008-10-31", Plan())["foreign"] is False
    dec = decide(data, "2008-10-31", Plan(), "growth", 1, selector=STUB)
    assert dec["info"]["sleeves"]["foreign"]["weights"] == {CASH: 1.0}
    assert not {"BBCA", "EWA", "VEA"} & set(dec["weights"])


# ---------------------------------------------------------------- 6. old modes unchanged
LEGACY = dict(foreign_share_of_equity=None, fallback_holdings={"domestic": "SPY", "foreign": "EFA"})


def test_old_modes_reproduce_previous_results(data):
    """legacy_baseline.json was recorded with the code before the country-ETF sleeve (same data_us folder)."""
    base = json.load(open(os.path.join(HERE, "legacy_baseline.json")))
    sel = {"stub": STUB, "mid": lambda d, a, c, u: SimpleNamespace(weights={"AAPL": 1.0}, status="TARGET MET", mc={"p_hit": 0.55})}
    for key, want in base.items():
        if key.endswith("lifecycle"):
            continue
        mode, ver, phase, asof, sn = key.split("|")
        kw = dict(foreign_index_only=True) if mode == "index" else dict(foreign_mode="funnel")
        p = Plan(phases=v3_phases() if ver == "v3" else default_phases(), **LEGACY, **kw)
        yr = {"growth": 1, "post_gift": 7, "reserve": 5, "pre_gift_y5": 5, "pre_gift": 6}[phase]
        got = decide(data, asof, p, phase, yr, selector=sel[sn])["weights"]
        assert got == pytest.approx(want, abs=1e-10), key
    cash = lambda d, a, c, u: SimpleNamespace(weights={CASH: 1.0}, status="STUB", mc={})
    for mode in ("index", "funnel"):
        kw = dict(foreign_index_only=True) if mode == "index" else dict(foreign_mode="funnel")
        sm = lifecycle_backtest(data, "2007-12-31", Plan(phases=default_phases(), **LEGACY, **kw), selector=cash, verbose=False)["summary"]
        want = base[f"{mode}|lifecycle"]
        assert sm["value_2031"] == pytest.approx(want["value_2031"], abs=1e-3)
        assert sm["gift"] == pytest.approx(want["gift"], abs=1e-3)
        assert sm["reserve_left_2043"] == pytest.approx(want["reserve_left"], abs=1e-3)
        assert sm["funding_left_2043"] == pytest.approx(want["funding_left"], abs=1e-3)


def test_foreign_index_only_alias():
    assert Plan(foreign_index_only=True).foreign_mode == "index"
    with pytest.raises(ValueError):
        Plan(foreign_mode="stocks")
