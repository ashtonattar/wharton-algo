"""Tests for the sector tilt on the domestic stock funnel. Run:  python -m pytest -q test_sector_sleeve.py"""
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sector_sleeve as ss
from funnel_v2 import Config, MarketData, select
from laura_strategy import Plan, decide

HERE = os.path.dirname(os.path.abspath(__file__))
ASOF = "2026-09-30"
FAST = dict(screen_paths=400, mc_paths=800, resamples=4, stock_bootstrap_paths=300, grid_points=6, top_n=25)
EXPECTED = {   # sector: (score, risk, tier), in rank order
    "Technology": (6.5, 4.8, "overweight"), "Basic Materials": (4.2, 7.6, "neutral"), "Energy": (4.1, 7.6, "neutral"),
    "Healthcare": (4.0, 1.8, "overweight"), "Industrials": (4.0, 8.2, "neutral"), "Utilities": (3.8, 5.4, "overweight"),
    "Consumer Cyclical": (3.5, 8.0, "neutral"), "Communication Services": (3.2, 7.2, "neutral"),
    "Financial Services": (3.2, 9.0, "underweight"), "Real Estate": (2.5, 6.4, "underweight"),
    "Consumer Defensive": (0.7, 2.4, "underweight")}


def recorder(status="TARGET MET"):
    """A selector that records the Config each sleeve got."""
    seen = []
    def sel(data, asof, cfg, universe):
        seen.append(cfg)
        st = status(cfg) if callable(status) else status
        return SimpleNamespace(weights={universe[0]: 1.0}, status=st, mc={"p_hit": 0.9})
    sel.seen = seen
    return sel


@pytest.fixture(scope="module")
def data():
    return MarketData(os.path.join(HERE, "data_us"))


@pytest.fixture(scope="module")
def inputs():
    return ss.load_sector_inputs()


@pytest.fixture(scope="module")
def universe(data):
    """60 S&P members that day, spread over sectors (keeps select() fast)."""
    from build_us_universe import universe_fn
    members = [t for t in universe_fn(os.path.join(HERE, "data_us"))(ASOF) if t in data.price_usd.columns]
    by = data.meta.loc[members, "sector"]
    return sorted(by.groupby(by).head(6).index)


# ---------------------------------------------------------------- 1-2. scores
def test_scores_match_the_workbook(inputs):
    df = ss.score_sectors(inputs)
    assert list(df.index) == list(EXPECTED)
    for s, (score, risk, tier) in EXPECTED.items():
        assert round(df.score[s], 1) == score, s
        assert round(df.risk[s], 1) == risk, s
        assert df.tier[s] == tier, s


def test_eps_growth_compares_to_the_sp_not_a_blank_cell(inputs):
    df = ss.score_sectors(inputs)
    assert inputs.loc["Basic Materials", "eps_growth_cy2027"] < 0.16
    assert not df.s5_pass["Basic Materials"]
    assert round(df.score["Basic Materials"], 1) == 4.2           # 5.2 with the workbook's unlocked G44
    assert set(df.index[df.s5_pass]) == {"Technology", "Healthcare", "Industrials"}


def test_missing_value_fails_loudly(tmp_path):
    df = pd.read_csv(ss.INPUTS)
    df.loc[df.yahoo_sector == "Energy", "vol_10y"] = None
    p = tmp_path / "x.csv"; df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="vol_10y"):
        ss.load_sector_inputs(str(p))
    ok = pd.read_csv(ss.INPUTS)
    assert ok.ev_ebitda_ttm.isna().any()                         # Financials' blank EV/EBITDA is fine
    ss.load_sector_inputs()


# ---------------------------------------------------------------- 3. strict inequalities
def test_value_on_a_threshold_fails(inputs):
    x = inputs.copy()
    x.loc["Utilities", "corr_spx_10y"] = 0.60
    x.loc["Healthcare", "vol_10y"] = 0.17
    df = ss.score_sectors(x)
    assert not df.s1_pass["Utilities"]
    assert not df.r4_pass["Healthcare"]
    assert df.r4_pts["Healthcare"] == 1.2


# ---------------------------------------------------------------- 4. tiers
def test_veto_moves_overweight_to_the_next_in_line(inputs):
    x = inputs.copy()
    x.loc["Healthcare", "max_drawdown_10y"] = -0.50                # +1.8 risk
    x.loc["Healthcare", "vol_10y"] = 0.20                          # +1.2
    x.loc["Healthcare", "avg_corr_other_sectors"] = 0.70           # +1.4
    x.loc["Healthcare", "months_peak_to_recovery"] = 120           # +1.0 -> 7.2 (r7 already fails)
    df = ss.score_sectors(x)
    assert df.risk["Healthcare"] >= 7.0
    assert set(df.index[df.tier == "overweight"]) == {"Technology", "Utilities", "Real Estate"}
    assert df.tier["Healthcare"] == "neutral"


def test_ties_broken_by_lower_risk(inputs):
    df = ss.score_sectors(inputs)
    order = list(df.index)
    assert df.score["Healthcare"] == pytest.approx(df.score["Industrials"])
    assert order.index("Healthcare") < order.index("Industrials")
    assert order.index("Communication Services") < order.index("Financial Services")


def test_overweight_never_underweight(inputs):
    df = ss.score_sectors(inputs, SimpleNamespace(n_overweight_sectors=3, n_underweight_sectors=9, sector_risk_veto=7.0))
    assert (df.tier == "overweight").sum() == 3 and (df.tier == "underweight").sum() == 8


# ---------------------------------------------------------------- 5. caps in select()
def test_tilted_caps_hold_and_flat_caps_unchanged(data, universe):
    st = ss.sector_tilt(Plan())
    base = Config.for_backtest(**FAST)
    tilted = replace(base, sector_caps=st["caps"], sector_cap=st["neutral_cap"])
    res = select(data, ASOF, tilted, universe)
    assert res.weights
    sec_w = pd.Series(res.weights).groupby(data.meta["sector"].reindex(list(res.weights)).values).sum()
    for s, x in sec_w.items():
        assert x <= st["caps"].get(s, st["neutral_cap"]) + 1e-6, s
    c6 = [g for g in res.gates if g["gate"] == "C6 Concentration"][0]
    assert not any("max sector" in n and "names" not in n for n in res.notes)
    if c6["info"]["names"] >= tilted.min_names:                   # C6 can also fail on the name count alone
        assert c6["n_out"] == c6["n_in"]
    # every sector at the flat cap = no tilt: identical weights
    flat = select(data, ASOF, base, universe)
    same = select(data, ASOF, replace(base, sector_caps={s: base.sector_cap for s in ss.SECTORS}), universe)
    assert same.weights == flat.weights


# ---------------------------------------------------------------- 6. B1 pillar
def test_sector_pillar(data, universe):
    base = Config.for_backtest(**FAST)
    off = select(data, ASOF, base, universe).table.score.dropna()
    zero = select(data, ASOF, replace(base, sector_scores={s: 1.0 for s in ss.SECTORS}), universe).table.score.dropna()
    pd.testing.assert_series_equal(off, zero)                      # pillar not weighted -> B1 unchanged
    pw = {k: v * 0.9 for k, v in base.pillar_weights.items()} | {"sector": 0.1}
    assert sum(pw.values()) == pytest.approx(1.0)
    scores = ss.sector_tilt(Plan())["scores"]
    on = select(data, ASOF, replace(base, sector_scores=scores, pillar_weights=pw), universe).table.score.dropna()
    assert not np.allclose(on.values, off.reindex(on.index).values)


def test_unknown_sector_gets_the_median_not_the_lowest_rank():
    from funnel_v2 import pct_rank
    scores = ss.sector_tilt(Plan())["scores"]
    mid = float(np.median(list(scores.values())))
    assert mid == pytest.approx(3.8)
    ranks = pct_rank([scores["Technology"], scores["Consumer Defensive"], mid])
    assert ranks[2] == 0.5 and ranks[1] == 0.0


# ---------------------------------------------------------------- 7. fallback to flat caps
def test_infeasible_tilt_falls_back_to_flat_caps(data, capsys):
    sel = recorder(lambda c: "NOT ENOUGH STOCKS TO SATISFY CAPS" if c.sector_caps else "TARGET MET")
    dec = decide(data, ASOF, Plan(), "growth", 1, selector=sel)
    dom = dec["info"]["sleeves"]["domestic"]
    assert dom["status"] == "TARGET MET"
    assert dom["sector"]["relaxed"] == "sector tilt relaxed: NOT ENOUGH STOCKS TO SATISFY CAPS"
    assert "sector tilt relaxed" in capsys.readouterr().out
    assert [c.sector_caps is None for c in sel.seen[:2]] == [False, True]


def test_select_reports_infeasible_tilted_caps(data, universe):
    fin = [t for t in universe if data.meta.loc[t, "sector"] in ("Financial Services", "Real Estate")]
    st = ss.sector_tilt(Plan())
    res = select(data, ASOF, Config.for_backtest(**FAST, sector_caps=st["caps"], sector_cap=st["neutral_cap"],
                                                 min_psr=-10, min_sortino=-10, min_p_beat_cash=0), fin)
    assert res.status.startswith(("NOT ENOUGH", "NO STOCKS"))


# ---------------------------------------------------------------- 8. look-ahead guard
def test_tilt_skipped_before_the_snapshot(data):
    for asof in ("2015-12-31", "2026-08-31"):
        sel = recorder()
        dec = decide(data, asof, Plan(), "growth", 1, selector=sel)
        assert all(c.sector_caps is None and "sector" not in c.pillar_weights for c in sel.seen)
        assert dec["info"]["sector_tilt_msg"].startswith("sector tilt skipped")
    sel = recorder()
    dec = decide(data, "2015-12-31", Plan(sector_tilt_in_backtests=True), "growth", 1, selector=sel)
    assert sel.seen[0].sector_caps is not None and dec["info"]["sector_tilt_msg"].startswith("WARNING look-ahead")
    sel = recorder()
    decide(data, ASOF, Plan(), "growth", 1, backtest_mode=True, selector=sel)      # run_today's live call
    assert sel.seen[0].sector_caps is not None


def test_off_mode_is_the_old_config(data):
    a, b = recorder(), recorder()
    decide(data, ASOF, Plan(sector_mode="off"), "growth", 1, selector=a)
    decide(data, "2015-12-31", Plan(), "growth", 1, selector=b)
    assert a.seen[0].sector_caps is None and a.seen[0].sector_scores is None
    assert set(a.seen[0].pillar_weights) == set(Config().pillar_weights)
    with pytest.raises(ValueError):
        Plan(sector_mode="sideways")


def test_tilted_pillar_weights_scale_the_phase_weights(data):
    sel = recorder()
    decide(data, ASOF, Plan(), "growth", 1, selector=sel)
    off = recorder()
    decide(data, ASOF, Plan(sector_mode="off"), "growth", 1, selector=off)
    pw, pw0 = sel.seen[0].pillar_weights, off.seen[0].pillar_weights
    assert sum(pw.values()) == pytest.approx(1.0) and pw["sector"] == 0.10
    assert all(pw[k] == pytest.approx(0.9 * v) for k, v in pw0.items())


# ---------------------------------------------------------------- 9. foreign sleeve untouched
def test_foreign_sleeve_identical_with_and_without_tilt(data):
    import copy
    data = copy.copy(data)                                          # data_us has no foreign stocks: relabel two
    data.meta = data.meta.copy()
    data.meta.loc[["AAPL", "XOM"], "country"] = "Canada"
    for mode in ("country_etf", "funnel", "index"):
        got = {}
        for sm in ("tilt", "off"):
            sel = recorder()
            dec = decide(data, ASOF, Plan(foreign_mode=mode, sector_mode=sm), "growth", 1, selector=sel)
            got[sm] = dec["info"]["sleeves"]["foreign"]["weights"]
            if mode == "funnel":
                foreign_cfgs = [c for c in sel.seen if c.country_cap < 1.0]
                assert foreign_cfgs and all(c.sector_caps is None and c.sector_scores is None for c in foreign_cfgs)
        assert got["tilt"] == got["off"], mode


# ---------------------------------------------------------------- account -> sleeve cap conversion
def test_sector_caps_are_a_share_of_the_accounts_equities(data):
    """A sleeve weight w is w x share / eq of the account's equities, so the sleeve cap is cap x eq / share.
    With the 70/30 split the domestic sleeve is 70% of equities at every bond level: 35% -> 50% of the sleeve."""
    for sm, want in (("off", {"Technology": 0.5, "Unknown": 0.5}),
                     ("tilt", {"Technology": 0.5, "Basic Materials": 0.25 / 0.7, "Real Estate": 0.10 / 0.7, "Unknown": 0.25 / 0.7})):
        sel = recorder()
        decide(data, ASOF, Plan(sector_mode=sm), "growth", 1, selector=sel)
        assert len(sel.seen) >= 2                                   # several bond levels tried
        for c in sel.seen:
            got = {s: (c.sector_caps or {}).get(s, c.sector_cap) for s in want}
            assert got == pytest.approx(want), sm
    sel = recorder()
    data2 = __import__("copy").copy(data)
    data2.meta = data.meta.copy()
    data2.meta.loc[["AAPL", "XOM"], "country"] = "Canada"
    decide(data2, ASOF, Plan(foreign_mode="funnel", sector_mode="off"), "growth", 1, selector=sel)
    assert all(c.sector_cap == 1.0 for c in sel.seen if c.country_cap < 1.0)   # 35% / 0.3 > 1: no cap inside foreign


def test_skip_line_printed_once_per_run(data, capsys):
    ss._ANNOUNCED.clear()
    for asof in ("2015-12-31", "2016-03-31", "2016-06-30"):
        decide(data, asof, Plan(), "growth", 1, selector=recorder())
    assert capsys.readouterr().out.count("sector tilt skipped") == 1
