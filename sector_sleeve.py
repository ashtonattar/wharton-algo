"""
sector_sleeve.py - the sector screener as a tilt on the domestic (U.S.) stock funnel.

    inputs  : ../(C) Sector Screener Inputs.csv (Book 9.xlsx "Sector Screener" sheet), one row per sector + the
              S&P 500 benchmark row; joined to stocks on `yahoo_sector` (MarketData.meta["sector"])
    score   : 0-10, higher is better (workbook C93:C103): points for each test passed
    risk    : 0-10, higher is riskier (workbook K3:K13): a penalty for each safe condition NOT met
    tiers   : sectors by score (ties: lower risk). Overweight = first n with risk < veto; Underweight = last n
              (never an Overweight one); the rest Neutral
    tilt    : the domestic sleeve gets per-sector caps by tier (C2-C6) and a "sector" pillar in B1 (funnel_v2.py)

The workbook compares every sector's EPS growth to G44 with a relative reference (G33>G44 filled down), so rows
after IT compare to a blank cell. Here every sector is compared to the S&P's growth (the fixed $G$44).

LOOK-AHEAD: the inputs are one October 2026 snapshot (forward P/E, EPS revisions since Sep 30, analyst numbers).
The tilt is skipped before SNAPSHOT_DATE unless Plan.sector_tilt_in_backtests is set (see tilt_status).
"""
from __future__ import annotations

import os
import statistics

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "..", "(C) Sector Screener Inputs.csv")
SNAPSHOT_DATE = pd.Timestamp("2026-09-30")
BENCHMARK = "BENCHMARK"
SECTORS = ("Technology", "Healthcare", "Financial Services", "Consumer Cyclical", "Consumer Defensive",
           "Basic Materials", "Communication Services", "Industrials", "Energy", "Real Estate", "Utilities")
NEEDED = ("beta_10y", "vol_10y", "corr_spx_10y", "avg_corr_other_sectors", "max_drawdown_10y", "months_peak_to_recovery",
          "upside_capture_10y", "downside_capture_10y", "sharpe_10y", "sortino_10y", "worst_rolling_3y_return",
          "trailing_pe", "price_to_book", "eps_growth_cy2027", "eps_revision_cy2027_since_sep30", "fwd_pe_vs_10y_avg",
          "dividend_yield", "return_per_1pt_10y_yield", "top10_holdings_weight")

# (name, test, points). Inequalities are strict, as in the workbook.
SCORE_TESTS = [
    ("s1", lambda r: r.corr_spx_10y < 0.60, 1.8),
    ("s2", lambda r: -0.15 < r.fwd_pe_vs_10y_avg < 0.05, 1.5),
    ("s3", lambda r: r.eps_revision_cy2027_since_sep30 > 0, 1.2),
    ("s4", lambda r: 0.50 < r.downside_capture_10y < 0.90 and r.downside_capture_10y < r.upside_capture_10y, 1.2),
    ("s5", lambda r: r.eps_growth_cy2027 > r.spx_eps_growth, 1.0),
    ("s6", lambda r: r.sortino_10y > 0.9 and r.sharpe_10y > 0.6, 0.8),
    ("s7", lambda r: 0.8 < r.beta_10y < 1.2, 0.8),
    ("s8", lambda r: r.implied_roe > 0.15, 0.7),
    ("s9", lambda r: -0.02 < r.return_per_1pt_10y_yield < 0.02, 0.5),
    ("s10", lambda r: 0.015 < r.dividend_yield < 0.035 and 0.30 < r.payout_ratio < 0.60, 0.3),
    ("s11", lambda r: r.top10_holdings_weight < 0.60, 0.2),
]
# (name, safe condition, penalty if NOT safe)
RISK_TESTS = [
    ("r1", lambda r: r.max_drawdown_10y > -0.35, 1.8),
    ("r2", lambda r: 0.50 < r.downside_capture_10y < 0.90, 1.6),
    ("r3", lambda r: r.avg_corr_other_sectors < 0.60, 1.4),
    ("r4", lambda r: r.vol_10y < 0.17, 1.2),
    ("r5", lambda r: r.months_peak_to_recovery < 108, 1.0),
    ("r6", lambda r: 0.5 < r.beta_10y < 0.9, 0.8),
    ("r7", lambda r: r.worst_rolling_3y_return > -0.05, 0.8),
    ("r8", lambda r: r.fwd_pe_vs_10y_avg < 0.10, 0.6),
    ("r9", lambda r: -0.03 < r.return_per_1pt_10y_yield < 0.03, 0.4),
    ("r10", lambda r: r.top10_holdings_weight < 0.60, 0.4),
]


def load_sector_inputs(path: str = INPUTS) -> pd.DataFrame:
    """The screener inputs indexed by yahoo_sector (11 sectors + BENCHMARK). Fails loudly on a missing column or a
    missing / non-numeric value for any sector (ev_ebitda_ttm may be blank: it is not used)."""
    df = pd.read_csv(path)
    missing = [c for c in ("sector", "yahoo_sector") + NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"sector inputs: columns missing: {missing}")
    df["yahoo_sector"] = df.yahoo_sector.str.strip()
    if df.yahoo_sector.duplicated().any():
        raise ValueError(f"sector inputs: duplicate sectors {list(df.yahoo_sector[df.yahoo_sector.duplicated()])}")
    df = df.set_index("yahoo_sector")
    gone = [s for s in SECTORS + (BENCHMARK,) if s not in df.index]
    if gone:
        raise ValueError(f"sector inputs: rows missing: {gone}")
    for c in NEEDED:
        vals = pd.to_numeric(df.loc[list(SECTORS), c], errors="coerce")
        bad = list(vals[vals.isna()].index)
        if bad:
            raise ValueError(f"sector inputs: {c} missing or not a number for {bad}")
    bench = pd.to_numeric(pd.Series([df.loc[BENCHMARK, "eps_growth_cy2027"]]), errors="coerce").iloc[0]
    if pd.isna(bench):
        raise ValueError("sector inputs: eps_growth_cy2027 missing for the BENCHMARK row")
    for c in NEEDED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def score_sectors(inputs: pd.DataFrame, plan=None) -> pd.DataFrame:
    """One row per sector: inputs, implied_roe, payout_ratio, each test's pass (s1..s11, r1..r10) and points,
    score, risk, tier. Sorted by score (highest first), ties by lower risk. Scores are unrounded floats."""
    n_ow = getattr(plan, "n_overweight_sectors", 3)
    n_uw = getattr(plan, "n_underweight_sectors", 3)
    veto = getattr(plan, "sector_risk_veto", 7.0)
    df = inputs.loc[list(SECTORS)].copy()
    df["implied_roe"] = df.price_to_book / df.trailing_pe
    df["payout_ratio"] = df.trailing_pe * df.dividend_yield
    df["spx_eps_growth"] = float(inputs.loc[BENCHMARK, "eps_growth_cy2027"])   # one number for every sector
    score, risk = pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
    for name, test, pts in SCORE_TESTS:
        ok = pd.Series({s: bool(test(r)) for s, r in df.iterrows()})
        df[f"{name}_pass"], df[f"{name}_pts"] = ok, ok * pts
        score += ok * pts
    for name, safe, pen in RISK_TESTS:
        ok = pd.Series({s: bool(safe(r)) for s, r in df.iterrows()})
        df[f"{name}_pass"], df[f"{name}_pts"] = ok, (~ok) * pen
        risk += (~ok) * pen
    df["score"], df["risk"] = score, risk
    df = df.iloc[sorted(range(len(df)), key=lambda i: (-df.score.iloc[i], df.risk.iloc[i]))]
    tier = pd.Series("neutral", index=df.index)
    ow = [s for s in df.index if df.risk[s] < veto][:n_ow]
    tier[ow] = "overweight"
    uw = [s for s in df.index[::-1] if s not in ow][:n_uw]
    tier[uw] = "underweight"
    df["tier"] = tier
    df["vetoed"] = df.risk >= veto
    return df


def tilt_status(plan, asof) -> tuple:
    """(apply?, message or None). Off in "off" mode; skipped before the snapshot date unless forced."""
    if plan.sector_mode == "off":
        return False, None
    asof = pd.Timestamp(asof)
    if asof >= SNAPSHOT_DATE:
        return True, None
    if not plan.sector_tilt_in_backtests:
        return False, f"sector tilt skipped (snapshot is Oct 2026; asof={asof.date()})"
    return True, (f"WARNING look-ahead: sector tilt uses the Oct 2026 sector snapshot on {asof.date()}; "
                  "forward P/E, EPS revisions and analyst numbers are not point-in-time.")


_ANNOUNCED = set()


def announce(msg: str):
    """Print the skip / look-ahead line once per run, not at every rebalance."""
    kind = "skip" if msg.startswith("sector tilt skipped") else "warn"
    if kind not in _ANNOUNCED:
        _ANNOUNCED.add(kind)
        print(msg)


_CACHE = {}


def sector_tilt(plan) -> dict:
    """What the domestic sleeve needs: scored table, sector -> account-level cap, sector -> score, the median score."""
    key = (plan.sector_inputs, plan.n_overweight_sectors, plan.n_underweight_sectors, plan.sector_risk_veto,
           tuple(sorted(plan.sector_tier_caps.items())))
    if key not in _CACHE:
        table = score_sectors(load_sector_inputs(plan.sector_inputs), plan)
        _CACHE[key] = dict(table=table,
                           caps={s: plan.sector_tier_caps[t] for s, t in table.tier.items()},
                           scores={s: float(x) for s, x in table.score.items()},
                           median=float(statistics.median(table.score)),
                           neutral_cap=plan.sector_tier_caps["neutral"])
    return _CACHE[key]
