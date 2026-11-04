"""
sector_report.py - all 11 sectors with every input, each screener test's pass/fail and points, score, risk and tier,
for the IPS appendix.

    python sector_report.py               # -> sector_report.csv
    python sector_report.py out.csv

Scores follow the workbook's Sector Screener sheet (C93:C103, K3:K13), with the EPS-growth test compared to the
S&P's 16% for every sector (the sheet's G44 reference is not locked).
"""
import os
import sys

import sector_sleeve as ss
from laura_strategy import Plan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sector_report.csv")


def report(out: str = OUT):
    plan = Plan()
    df = ss.score_sectors(ss.load_sector_inputs(plan.sector_inputs), plan)
    tests = [c for n, _, _ in ss.SCORE_TESTS + ss.RISK_TESTS for c in (f"{n}_pass", f"{n}_pts")]
    first = ["sector", "score", "risk", "tier", "vetoed", "implied_roe", "payout_ratio", "spx_eps_growth"]
    rest = [c for c in df.columns if c not in first + tests]
    df["tier_cap"] = df.tier.map(plan.sector_tier_caps)
    df = df[first + ["tier_cap"] + tests + rest]
    df.round({"score": 1, "risk": 1}).to_csv(out, index_label="yahoo_sector", float_format="%.6g")
    print(f"{len(df)} sectors -> {out}   (S&P EPS growth {df.spx_eps_growth.iloc[0]:.0%}; risk veto {plan.sector_risk_veto})")
    print(f"  {'Sector':24s} {'Score':>5s} {'Risk':>5s}  {'Tier':20s} {'Cap':>4s}")
    for s, r in df.iterrows():
        tier = r.tier.capitalize() + (" (vetoed)" if r.vetoed and r.tier == "neutral" else "")
        print(f"  {s:24s} {r.score:5.1f} {r.risk:5.1f}  {tier:20s} {r.tier_cap:4.0%}")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else OUT)
