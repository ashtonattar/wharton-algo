"""
bond_report.py - all approved bonds with every metric and both scores, for the IPS appendix.

    python bond_report.py                 # -> bond_report.csv
    python bond_report.py out.csv

Headers follow the bond workbook. The across-maturities score is blank for GBP/EUR while their 1-yr risk-free
rates are blank on the workbook's Settings sheet.
"""
import os
import sys

import bond_sleeve as bs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bond_report.csv")

COLUMNS = {   # our column -> workbook header (with its column letter)
    "stocktrak_no": "StockTrak #", "bond": "Bond", "currency": "Currency", "maturity": "Maturity",
    "coupon": "Coupon", "freq": "Coupons / yr", "ytm": "YTM", "cds_bp": "5-yr CDS (bp)",
    "default_risk": "Default risk (annual)", "years_left": "Years left", "bucket": "Maturity bucket",
    "clean": "Clean price", "accrued": "Accrued interest", "dirty": "Price incl. accrued",
    "mac_dur": "Macaulay duration", "mod_dur": "Modified duration", "convexity": "Convexity",
    "dv01_100k": "DV01 per $100k face", "price_1y": "Price in 1 yr", "coupons_1y": "Coupons received (1 yr)",
    "carry_1y": "1-yr carry", "rolldown": "1-yr roll-down", "ret_1y": "1-yr return (carry + roll)",
    "breakeven_bp": "Breakeven rate rise (bp)", "rf_1y": "1-yr risk-free (local)", "excess": "Excess return",
    "excess_per_dur_bp": "Excess return per unit of duration (bp)", "ret_per_dur_bp": "Return per unit of duration (bp)",
    "group_within": "Group size (same bucket + currency)", "within_score": "WITHIN-BUCKET score (0-100)",
    "within_rank": "Rank within bucket", "group_across": "Group size (same currency)",
    "across_score": "ACROSS-MATURITIES score (0-100)", "across_rank": "Rank across maturities", "key": "Ticker key"}


def report(out: str = OUT):
    s = bs.BondSettings.load()
    df = bs.score_bonds(s)[list(COLUMNS)].copy()
    df["maturity"] = df.maturity.dt.date
    for c in ("within_rank", "across_rank", "group_within", "group_across"):
        df[c] = df[c].astype("Int64")
    df.rename(columns=COLUMNS).to_csv(out, index=False, float_format="%.6g")
    usd = df[df.currency == "USD"]
    print(f"{len(df)} bonds -> {out}   (yields as of {s.settle}; USD 1-yr {s.rf_usd:.2%})")
    print(f"  USD best across maturities: {', '.join(usd.nsmallest(3, 'across_rank').bond)}")
    blank = sorted(set(df[df.across_score.isna() & (df.currency != 'USD')].currency))
    if blank:
        print(f"  across-maturities score blank for {', '.join(blank)}: enter their 1-yr risk-free on Settings")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else OUT)
