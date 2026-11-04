"""
build_approved_bonds.py - turn the bond workbook's INPUT cells into approved_bonds.csv + bond_settings.json.

    python build_approved_bonds.py                     # ../(C) Bond Model (Fixed).xlsx
    python build_approved_bonds.py <path-to-xlsx>      # e.g. after the 26-27 list or new yields are pasted in

Only input columns are read (the workbook's formula cells have no cached values; bond_sleeve.py recomputes them):
    A StockTrak #, B Bond, C Currency, D Maturity, E Coupon, F Coupons/yr, G YTM, H 5-yr CDS (bp),
    Q Convexity, V 1-yr roll-down, X Breakeven rise (bp)
Settings: B1 settlement date, B3 1-yr USD T-bill, B4/B5 GBP/EUR risk-free (may be blank), B6 recovery,
B7-B10 score weights, B11-B13 bucket cutoffs.
Do NOT use ../Book 9(Bond model).csv: its maturities, buckets and durations are wrong for most rows.
"""
import datetime as dt
import json
import os
import sys

import openpyxl
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "(C) Bond Model (Fixed).xlsx")
OUT = os.path.join(HERE, "approved_bonds.csv")
SETTINGS_OUT = os.path.join(HERE, "bond_settings.json")

COLS = {"A": "stocktrak_no", "B": "bond", "C": "currency", "D": "maturity", "E": "coupon", "F": "freq",
        "G": "ytm", "H": "cds_bp", "Q": "convexity", "V": "rolldown", "X": "breakeven_bp"}
SETTINGS = {"B1": "settle", "B3": "rf_usd", "B4": "rf_gbp", "B5": "rf_eur", "B6": "recovery",
            "B7": "w_ret_per_dur", "B8": "w_ret", "B9": "w_roll", "B10": "w_default",
            "B11": "cut1", "B12": "cut2", "B13": "cut3"}


def _input(v, what):
    if isinstance(v, str) and v.startswith("="):
        raise ValueError(f"{what} is a formula ({v[:30]}...), expected an input value")
    return v


def read_workbook(src: str = SRC):
    wb = openpyxl.load_workbook(src)                   # formulas as text: we only want inputs
    ws, st = wb["Bond Model"], wb["Settings"]
    head = [c.value for c in ws[1]]
    if head[0] != "StockTrak #" or head[1] != "Bond" or head[16] != "Convexity" or head[21] != "1-yr roll-down":
        raise ValueError(f"Bond Model columns moved: {head[:24]}")
    rows = []
    for r in range(2, ws.max_row + 1):
        if ws[f"B{r}"].value in (None, ""):
            continue
        rec = {name: _input(ws[f"{col}{r}"].value, f"Bond Model!{col}{r}") for col, name in COLS.items()}
        if not isinstance(rec["maturity"], (dt.datetime, dt.date)):
            raise ValueError(f"Bond Model!D{r}: maturity is not a date: {rec['maturity']!r}")
        rec["maturity"] = pd.Timestamp(rec["maturity"]).date().isoformat()
        rows.append(rec)
    s = {name: _input(st[cell].value, f"Settings!{cell}") for cell, name in SETTINGS.items()}
    s["settle"] = pd.Timestamp(s["settle"]).date().isoformat()
    w = s["w_ret_per_dur"] + s["w_ret"] + s["w_roll"] + s["w_default"]
    if abs(w - 1) > 1e-9:
        raise ValueError(f"score weights add to {w}, not 1")
    return pd.DataFrame(rows), s


def main(src: str = SRC):
    df, s = read_workbook(src)
    df.to_csv(OUT, index=False)
    json.dump(s, open(SETTINGS_OUT, "w"), indent=2)
    print(f"{len(df)} bonds ({', '.join(f'{c} {n}' for c, n in df.currency.value_counts().items())}) -> {OUT}")
    print(f"settings (settlement {s['settle']}, USD 1-yr {s['rf_usd']:.2%}) -> {SETTINGS_OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SRC)
