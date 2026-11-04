"""
build_approved_etfs.py - turn the competition's approved ETF list (PDF) into approved_etfs.csv.

    python build_approved_etfs.py                    # 25-26 list (URL below)
    python build_approved_etfs.py <url-or-pdf-path>  # e.g. the 26-27 list when it is published

Columns: ticker, name, issuer, target_country, market_tracking (Y/N), notes, expense_ratio, geo_focus, asset_class.
"Market-tracking" = a broad equity index fund for ONE country. Sector, thematic, factor (value, growth,
dividend, quality, momentum, min-vol, ESG, equal-weight), size-only (mid/small cap), active, bond and
multi-country/regional funds are N. The PDF has a "Geographical Focus" column; when it names a single
country, that is target_country. Rows the rules could not settle cleanly get "CHECK" in notes.
"""
import io, os, re, sys, urllib.request
import pandas as pd
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://globalyouth.wharton.upenn.edu/wp-content/uploads/2025/09/25-26-WGHIC-Approved-ETF-List.pdf"
OUT = os.path.join(HERE, "approved_etfs.csv")

MULTI = {"United States", "Global", "International", "European Region"}     # US is one country but never foreign
INDUSTRIES = ["N.A.", "Thematic", "Energy", "Health Care", "Industrials", "Utilities", "Real Estate", "Communications",
              "Consumer Staples", "Consumer Discretionary", "Financial", "Materials", "Technology"]
CAPS = ["Broad Market", "Large-cap", "Mid-cap", "Small-cap"]
FACTOR_WORDS = ("DIVIDEND", "DVD", "ESG", "QUALITY", "MIN VOL", "MOMENTUM", "MOAT", "IPO", "CASH COWS", "INNOVATION",
                "EQUAL WEIGHT", "SOCIAL", "CARBON", "PREMIUM INCO", "GROWTH", "VALUE", "SMALL", "MID-CAP", "INFRASTRUCTURE")
ISSUERS = [("ISHARES", "iShares"), ("VANGUARD", "Vanguard"), ("JPMORGAN", "JPMorgan"), ("SPDR", "State Street SPDR"),
           ("SELECT SECT", "State Street SPDR"), ("INVESCO", "Invesco"), ("FIRST TR", "First Trust"), ("GLOBAL X", "Global X"),
           ("SCHWAB", "Schwab"), ("VANECK", "VanEck"), ("PROSHARES", "ProShares"), ("PACER", "Pacer"), ("AVANTIS", "Avantis"),
           ("ADVISORSHARES", "AdvisorShares"), ("RENAISSANCE", "Renaissance"), ("ARK ", "ARK Invest"), ("BLACKROCK", "BlackRock"),
           ("WISDOMTREE", "WisdomTree"), ("CONSUMER STAPLES SPDR", "State Street SPDR")]
ROW = re.compile(r"^(\d+) ([A-Z.]+) (.+?) (Equity|Fixed Income) (.+) (\d*\.?\d+) ([YN])$")


def read_pdf(src: str) -> list:
    raw = urllib.request.urlopen(src).read() if src.startswith("http") else open(src, "rb").read()
    lines = []
    for page in PdfReader(io.BytesIO(raw)).pages:
        lines += [l.strip() for l in page.extract_text().splitlines()]
    return lines


def issuer(name: str) -> str:
    return next((v for k, v in ISSUERS if k in name), "Unknown")


def parse_equity(mid: str):
    """'<cap> <strategy> <industry> <geo> N.A. N.A.' -> (cap, strategy, industry, geo)"""
    cap = next((c for c in CAPS if mid.startswith(c + " ")), None)
    if cap is None:
        raise ValueError(f"unknown market-cap focus in: {mid!r}")
    rest = mid[len(cap) + 1:]
    strategy, rest = rest.split(" ", 1)
    if not rest.endswith(" N.A. N.A."):
        raise ValueError(f"unexpected equity tail in: {mid!r}")
    rest = rest[: -len(" N.A. N.A.")]
    industry = next((i for i in sorted(INDUSTRIES, key=len, reverse=True) if rest.startswith(i + " ")), None)
    if industry is None:
        raise ValueError(f"unknown industry focus in: {mid!r}")
    return cap, strategy, industry, rest[len(industry) + 1:]


def classify(rows: list) -> pd.DataFrame:
    out = []
    for line in rows:
        m = ROW.match(line)
        if not m:
            continue
        n, ticker, name, asset, mid, er, active = m.groups()
        rec = dict(ticker=ticker, name=name, issuer=issuer(name), target_country="", market_tracking="N", notes="",
                   expense_ratio=float(er), geo_focus="", asset_class=asset)
        if asset == "Fixed Income":
            rec["geo_focus"] = next((g for g in sorted(MULTI, key=len, reverse=True) if f" {g} " in f" {mid} "), "")
            rec["notes"] = "bond fund"
            out.append(rec)
            continue
        cap, strategy, industry, geo = parse_equity(mid)
        rec["geo_focus"] = geo
        single = geo not in MULTI
        if single or geo == "United States":
            rec["target_country"] = geo
        why = []
        if not single:
            why.append("multi-country/regional" if geo != "United States" else "US fund (never in the foreign sleeve)")
        if active == "Y":
            why.append("actively managed")
        if industry != "N.A.":
            why.append(f"{industry.lower()} fund")
        if cap not in ("Broad Market", "Large-cap"):
            why.append(f"{cap.lower()} only")
        if strategy != "Blend":
            why.append(f"{strategy.lower()} style")
        hit = [w for w in FACTOR_WORDS if w in name]
        if hit:
            why.append("factor/thematic name: " + ", ".join(w.lower() for w in hit))
        if "HEDGED" in name:
            why.append("hedged")
        if single and not [w for w in why if w != "hedged"]:
            rec["market_tracking"] = "Y"
            rec["notes"] = "; ".join(why)
            if cap == "Broad Market":       # MSCI country funds are usually tagged large-cap; flag the odd ones
                rec["notes"] = (rec["notes"] + "; " if rec["notes"] else "") + "CHECK: tagged broad-market, confirm it is a plain country index"
        else:
            rec["notes"] = "; ".join(why)
        out.append(rec)
    df = pd.DataFrame(out)
    if df.empty:
        raise ValueError("no ETF rows parsed - has the PDF layout changed?")
    return df


def main(src: str = URL):
    lines = read_pdf(src)
    expected = [int(l.split(" ", 1)[0]) for l in lines if re.match(r"^\d+ [A-Z.]+ ", l)]
    df = classify(lines)
    if len(df) != len(expected):
        missed = sorted(set(expected) - set(range(1, len(df) + 1)))
        raise ValueError(f"parsed {len(df)} of {len(expected)} numbered rows (check rows {missed})")
    df.to_csv(OUT, index=False)
    y = df[df.market_tracking == "Y"]
    print(f"{len(df)} ETFs, {len(y)} single-country market-tracking -> {OUT}")
    for r in y.itertuples():
        print(f"  {r.ticker:6s} {r.target_country:12s} {r.name}  {r.notes}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else URL)
