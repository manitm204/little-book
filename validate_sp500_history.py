"""Compare our rebuilt membership with an independent dataset (github.com/fja05680/sp500).

Prints year-end overlap and the tickers each side has that the other lacks.
"""
import sys
from io import StringIO
from urllib.parse import quote

import pandas as pd
import requests

from build_sp500_history import OUT, members_on

REFERENCE_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/" + quote(
    "S&P 500 Historical Components & Changes (Updated).csv"
)


# Same company, spelled differently. Our log uses latest tickers; the reference often uses older ones.
ALIASES = {
    "ABC": "COR", "ABKFQ": "AMBC", "ANRZQ": "ANR", "ANTM": "ELV", "ARNC": "HWM", "BBT": "TFC", "BHGE": "BKR",
    "BK": "BNY", "BLL": "BALL", "BTUUQ": "BTU", "CCTYQ": "CC", "CCU": "IHRT", "CDAY": "DAY", "CITGQ": "CIT",
    "COG": "CTRA", "CPNLQ": "CPN", "CTL": "LUMN", "CVC": "RMG", "DALRQ": "DAL", "DCNAQ": "DAN", "DDR": "SITC",
    "DISCA": "WBD", "DPHIQ": "DPH", "DXC": "CSC", "EKDKQ": "KODK", "EQR": "VMRK", "ESV": "VAL", "FB": "META",
    "FBHS": "FBIN", "FI": "FISV", "FII": "FHI", "FLT": "CPAY", "FSL": "FSLB", "HCP": "DOC", "HET": "CZR",
    "HRS": "LHX", "IR": "TT", "JEC": "J", "KDP": "DPS", "KORS": "CPRI", "KRFT": "KHC", "LB": "BBWI", "MMC": "MRSH",
    "MTLQQ": "GM", "MWV": "WRK", "MYL": "VTRS", "NLOK": "GEN", "PARA": "SKYD", "CBS": "SKYD", "VIAC": "SKYD",
    "PSKY": "SKYD", "PEAK": "DOC", "PKI": "RVTY", "PX": "LIN", "RE": "EG", "RSHCQ": "RSH", "RX": "IMS",
    "SYMC": "GEN", "TMK": "GL", "TRB": "TRCO", "UTX": "RTX", "VSTNQ": "VC", "WAMUQ": "COOP", "WLTW": "WTW",
    "AABA": "YHOO", "PTC": "PAR", "AA": "HWM", "DXC": "CSC", "IR": "TT", "KDP": "DPS",
}


def load_reference():
    path = OUT / "raw" / "reference_components.csv"
    if not path.exists():
        response = requests.get(REFERENCE_URL, timeout=120)
        response.raise_for_status()
        path.write_text(response.text)
    reference = pd.read_csv(StringIO(path.read_text()), parse_dates=["date"]).set_index("date").sort_index()
    return reference["tickers"]


def main():
    verbose = "--diffs" in sys.argv
    spells = pd.read_csv(OUT / "sp500_spells.csv", parse_dates=["start", "end"])
    reference = load_reference()
    print(f"reference covers {reference.index.min().date()} to {reference.index.max().date()}")
    print(f"{'date':<12}{'ours':>6}{'ref':>6}{'shared':>8}{'only_ours':>11}{'only_ref':>10}")
    for year in range(2005, 2027):
        date = pd.Timestamp(f"{year}-06-30")
        ref_date = reference.index[reference.index <= date].max()
        ours = {ALIASES.get(t, t).replace("-", ".") for t in members_on(spells, ref_date)["ticker"]}
        theirs = {ALIASES.get(t, t) for t in reference.loc[ref_date].split(",")}
        print(f"{ref_date.date()!s:<12}{len(ours):>6}{len(theirs):>6}{len(ours & theirs):>8}"
              f"{len(ours - theirs):>11}{len(theirs - ours):>10}")
        if verbose:
            print("   only ours:", sorted(ours - theirs))
            print("   only ref :", sorted(theirs - ours))


if __name__ == "__main__":
    main()
