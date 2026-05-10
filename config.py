"""
Configuration for P2-ETF-DISSIPATION-ANALYSIS engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-dissipation-analysis-results"

# --- Universe definitions ---
FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}

# --- Dissipation parameters ---
WINDOWS = [30, 60, 90]               # sliding window lengths (days)
N_BINS = 20                          # number of bins for 2D histogram (KDE)
TOP_N = 3                            # number of top ETFs per universe

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
