# P2-ETF-DISSIPATION-ANALYSIS

**Non‑equilibrium statistical mechanics for detecting market instability.**  
Uses entropy production rate (Kullback‑Leibler divergence), fluctuation theorem, and Jarzynski equality to identify ETFs far from equilibrium.

## Features

- Computes entropy production rate as time‑reversal asymmetry (forward vs reverse conditional probabilities).
- Evaluates fluctuation theorem (fraction of negative entropy events).
- Jarzynski proxy: average exp(-cumulative return) over sliding windows.
- Tests 30, 60, 90‑day windows and selects the one with highest average entropy production (most unstable).
- Outputs ranking by entropy production (higher = more unstable).

## Data

Uses `P2SAMAPA/fi-etf-macro-signal-master-data` (2008–present).  
Results pushed to `P2SAMAPA/p2-etf-dissipation-analysis-results`.

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DISSIPATION-ANALYSIS.git
cd P2-ETF-DISSIPATION-ANALYSIS
pip install -r requirements.txt
