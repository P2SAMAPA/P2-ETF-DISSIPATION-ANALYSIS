"""
Trainer for dissipation analysis: compute metrics for each window, pick best window (max average entropy production), rank ETFs.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from dissipation_analyzer import DissipationAnalyzer
import push_results

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        # For each window, compute metrics for each ticker
        per_window = {}
        for w in config.WINDOWS:
            per_window[w] = {}
            analyzer = DissipationAnalyzer(window=w, n_bins=config.N_BINS)
            for ticker in tickers:
                if ticker not in returns.columns:
                    continue
                series = returns[ticker].dropna().values
                if len(series) < w:
                    per_window[w][ticker] = {"entropy_production": np.nan}
                    continue
                metrics = analyzer.compute_all_metrics(series)
                per_window[w][ticker] = metrics

        # Choose best window per universe: maximize average entropy production across all tickers (excluding NaN)
        best_window = None
        best_avg_entropy = -np.inf
        for w, ticker_dict in per_window.items():
            entropies = [v["entropy_production"] for v in ticker_dict.values() if not np.isnan(v["entropy_production"])]
            if entropies:
                mean_ent = np.mean(entropies)
                if mean_ent > best_avg_entropy:
                    best_avg_entropy = mean_ent
                    best_window = w
        if best_window is None:
            print(f"  No valid metrics for any window in {universe_name}")
            continue

        print(f"  Best window for {universe_name}: {best_window} days (avg entropy {best_avg_entropy:.4f})")
        best_data = per_window[best_window]

        # Create ranking: sort by entropy production (descending = most unstable)
        rankings = []
        for ticker, metrics in best_data.items():
            ent = metrics.get("entropy_production")
            if np.isnan(ent):
                continue
            rankings.append({
                "ticker": ticker,
                "entropy_production": ent,
                "fraction_negative_entropy": metrics.get("fraction_negative", 0.0),
                "mean_window_entropy": metrics.get("mean_window_entropy", 0.0),
                "jarzynski_exponential": metrics.get("jarzynski_exponential", 0.0)
            })
        rankings = sorted(rankings, key=lambda x: x["entropy_production"], reverse=True)

        universe_results = {
            "selected_window": best_window,
            "average_entropy": best_avg_entropy,
            "rankings": rankings[:config.TOP_N],
            "all_tickers": {r["ticker"]: {k: v for k, v in r.items() if k != "ticker"} for r in rankings}
        }
        all_results[universe_name] = universe_results

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/dissipation_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Dissipation analysis complete ===")

if __name__ == "__main__":
    main()
