"""
Non‑equilibrium statistical mechanics metrics:
- Entropy production rate (Kullback‑Leibler divergence forward/backward)
- Fluctuation theorem (fraction of windows with negative cumulative return)
- Jarzynski equality proxy (work fluctuations)
"""

import numpy as np
from scipy.stats import gaussian_kde

class DissipationAnalyzer:
    def __init__(self, window=60, n_bins=20):
        self.window = window
        self.n_bins = n_bins

    def entropy_production_rate(self, returns):
        """
        Returns approximate entropy production rate (non-negative).
        Computed as KL divergence between forward and reverse joint distributions.
        """
        r_t = returns[:-1]
        r_t1 = returns[1:]
        # Build joint histogram
        H, xedges, yedges = np.histogram2d(r_t, r_t1, bins=self.n_bins)
        H += 1e-12
        P = H / H.sum()
        # Reverse joint: (r_t1, r_t)
        H_rev, _, _ = np.histogram2d(r_t1, r_t, bins=[xedges, yedges])
        H_rev += 1e-12
        P_rev = H_rev / H_rev.sum()
        # KL(P(x,y) || P(y,x))
        P_flat = P.flatten()
        P_rev_flat = P_rev.flatten()
        eps = 1e-12
        kl = np.sum(np.maximum(P_flat, eps) * np.log(np.maximum(P_flat, eps) / np.maximum(P_rev_flat, eps)))
        return kl

    def fluctuation_theorem(self, returns):
        """
        Compute fraction of windows where cumulative return is negative (proxy for negative entropy production).
        Also return mean entropy production per window (using KL, but keep for compatibility).
        """
        T = len(returns)
        window = self.window
        if T < window:
            return 0.0, 0.0
        n_windows = T // window
        neg_count = 0
        entropy_productions = []
        for i in range(n_windows):
            seg = returns[i*window : (i+1)*window]
            if len(seg) != window:
                continue
            # Cumulative return (log return sum) as work proxy
            work = np.sum(seg)
            if work < 0:
                neg_count += 1
            # Also compute entropy production on this segment (optional)
            try:
                r_t = seg[:-1]
                r_t1 = seg[1:]
                H, xedges, yedges = np.histogram2d(r_t, r_t1, bins=self.n_bins)
                H += 1e-12
                P = H / H.sum()
                H_rev, _, _ = np.histogram2d(r_t1, r_t, bins=[xedges, yedges])
                H_rev += 1e-12
                P_rev = H_rev / H_rev.sum()
                P_flat = P.flatten()
                P_rev_flat = P_rev.flatten()
                kl = np.sum(np.maximum(P_flat, 1e-12) * np.log(np.maximum(P_flat, 1e-12) / np.maximum(P_rev_flat, 1e-12)))
                entropy_productions.append(kl)
            except:
                entropy_productions.append(0.0)
        fraction_negative = neg_count / n_windows
        mean_entropy = np.mean(entropy_productions) if entropy_productions else 0.0
        return fraction_negative, mean_entropy

    def jarzynski_proxy(self, returns):
        """
        Jarzynski equality: ⟨exp(-W)⟩ = exp(-ΔF). Compute average of exp(-cumulative return) over sliding windows.
        """
        T = len(returns)
        step = self.window // 2
        if T < self.window:
            return np.nan
        works = []
        for i in range(0, T - self.window + 1, step):
            window_returns = returns[i:i+self.window]
            W = np.sum(window_returns)
            works.append(np.exp(-W))
        return np.mean(works)

    def compute_all_metrics(self, returns):
        entropy = self.entropy_production_rate(returns)
        p_neg, mean_entropy = self.fluctuation_theorem(returns)
        jarzynski = self.jarzynski_proxy(returns)
        return {
            "entropy_production": entropy,
            "fraction_negative_entropy": p_neg,
            "mean_window_entropy": mean_entropy,
            "jarzynski_exponential": jarzynski
        }
