"""
Non‑equilibrium statistical mechanics metrics:
- Entropy production rate (Kullback‑Leibler divergence forward/backward)
- Jarzynski equality proxy (work fluctuations)
"""

import numpy as np

class DissipationAnalyzer:
    def __init__(self, window=60, n_bins=20):
        self.window = window
        self.n_bins = n_bins

    def entropy_production_rate(self, returns):
        """
        Returns entropy production rate (>=0) using KL( P(x,y) || P(y,x) ).
        """
        r_t = returns[:-1]
        r_t1 = returns[1:]
        # Build joint histogram
        H, xedges, yedges = np.histogram2d(r_t, r_t1, bins=self.n_bins)
        H += 1e-12
        P = H / H.sum()
        # Reverse joint
        H_rev, _, _ = np.histogram2d(r_t1, r_t, bins=[xedges, yedges])
        H_rev += 1e-12
        P_rev = H_rev / H_rev.sum()
        # Flatten and compute KL
        P_flat = P.flatten()
        P_rev_flat = P_rev.flatten()
        kl = np.sum(P_flat * np.log(P_flat / P_rev_flat))
        return kl

    def jarzynski_proxy(self, returns):
        """
        Jarzynski equality: ⟨exp(-W)⟩ over sliding windows.
        Returns the average of exp(-cumulative return) across windows.
        """
        T = len(returns)
        if T < self.window:
            return np.nan
        step = self.window // 2
        works = []
        for i in range(0, T - self.window + 1, step):
            window_returns = returns[i:i+self.window]
            W = np.sum(window_returns)   # cumulative log return = work proxy
            works.append(np.exp(-W))
        return np.mean(works)

    def compute_all_metrics(self, returns):
        """
        Return entropy production and Jarzynski proxy.
        """
        entropy = self.entropy_production_rate(returns)
        jarzynski = self.jarzynski_proxy(returns)
        return {
            "entropy_production": entropy,
            "jarzynski_exponential": jarzynski
        }
