"""
Non‑equilibrium statistical mechanics metrics:
- Entropy production rate (Kullback‑Leibler divergence forward/backward)
- Fluctuation theorem (probability of negative entropy production)
- Jarzynski equality proxy (work fluctuations)
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import kl_div

class DissipationAnalyzer:
    def __init__(self, window=60, n_bins=20):
        self.window = window
        self.n_bins = n_bins

    def _estimate_joint_pdf(self, r_t, r_t1):
        """Estimate joint PDF of (r_t, r_{t+1}) using 2D Gaussian KDE."""
        data = np.vstack([r_t, r_t1])
        try:
            kde = gaussian_kde(data)
            # Evaluate on a grid for KL divergence (optional)
            # For simplicity, we compute KL directly on the histogram
            # But we need the conditional distribution.
            # We'll use histogram binning for robustness.
            pass
        except:
            pass
        # Use 2D histogram for conditional probabilities
        H, xedges, yedges = np.histogram2d(r_t, r_t1, bins=self.n_bins)
        # Add small epsilon to avoid zeros
        H += 1e-8
        # Normalise to get joint probability
        P = H / H.sum()
        # Marginal P(r_t)
        P_x = P.sum(axis=1)
        # Conditional P(r_{t+1} | r_t)
        P_y_given_x = P / P_x[:, None]
        return P, P_x, P_y_given_x, xedges, yedges

    def entropy_production_rate(self, returns):
        """
        Returns approximate entropy production rate (non-negative).
        Computed as KL divergence between forward and reverse conditional distributions.
        """
        # Move window over the series
        T = len(returns)
        if T < self.window + 1:
            return np.nan
        rates = []
        for i in range(self.window, T-1):
            # Use sliding window to estimate conditional distributions
            # For simplicity, we use whole series to compute one value;
            # but we want rolling entropy. Let's compute over the entire series using overlapping windows?
            # Better: compute over whole available data (or rolling on last window)
            pass
        # For simplicity, we compute a single KL divergence over the entire series using adjacent pairs.
        # More robust: use a 2D histogram on all (r_t, r_{t+1}) pairs.
        r_t = returns[:-1]
        r_t1 = returns[1:]
        # Build joint histogram
        H, xedges, yedges = np.histogram2d(r_t, r_t1, bins=self.n_bins)
        H += 1e-12
        P = H / H.sum()
        P_x = P.sum(axis=1)
        P_y_given_x = P / P_x[:, None]

        # Reverse: (r_{t+1}, r_t)
        H_rev, _, _ = np.histogram2d(r_t1, r_t, bins=[xedges, yedges])
        H_rev += 1e-12
        P_rev = H_rev / H_rev.sum()
        P_rev_x = P_rev.sum(axis=1)  # marginal of r_{t+1} in reverse
        P_rev_y_given_x = P_rev / P_rev_x[:, None]   # P(r_t | r_{t+1})

        # Compute KL divergence for each grid cell, weighted by P_x
        # We need Σ_x P(x) * KL(P(y|x) || P(x|y)? Actually forward vs reverse.
        # Entropy production rate = Σ_{x,y} P(x,y) * log( P(x,y) / (P(x) P(y))? No.
        # For time-reversal asymmetry: KL( P_forward || P_reverse )
        # where P_forward = P(x,y), P_reverse = P(y,x).
        # Then entropy production rate = (1/2) * KL(P(x,y) || P(y,x))
        # Because detailed balance would give equality.
        # So we compute KL(P(x,y) || P(y,x)).
        # Flatten joint distributions
        P_flat = P.flatten()
        P_rev_flat = P_rev.flatten()
        # Avoid zeros
        eps = 1e-12
        P_flat = np.maximum(P_flat, eps)
        P_rev_flat = np.maximum(P_rev_flat, eps)
        kl = np.sum(P_flat * np.log(P_flat / P_rev_flat))
        # This is the entropy production rate (bits per step).
        return kl

    def fluctuation_theorem(self, returns):
        """Compute the probability of negative entropy production (p_neg) and ratio."""
        # For a sliding window, we can compute local entropy productions (using the above method over short windows).
        # Then estimate fraction of negative values.
        # A simple proxy: use the sign of the Jarzynski work proxy (cumulative return).
        # Here we implement: split series into non-overlapping windows of length self.window,
        # compute entropy production per window, then fraction negative.
        T = len(returns)
        n_windows = T // self.window
        if n_windows < 2:
            return 0.0, 0.0
        entropies = []
        for i in range(n_windows):
            seg = returns[i*self.window : (i+1)*self.window]
            if len(seg) < self.window:
                continue
            # Compute entropy production on this segment (using the method above)
            # For speed, we use a simplified version: compute KL on the segment's transition pairs.
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
            eps = 1e-12
            kl = np.sum(np.maximum(P_flat, eps) * np.log(np.maximum(P_flat, eps) / np.maximum(P_rev_flat, eps)))
            entropies.append(kl)
        entropies = np.array(entropies)
        p_neg = np.mean(entropies < 0)
        # Also compute the mean and std of positive entropy productions? Keep simple.
        return p_neg, np.mean(entropies)

    def jarzynski_proxy(self, returns):
        """
        Jarzynski equality: ⟨exp(-W)⟩ = exp(-ΔF). Here W is the "work" (cumulative return over window).
        Compute the average of exp(-W) over many sub‑windows.
        If ⟨exp(-W)⟩ > 1, it indicates non-equilibrium (excess work).
        """
        T = len(returns)
        step = self.window // 2
        if T < self.window:
            return np.nan
        works = []
        for i in range(0, T - self.window + 1, step):
            window_returns = returns[i:i+self.window]
            W = np.sum(window_returns)   # cumulative log return = work proxy
            works.append(np.exp(-W))
        avg_exp = np.mean(works)
        return avg_exp

    def compute_all_metrics(self, returns):
        """
        Return dictionary with entropy, p_neg, jarzynski.
        """
        entropy = self.entropy_production_rate(returns)
        p_neg, mean_entropy = self.fluctuation_theorem(returns)
        jarzynski = self.jarzynski_proxy(returns)
        return {
            "entropy_production": entropy,
            "fraction_negative": p_neg,
            "mean_window_entropy": mean_entropy,
            "jarzynski_exponential": jarzynski
        }
