"""
Neuroscience Diagnostics for PyQuifer

Five Tier 1 empirical metrics for validating oscillatory dynamics against
published neuroscience data. All functions are pure-tensor utilities — no
nn.Module state, not in the hot path.

References:
- Lendner et al. (2020). Spectral slope as consciousness marker. Nat Comms.
- Hardstone et al. (2012). DFA for long-range temporal correlations.
- Beggs & Plenz (2003). Neuronal avalanches in cortical circuits.
- Bandt & Pompe (2002). Permutation entropy.
- Rosso et al. (2007). Complexity-entropy causality plane.
- Lempel & Ziv (1976). Sequence complexity (LZ76).
- Kaspar & Schuster (1987). Normalized LZ complexity.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


def spectral_exponent(signal: Tensor, fs: float = 1.0) -> float:
    """Estimate the 1/f spectral slope of a time series PSD.

    Computes FFT -> PSD -> log-log linear regression of S(f) vs f,
    excluding DC and Nyquist. Returns the slope directly, following
    the neuroscience convention (Lendner et al. 2020).

    Targets (neuroscience convention — slope is negative for 1/f):
    - Wake: [-1.5, -2.5] (shallower = more complex dynamics)
    - Sleep N3: [-3.0, -4.0] (steeper = less complex)

    Args:
        signal: 1-D time series, shape (T,). Needs T >= 16.
        fs: Sampling frequency in Hz (default 1.0 = per-tick).

    Returns:
        Slope of log10(PSD) vs log10(f). Negative for 1/f-like signals.
        More negative = steeper decay = less complex dynamics.
    """
    signal = signal.detach().float()
    T = signal.shape[0]
    if T < 16:
        return 0.0

    # Remove mean to suppress DC leakage
    signal = signal - signal.mean()

    # FFT -> one-sided PSD
    fft_vals = torch.fft.rfft(signal)
    psd = (fft_vals.abs() ** 2) / T

    # Frequency axis (exclude DC=0 and Nyquist)
    n_freqs = psd.shape[0]
    freqs = torch.arange(1, n_freqs, device=signal.device, dtype=torch.float32) * (fs / T)
    psd = psd[1:]  # drop DC

    # Filter out zero-power bins
    mask = psd > 0
    if mask.sum() < 4:
        return 0.0

    log_f = torch.log10(freqs[mask])
    log_psd = torch.log10(psd[mask])

    # Linear regression in log-log space: log(PSD) = slope * log(f) + c
    mean_x = log_f.mean()
    mean_y = log_psd.mean()
    denom = ((log_f - mean_x) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    slope = ((log_f - mean_x) * (log_psd - mean_y)).sum() / denom

    # Return slope directly (negative for 1/f decay) — neuroscience convention
    return slope.item()


def dfa_exponent(signal: Tensor, scales: Optional[List[int]] = None) -> float:
    """Detrended Fluctuation Analysis exponent for long-range temporal correlations.

    Cumulative sum -> detrend in non-overlapping windows -> RMS fluctuation F(n)
    -> log-log slope gives the DFA exponent alpha.

    Target: alpha in [0.6, 0.8] for LRTC (Hardstone et al. 2012).
    alpha = 0.5 -> white noise, alpha = 1.0 -> 1/f noise, alpha = 1.5 -> Brownian.

    Args:
        signal: 1-D time series, shape (T,). Needs T >= 32.
        scales: Window sizes for detrending. Default [4,8,16,32,64,128]
                (capped at T//4).

    Returns:
        DFA exponent alpha.
    """
    signal = signal.detach().float()
    T = signal.shape[0]
    if T < 32:
        return 0.5

    # Cumulative sum of mean-subtracted signal (integration step)
    y = (signal - signal.mean()).cumsum(0)

    if scales is None:
        scales = [4, 8, 16, 32, 64, 128]
    # Keep only scales that fit
    scales = [s for s in scales if s <= T // 4 and s >= 4]
    if len(scales) < 2:
        return 0.5

    log_scales = []
    log_flucts = []

    for n in scales:
        n_segments = T // n
        if n_segments < 1:
            continue
        # Trim to fit exact segments
        trimmed = y[:n_segments * n].reshape(n_segments, n)

        # Linear detrend each segment: fit y = a*x + b, subtract
        x = torch.arange(n, device=signal.device, dtype=torch.float32)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        # Vectorized over segments
        seg_means = trimmed.mean(dim=1, keepdim=True)
        slopes = ((trimmed - seg_means) * (x - x_mean).unsqueeze(0)).sum(dim=1, keepdim=True) / x_var
        intercepts = seg_means - slopes * x_mean
        trends = slopes * x.unsqueeze(0) + intercepts
        residuals = trimmed - trends

        # RMS fluctuation for this scale
        F_n = (residuals ** 2).mean().sqrt()
        if F_n > 0:
            log_scales.append(math.log(n))
            log_flucts.append(math.log(F_n.item()))

    if len(log_scales) < 2:
        return 0.5

    # Linear regression in log-log space
    ls = torch.tensor(log_scales)
    lf = torch.tensor(log_flucts)
    mean_x = ls.mean()
    mean_y = lf.mean()
    slope = ((ls - mean_x) * (lf - mean_y)).sum() / ((ls - mean_x) ** 2).sum()
    return slope.item()


def lempel_ziv_complexity(signal: Tensor, threshold: Optional[float] = None) -> float:
    """Spontaneous Lempel-Ziv complexity (LZ76) of a binarized time series.

    Binarize signal (above/below median or given threshold), then count the
    number of distinct subsequences using the Kaspar-Schuster (1987) variant
    of the LZ76 algorithm. Normalize by the asymptotic random baseline.

    Target: LZc in [0.4, 0.8] for conscious-like spontaneous dynamics.
    LZc ~ 0.2 for seizure/deep sleep, LZc ~ 0.6-0.7 for wakefulness.

    Args:
        signal: 1-D time series, shape (T,). Needs T >= 8.
        threshold: Binarization threshold. Default: median.

    Returns:
        Normalized LZ complexity. Values near 1.0 = random, < 0.5 = structured.
    """
    signal = signal.detach().float()
    T = signal.shape[0]
    if T < 8:
        return 0.0

    if threshold is None:
        threshold = signal.median().item()

    # Binarize
    s = (signal > threshold).tolist()
    n = len(s)

    # Kaspar-Schuster (1987) LZ76 implementation
    # Scan string, count number of new words (substrings not seen before)
    c = 1  # complexity counter (first symbol is always new)
    i = 1  # start of current new word
    k = 1  # length of current candidate

    while i + k - 1 < n:
        # Check if s[i:i+k] appears as substring in s[0:i+k-1]
        # (the "vocabulary" is everything before the end of current candidate)
        substr = s[i: i + k]
        vocab_end = i + k - 1  # search in s[0:vocab_end]
        found = False
        for j in range(vocab_end - k + 1):
            if s[j: j + k] == substr:
                found = True
                break

        if found:
            k += 1
        else:
            c += 1
            i = i + k
            k = 1

    # Normalize by asymptotic random baseline: c_random ~ n / log2(n)
    # Kaspar & Schuster (1987): c_norm = c(n) * log2(n) / n → 1.0 as n → ∞.
    # Values slightly > 1.0 at finite n are a known artifact (finite-size effect).
    if n > 1:
        c_norm = c * math.log2(n) / n
    else:
        c_norm = 0.0

    return c_norm


def avalanche_statistics(signal: Tensor, threshold: float = 1.0) -> Dict[str, float]:
    """Compute neuronal avalanche size and duration distributions.

    Define avalanches as contiguous epochs where signal > mean + threshold*std.
    Fit power-law exponents via log-log regression on the complementary CDF.

    For best results, pass neural ACTIVITY signal (sum of oscillator outputs),
    not the order parameter R(t). R(t) is too smooth for proper avalanche
    scaling; individual oscillator activity preserves the burst structure
    needed for power-law detection.

    Targets (Beggs & Plenz 2003):
    - Size exponent ~ -1.5
    - Duration exponent ~ -2.0

    Args:
        signal: 1-D time series of neural activity, shape (T,).
        threshold: Number of standard deviations above mean for avalanche onset.

    Returns:
        Dict with keys: size_exponent, duration_exponent, n_avalanches, mean_size.
    """
    signal = signal.detach().float()
    T = signal.shape[0]
    result = {
        'size_exponent': 0.0,
        'duration_exponent': 0.0,
        'n_avalanches': 0,
        'mean_size': 0.0,
    }
    if T < 16:
        return result

    mu = signal.mean()
    sigma = signal.std()
    if sigma < 1e-8:
        return result

    thresh_val = (mu + threshold * sigma).item()
    above = (signal > thresh_val).tolist()

    # Extract avalanche sizes (sum of signal above threshold) and durations
    sizes = []
    durations = []
    in_avalanche = False
    current_size = 0.0
    current_dur = 0

    signal_list = signal.tolist()
    for i in range(T):
        if above[i]:
            if not in_avalanche:
                in_avalanche = True
                current_size = 0.0
                current_dur = 0
            current_size += signal_list[i] - thresh_val
            current_dur += 1
        else:
            if in_avalanche:
                sizes.append(max(current_size, 1e-8))
                durations.append(current_dur)
                in_avalanche = False
    # Close trailing avalanche
    if in_avalanche:
        sizes.append(max(current_size, 1e-8))
        durations.append(current_dur)

    n_av = len(sizes)
    result['n_avalanches'] = n_av
    if n_av < 5:
        return result

    result['mean_size'] = sum(sizes) / n_av

    # Fit power-law exponents via complementary CDF (CCDF)
    def _fit_exponent(values: List[float]) -> float:
        """Fit power-law exponent from CCDF: P(X >= x) ~ x^alpha."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        log_x = []
        log_p = []
        for i, v in enumerate(sorted_vals):
            if v > 0:
                # CCDF: probability of being >= this value
                ccdf = (n - i) / n
                if ccdf > 0:
                    log_x.append(math.log(v))
                    log_p.append(math.log(ccdf))
        if len(log_x) < 3:
            return 0.0
        lx = torch.tensor(log_x)
        lp = torch.tensor(log_p)
        mx = lx.mean()
        mp = lp.mean()
        denom = ((lx - mx) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        slope = ((lx - mx) * (lp - mp)).sum() / denom
        return slope.item()

    result['size_exponent'] = _fit_exponent(sizes)
    result['duration_exponent'] = _fit_exponent([float(d) for d in durations])

    return result


def complexity_entropy(signal: Tensor, m: int = 5) -> Tuple[float, float]:
    """Position on the complexity-entropy plane (Bandt-Pompe + Jensen-Shannon).

    Computes permutation entropy H (normalized) and Jensen-Shannon statistical
    complexity C from ordinal patterns of embedding dimension m.

    Uses m=5 by default (120 ordinal patterns), matching standard neuroscience
    practice (Rosso et al. 2007). Higher m gives better discrimination between
    complex and random dynamics.

    Targets for conscious-like dynamics (m=5):
    - H in [0.7, 0.9] (high but not maximal entropy)
    - C in [0.05, 0.25] (intermediate complexity, not random or periodic)

    Args:
        signal: 1-D time series, shape (T,). Needs T >= m! + m (125 for m=5).
        m: Embedding dimension (default 5 -> 120 ordinal patterns).

    Returns:
        (H, C) -- normalized permutation entropy and statistical complexity.
    """
    signal = signal.detach().float()
    T = signal.shape[0]
    n_patterns = math.factorial(m)

    if T < n_patterns + m:
        return (0.0, 0.0)

    # Count ordinal patterns
    signal_list = signal.tolist()
    pattern_counts = {}
    n_windows = T - m + 1

    for i in range(n_windows):
        window = signal_list[i: i + m]
        # Ordinal pattern: argsort (rank ordering)
        pattern = tuple(sorted(range(m), key=lambda k: window[k]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Convert to probability distribution
    total = sum(pattern_counts.values())

    # Build full probability vector (all n_patterns entries, 0 for unobserved)
    # Use a canonical ordering of all possible permutations
    from itertools import permutations
    all_patterns = list(permutations(range(m)))
    full_probs = [pattern_counts.get(p, 0) / total for p in all_patterns]

    # Normalized permutation entropy
    H_max = math.log(n_patterns)
    if H_max < 1e-12:
        return (0.0, 0.0)

    H = -sum(p * math.log(p) for p in full_probs if p > 0) / H_max

    # Jensen-Shannon statistical complexity C = Q_JS * H_S
    # Q_JS = JS divergence between P and uniform, normalized by max possible
    uniform_p = 1.0 / n_patterns

    # JS divergence: D_JS(P||U) = 0.5 * KL(P||M) + 0.5 * KL(U||M)
    # where M = (P + U) / 2
    js_div = 0.0
    for p in full_probs:
        m_val = (p + uniform_p) / 2.0
        if p > 0 and m_val > 0:
            js_div += 0.5 * p * math.log(p / m_val)
        if m_val > 0:
            js_div += 0.5 * uniform_p * math.log(uniform_p / m_val)

    # Normalization constant Q_0: max JS divergence for K bins
    # Q_0 = -0.5 * {(K+1)/K * ln((K+1)/(2K)) + ln(K) - 2*ln(2K)} / ln(K)
    # Simplified from Rosso et al. 2007 Eq. (5)
    K = n_patterns
    # Max JS divergence occurs for a delta distribution (one bin = 1, rest = 0)
    # vs uniform. We compute it directly:
    # D_JS(delta || uniform) = 0.5 * [1*ln(1/((1+1/K)/2)) + sum_{i!=peak} 0]
    #                        + 0.5 * [K * (1/K)*ln((1/K)/((p_i+1/K)/2))]
    # Peak bin: p=1, M=(1+1/K)/2
    # Non-peak bins (K-1 of them): p=0, M=(0+1/K)/2=1/(2K)
    m_peak = (1.0 + uniform_p) / 2.0
    m_zero = uniform_p / 2.0

    Q_0 = 0.0
    # KL(delta || M):
    Q_0 += 0.5 * 1.0 * math.log(1.0 / m_peak)  # peak bin
    # KL(uniform || M):
    Q_0 += 0.5 * uniform_p * math.log(uniform_p / m_peak)  # peak bin
    Q_0 += 0.5 * (K - 1) * uniform_p * math.log(uniform_p / m_zero)  # non-peak bins

    if abs(Q_0) < 1e-12:
        Q_0 = 1.0

    Q_JS = js_div / Q_0

    C = Q_JS * H

    return (H, C)
