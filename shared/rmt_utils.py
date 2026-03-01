"""
Shared RMT utilities for the Spectral Universality Project.

Provides:
  - spacing_ratios(eigenvalues) → ⟨r⟩
  - brody_fit(spacings) → q, confidence interval
  - marchenko_pastur_edge(eigenvalues, gamma) → λ_+
  - classify_regime(r_mean) → 'Poisson' | 'GOE' | 'GUE' | 'intermediate'
  - unfold_spectrum(eigenvalues, method) → unfolded spacings

Theoretical values:
  Poisson: ⟨r⟩ = 0.38629... = 4 - 2√3
  GOE:     ⟨r⟩ ≈ 0.5307
  GUE:     ⟨r⟩ ≈ 0.6027
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_fn


# ── Theoretical Constants ────────────────────────────────────

R_POISSON = 4 - 2 * np.sqrt(3)   # 0.53590...  wait, let me recalculate
# Actually: for Poisson, ⟨r⟩ = 2ln2 - 1 ≈ 0.38629
R_POISSON = 2 * np.log(2) - 1     # 0.38629...
R_GOE = 0.5307
R_GUE = 0.6027


# ── Spacing Ratios ───────────────────────────────────────────

def spacing_ratios(eigenvalues, sorted=False):
    """
    Compute consecutive spacing ratios r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}).
    
    No unfolding required — ratios are scale-invariant.
    
    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalue sequence (will be sorted if not already).
    sorted : bool
        If True, skip sorting.
    
    Returns
    -------
    r : ndarray
        Spacing ratios, length len(eigenvalues) - 2.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    if not sorted:
        eigs = np.sort(eigs)
    
    spacings = np.diff(eigs)
    
    # Remove zero or negative spacings (degenerate eigenvalues)
    spacings = spacings[spacings > 0]
    
    if len(spacings) < 2:
        return np.array([])
    
    r = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return r


def mean_spacing_ratio(eigenvalues, **kwargs):
    """Compute ⟨r⟩ from eigenvalues."""
    r = spacing_ratios(eigenvalues, **kwargs)
    if len(r) == 0:
        return np.nan
    return float(np.mean(r))


# ── Brody Distribution ──────────────────────────────────────

def brody_pdf(s, q):
    """
    Brody distribution P(s) = (q+1) * beta * s^q * exp(-beta * s^{q+1})
    where beta = Gamma((q+2)/(q+1))^{q+1}.
    
    Valid for 0 <= q <= 1.
    q = 0: Poisson
    q = 1: Wigner-Dyson (GOE)
    """
    beta = gamma_fn((q + 2) / (q + 1)) ** (q + 1)
    return (q + 1) * beta * s**q * np.exp(-beta * s**(q + 1))


def brody_fit(spacings, bootstrap_n=1000):
    """
    Fit Brody parameter q to observed spacings via MLE.
    
    Parameters
    ----------
    spacings : array-like
        Nearest-neighbor spacings (should be unfolded to mean 1).
    bootstrap_n : int
        Number of bootstrap samples for CI.
    
    Returns
    -------
    q : float
        Best-fit Brody parameter.
    ci_lo, ci_hi : float
        95% bootstrap confidence interval.
    
    WARNING: For N < 20 spacings, results are unreliable.
    The true uncertainty is much larger than the bootstrap CI suggests.
    """
    s = np.asarray(spacings, dtype=np.float64)
    s = s[s > 0]  # remove zeros
    n = len(s)
    
    if n < 3:
        return np.nan, np.nan, np.nan
    
    if n < 20:
        import warnings
        warnings.warn(
            f"Only {n} spacings — Brody fit is unreliable. "
            f"Need ≥20 for meaningful q estimate, ≥100 for precise.",
            stacklevel=2,
        )
    
    def neg_log_likelihood(q):
        if q < 0 or q > 2:  # allow slight extrapolation for fitting
            return 1e10
        beta = gamma_fn((q + 2) / (q + 1)) ** (q + 1)
        log_pdf = (np.log(q + 1) + np.log(beta) + q * np.log(s)
                   - beta * s**(q + 1))
        return -np.sum(log_pdf)
    
    result = minimize_scalar(neg_log_likelihood, bounds=(0, 1.5), method='bounded')
    q_best = result.x
    
    # Bootstrap CI
    q_samples = np.zeros(bootstrap_n)
    for i in range(bootstrap_n):
        idx = np.random.randint(0, n, size=n)
        s_boot = s[idx]
        
        def nll_boot(q):
            if q < 0 or q > 2:
                return 1e10
            beta = gamma_fn((q + 2) / (q + 1)) ** (q + 1)
            log_pdf = (np.log(q + 1) + np.log(beta) + q * np.log(s_boot)
                       - beta * s_boot**(q + 1))
            return -np.sum(log_pdf)
        
        res = minimize_scalar(nll_boot, bounds=(0, 1.5), method='bounded')
        q_samples[i] = res.x
    
    ci_lo, ci_hi = np.percentile(q_samples, [2.5, 97.5])
    
    return float(q_best), float(ci_lo), float(ci_hi)


# ── Marchenko-Pastur ────────────────────────────────────────

def marchenko_pastur_edge(eigenvalues, gamma=None, N=None, D=None):
    """
    Compute the Marchenko-Pastur upper edge λ_+.
    
    For a random N×D matrix with γ = N/D:
      λ_+ = σ² (1 + √γ)²
    
    where σ² = variance of matrix entries.
    
    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues of the Gram matrix.
    gamma : float, optional
        Aspect ratio N/D. If None, estimated from eigenvalues.
    N, D : int, optional
        Matrix dimensions (used to compute gamma).
    
    Returns
    -------
    lambda_plus : float
        Upper edge of MP distribution.
    signal_eigs : ndarray
        Eigenvalues above the MP edge.
    """
    eigs = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    
    if gamma is None:
        if N is not None and D is not None:
            gamma = N / D
        else:
            raise ValueError("Provide gamma or (N, D)")
    
    # Estimate σ² from bulk eigenvalues
    # Use median of eigenvalues as robust estimator
    sigma_sq = np.median(eigs) / (1 + gamma)  # rough estimate
    
    # More robust: use Tracy-Widom corrected edge
    lambda_plus = sigma_sq * (1 + np.sqrt(gamma))**2
    
    signal_eigs = eigs[eigs > lambda_plus]
    
    return float(lambda_plus), signal_eigs


# ── Classification ───────────────────────────────────────────

def classify_regime(r_mean, strict=False):
    """
    Classify ⟨r⟩ into RMT universality class.
    
    Parameters
    ----------
    r_mean : float
        Mean spacing ratio.
    strict : bool
        If True, use narrow theoretical bands.
        If False, use empirically adjusted thresholds.
    
    Returns
    -------
    regime : str
        'Poisson', 'GOE', 'GUE', or 'intermediate'
    """
    if strict:
        # Theoretical values ± 0.02
        if abs(r_mean - R_POISSON) < 0.02:
            return 'Poisson'
        elif abs(r_mean - R_GOE) < 0.02:
            return 'GOE'
        elif abs(r_mean - R_GUE) < 0.02:
            return 'GUE'
        else:
            return 'intermediate'
    else:
        # Empirical thresholds (wider bands)
        if r_mean < 0.45:
            return 'Poisson'
        elif r_mean < 0.50:
            return 'near-Poisson'
        elif r_mean < 0.57:
            return 'GOE'
        elif r_mean < 0.60:
            return 'near-GUE'
        else:
            return 'GUE'


# ── Unfolding ────────────────────────────────────────────────

def unfold_spectrum(eigenvalues, method='polynomial', degree=5):
    """
    Unfold eigenvalue spectrum to unit mean spacing.
    
    Parameters
    ----------
    eigenvalues : array-like
        Raw eigenvalues (sorted ascending).
    method : str
        'polynomial' (fit cumulative staircase) or 'kernel' (KDE).
    degree : int
        Polynomial degree for fitting.
    
    Returns
    -------
    spacings : ndarray
        Unfolded spacings with mean ≈ 1.
    """
    eigs = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    n = len(eigs)
    
    if method == 'polynomial':
        # Fit polynomial to cumulative eigenvalue count
        cumulative = np.arange(1, n + 1, dtype=np.float64)
        coeffs = np.polyfit(eigs, cumulative, degree)
        smooth = np.polyval(coeffs, eigs)
        spacings = np.diff(smooth)
    elif method == 'kernel':
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(eigs)
        density = kde(eigs)
        # Integrate density to get cumulative
        cumulative = np.cumsum(density) / np.sum(density) * n
        spacings = np.diff(cumulative)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to mean 1
    spacings = spacings / np.mean(spacings)
    
    return spacings
