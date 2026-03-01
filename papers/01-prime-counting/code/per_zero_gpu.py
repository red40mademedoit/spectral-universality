#!/usr/bin/env python3
"""
Per-Zero Hypothesis Test — GPU Accelerated (B200)
==================================================

For each known Riemann zeta zero γₙ:
  1. Predict FFT bin k_n via v2 mapping: k = γ·N/(2π·⟨p/log p⟩)
  2. Measure observed amplitude A_n at bin k_n
  3. Compare to local spectral neighborhood → SNR
  4. Analytical p-value (Rayleigh null)
  5. Local percentile p-value (vs spectral neighbors)
  6. GPU permutation test (shuffle gaps, recompute FFT at target bins)
  7. Bonferroni correction for multiple testing
  8. Amplitude decay test: A_n vs 1/|ρ_n| = 1/√(1/4 + γ_n²)
  9. Also runs the v2 "ladder" test for comparison with 10⁸ baseline

No free parameters. Exact frequency predictions per zero.

Usage:
    python3 per_zero_gpu.py --limit 1e9
    python3 per_zero_gpu.py --limit 1e10 --n-perms 5000
    python3 per_zero_gpu.py --limit 1e11 --no-perm  # skip GPU permutation

Authors: Dreadbot 3.2.666 + Shadow
Date: 2026-02-28
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.special import expi
from scipy.stats import pearsonr, spearmanr

# ── GPU detection ────────────────────────────────────────────
try:
    import cupy as cp
    HAS_GPU = True
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
except ImportError:
    HAS_GPU = False
    print("No GPU — CPU mode")

# ── primesieve detection ─────────────────────────────────────
try:
    import primesieve
    HAS_PRIMESIEVE_PY = True
except ImportError:
    HAS_PRIMESIEVE_PY = False

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

def li(x):
    """Logarithmic integral Li(x) = Ei(ln(x))"""
    return expi(np.log(x))

KNOWN_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
])


# ═══════════════════════════════════════════════════════════════
# PRIME GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_primes(limit):
    """Generate primes using best available method."""
    t0 = time.time()

    if HAS_PRIMESIEVE_PY:
        primes = np.array(primesieve.primes(int(limit)), dtype=np.int64)
    else:
        # CLI fallback
        result = subprocess.run(
            ['primesieve', str(int(limit)), '-p'],
            capture_output=True, text=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"primesieve failed: {result.stderr.decode()}")
        primes = np.fromstring(result.stdout, dtype=np.int64, sep='\n')

    dt = time.time() - t0
    print(f"  {len(primes):,} primes up to {limit:.0e} in {dt:.1f}s")
    return primes


# ═══════════════════════════════════════════════════════════════
# FFT (GPU or CPU)
# ═══════════════════════════════════════════════════════════════

def compute_fft(gaps):
    """Compute FFT of centered gap sequence. Returns amplitudes array."""
    N = len(gaps)
    mean_gap = np.mean(gaps.astype(np.float64))
    centered = gaps.astype(np.float64) - mean_gap

    t0 = time.time()
    if HAS_GPU:
        centered_gpu = cp.asarray(centered)
        spectrum_gpu = cp.fft.rfft(centered_gpu)
        # Extract amplitudes on GPU, transfer only the result
        amp_gpu = cp.abs(spectrum_gpu[1:N//2]) / N
        amplitudes = cp.asnumpy(amp_gpu)
        # Keep spectrum for later use
        spectrum = cp.asnumpy(spectrum_gpu)
        del centered_gpu, spectrum_gpu, amp_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        spectrum = np.fft.rfft(centered)
        amplitudes = np.abs(spectrum[1:N//2]) / N

    dt = time.time() - t0
    print(f"  FFT of {N:,} points in {dt:.2f}s")
    return amplitudes, spectrum


# ═══════════════════════════════════════════════════════════════
# PER-ZERO HYPOTHESIS TEST
# ═══════════════════════════════════════════════════════════════

def per_zero_analysis(amplitudes, primes, gaps, n_zeros=30, window=100):
    """
    For each known zero γₙ:
      - Predict FFT bin via v2 mapping
      - Measure amplitude and compare to local neighborhood
      - Compute SNR, p-values
    """
    N = len(gaps)

    # v2 mapping constant: ⟨p/log(p)⟩ over the prime range
    # Use float64 carefully for large primes
    chunk_size = 10_000_000  # Process in chunks for memory
    total_p_over_logp = 0.0
    n_processed = 0
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        p_chunk = primes[start:end].astype(np.float64)
        total_p_over_logp += np.sum(p_chunk / np.log(p_chunk))
        n_processed += end - start
    mean_p_over_logp = total_p_over_logp / n_processed

    var_g = np.var(gaps.astype(np.float64))
    sigma_rayleigh = np.sqrt(var_g / (2 * N))
    ladder_spacing = 2 * np.pi * mean_p_over_logp / N

    print(f"  ⟨p/log p⟩ = {mean_p_over_logp:,.2f}")
    print(f"  Ladder spacing = {ladder_spacing:.4f}")
    print(f"  σ_Rayleigh = {sigma_rayleigh:.2e}")
    print(f"  var(gaps) = {var_g:.4f}")

    results = []

    for i in range(min(n_zeros, len(KNOWN_ZEROS))):
        gamma = KNOWN_ZEROS[i]

        # Predicted FFT bin (1-based in spectrum, 0-based in amplitudes array)
        k_pred = gamma * N / (2 * np.pi * mean_p_over_logp)
        k_int = int(round(k_pred))

        if k_int < 1 or k_int >= len(amplitudes):
            continue

        # Observed amplitude (amplitudes is 0-indexed, spectrum[1] = amplitudes[0])
        A_obs = amplitudes[k_int - 1]

        # ── Local noise estimation ──
        # Use ±window bins, excluding ±5 around target
        lo = max(0, k_int - 1 - window)
        hi = min(len(amplitudes), k_int - 1 + window + 1)
        mask = np.ones(hi - lo, dtype=bool)
        # Exclude center ±5
        center_in_window = k_int - 1 - lo
        excl_lo = max(0, center_in_window - 5)
        excl_hi = min(hi - lo, center_in_window + 6)
        mask[excl_lo:excl_hi] = False

        local_amps = amplitudes[lo:hi][mask]
        if len(local_amps) < 10:
            local_amps = amplitudes[lo:hi]

        local_median = np.median(local_amps)
        local_mad = np.median(np.abs(local_amps - local_median))
        sigma_local = local_mad * 1.4826  # MAD → σ

        snr = (A_obs - local_median) / sigma_local if sigma_local > 0 else 0.0

        # ── Analytical p-value (global Rayleigh null) ──
        # Under random permutation, |DFT(k)|/N ~ Rayleigh(σ_R)
        # P(A > a) = exp(-a²/(2σ²))
        p_rayleigh = float(np.exp(-(A_obs ** 2) / (2 * sigma_rayleigh ** 2)))

        # ── Local percentile p-value ──
        p_local = float(np.mean(local_amps >= A_obs))

        # ── Is this a local peak (within ±2)? ──
        is_peak = True
        for dk in [-2, -1, 1, 2]:
            kn = k_int - 1 + dk
            if 0 <= kn < len(amplitudes):
                if amplitudes[kn] > A_obs:
                    is_peak = False
                    break

        # ── Check neighbors: which of k-1, k, k+1 has max amplitude? ──
        best_k = k_int
        best_a = A_obs
        for dk in [-1, 1]:
            kn = k_int - 1 + dk
            if 0 <= kn < len(amplitudes):
                if amplitudes[kn] > best_a:
                    best_k = kn + 1  # back to 1-based
                    best_a = amplitudes[kn]
        gamma_best = 2 * np.pi * (best_k / N) * mean_p_over_logp

        results.append({
            'zero_idx': i + 1,
            'gamma_known': float(gamma),
            'k_pred': float(k_pred),
            'k_int': k_int,
            'A_obs': float(A_obs),
            'local_median': float(local_median),
            'sigma_local': float(sigma_local),
            'snr': float(snr),
            'p_rayleigh': float(p_rayleigh),
            'p_local': float(p_local),
            'is_local_peak': is_peak,
            'best_neighbor_k': best_k,
            'best_neighbor_A': float(best_a),
            'best_neighbor_gamma': float(gamma_best),
            'delta_gamma': float(gamma_best - gamma),
        })

    meta = {
        'mean_p_over_logp': float(mean_p_over_logp),
        'ladder_spacing': float(ladder_spacing),
        'var_g': float(var_g),
        'sigma_rayleigh': float(sigma_rayleigh),
        'N': N,
    }

    return results, meta


# ═══════════════════════════════════════════════════════════════
# GPU PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════

def gpu_permutation_test(gaps, target_bins, n_perms=1000):
    """
    GPU-accelerated permutation test.
    For each target FFT bin, count how often a random permutation
    of the gap sequence produces amplitude ≥ observed.
    """
    if not HAS_GPU:
        print("  No GPU — skipping permutation test")
        return {}

    N = len(gaps)
    mean_gap = float(np.mean(gaps.astype(np.float64)))

    # Observed amplitudes at target bins
    centered = gaps.astype(np.float64) - mean_gap
    spectrum = np.fft.rfft(centered)
    obs_amps = {}
    for k in target_bins:
        obs_amps[k] = np.abs(spectrum[k]) / N

    print(f"  Permutation test: {n_perms} shuffles, {len(target_bins)} bins")
    print(f"  Target bins: {target_bins}")

    # Move centered gaps to GPU
    centered_gpu = cp.asarray(centered)
    perm_counts = {k: 0 for k in target_bins}

    t0 = time.time()
    for i in range(n_perms):
        # Shuffle on GPU (in-place permutation)
        perm_idx = cp.random.permutation(N)
        shuffled = centered_gpu[perm_idx]

        # FFT on GPU
        perm_spectrum = cp.fft.rfft(shuffled)

        # Check each target bin
        for k in target_bins:
            perm_amp = float(cp.abs(perm_spectrum[k]) / N)
            if perm_amp >= obs_amps[k]:
                perm_counts[k] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            print(f"    [{i+1}/{n_perms}] {rate:.0f}/s, ETA {eta:.0f}s")

    del centered_gpu
    cp.get_default_memory_pool().free_all_blocks()

    dt = time.time() - t0
    print(f"  {n_perms} permutations in {dt:.1f}s ({n_perms/dt:.0f}/s)")

    # Compute p-values
    p_values = {}
    for k in target_bins:
        p_values[k] = (perm_counts[k] + 1) / (n_perms + 1)

    return p_values


# ═══════════════════════════════════════════════════════════════
# V2 LADDER TEST (COMPARISON WITH 10⁸ BASELINE)
# ═══════════════════════════════════════════════════════════════

def v2_ladder_test(amplitudes, primes, gaps, n_modes=30, n_mc=100_000):
    """
    Reproduce the v2 ladder test for comparison with the 10⁸ result.
    Maps top FFT modes to γ via v2 mapping, counts hits within ±0.5.
    """
    N = len(gaps)

    # Mapping constant (same as per-zero)
    chunk_size = 10_000_000
    total = 0.0
    n_proc = 0
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        p = primes[start:end].astype(np.float64)
        total += np.sum(p / np.log(p))
        n_proc += end - start
    mean_p_over_logp = total / n_proc

    # Top modes by amplitude
    top_indices = np.argsort(amplitudes)[::-1][:n_modes]
    gammas = []
    for idx in top_indices:
        k = idx + 1  # 1-based
        f = k / N
        gamma = 2 * np.pi * f * mean_p_over_logp
        gammas.append((gamma, float(amplitudes[idx])))

    # Count hits against first 15 zeros
    test_zeros = KNOWN_ZEROS[:15]
    hits_05 = 0
    hits_03 = 0
    hit_details = []
    for z in test_zeros:
        for g, a in gammas:
            if abs(g - z) < 0.5:
                hits_05 += 1
                hit_details.append({'known': float(z), 'predicted': float(g),
                                    'delta': float(g - z), 'amplitude': a})
                if abs(g - z) < 0.3:
                    hits_03 += 1
                break

    # Monte Carlo significance (random ladder with same spacing)
    g_sorted = sorted([g for g, _ in gammas if g > 5])
    if len(g_sorted) > 2:
        spacing = np.mean(np.diff(g_sorted[:15]))
    else:
        spacing = 2 * np.pi * mean_p_over_logp / N

    hit_dist = np.zeros(n_mc)
    for trial in range(n_mc):
        phase = np.random.uniform(0, spacing)
        rungs = np.arange(phase + 10, 110, spacing)
        hits = sum(1 for z in test_zeros if np.any(np.abs(rungs - z) < 0.5))
        hit_dist[trial] = hits

    p_value = float(np.mean(hit_dist >= hits_05))
    mean_random = float(np.mean(hit_dist))
    std_random = float(np.std(hit_dist))

    return {
        'n_modes': n_modes,
        'ladder_spacing': float(spacing),
        'hits_05': hits_05,
        'hits_03': hits_03,
        'p_value': p_value,
        'mean_random': mean_random,
        'std_random': std_random,
        'sigma_above': float((hits_05 - mean_random) / std_random) if std_random > 0 else 0,
        'hit_details': hit_details,
    }


# ═══════════════════════════════════════════════════════════════
# AMPLITUDE DECAY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def amplitude_decay_analysis(results):
    """
    Test whether amplitudes at predicted bins follow the expected
    1/|ρ| = 1/√(1/4 + γ²) decay from the explicit formula.
    """
    gammas = np.array([r['gamma_known'] for r in results])
    amps = np.array([r['A_obs'] for r in results])

    # Expected weighting from explicit formula
    expected = 1.0 / np.sqrt(0.25 + gammas ** 2)
    # Normalize to match observed scale
    scale = amps[0] / expected[0] if expected[0] > 0 else 1.0
    expected_scaled = expected * scale

    # Correlations
    r_p, p_p = pearsonr(amps, expected_scaled)
    r_s, p_s = spearmanr(amps, expected)

    # Ratio analysis: A_obs / expected (should be ~constant if model correct)
    ratios = amps / expected_scaled
    ratio_mean = float(np.mean(ratios))
    ratio_std = float(np.std(ratios))

    return {
        'pearson_r': float(r_p),
        'pearson_p': float(p_p),
        'spearman_r': float(r_s),
        'spearman_p': float(p_s),
        'ratio_mean': ratio_mean,
        'ratio_std': ratio_std,
        'scale_factor': float(scale),
        'observed': amps.tolist(),
        'expected': expected_scaled.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# SPLIT-HALF VALIDATION
# ═══════════════════════════════════════════════════════════════

def split_half_validation(primes, n_zeros=15):
    """
    Split prime sequence in half, run per-zero test on each half independently.
    If signal is real, both halves should show similar pattern.
    """
    N_total = len(primes)
    mid = N_total // 2

    results_halves = []
    for label, p_slice in [("first_half", primes[:mid+1]), ("second_half", primes[mid:])]:
        gaps = np.diff(p_slice)
        N = len(gaps)
        if N < 100:
            continue

        # FFT
        centered = gaps.astype(np.float64) - np.mean(gaps)
        if HAS_GPU:
            c_gpu = cp.asarray(centered)
            s_gpu = cp.fft.rfft(c_gpu)
            amplitudes = cp.asnumpy(cp.abs(s_gpu[1:N//2]) / N)
            del c_gpu, s_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            spectrum = np.fft.rfft(centered)
            amplitudes = np.abs(spectrum[1:N//2]) / N

        # Mapping constant
        log_p = np.log(p_slice[:N].astype(np.float64))
        mean_pol = float(np.mean(p_slice[:N].astype(np.float64) / log_p))

        hits = 0
        snrs = []
        for i in range(min(n_zeros, len(KNOWN_ZEROS))):
            gamma = KNOWN_ZEROS[i]
            k = int(round(gamma * N / (2 * np.pi * mean_pol)))
            if k < 1 or k >= len(amplitudes):
                continue
            A = amplitudes[k - 1]

            # Local noise
            lo = max(0, k - 51)
            hi = min(len(amplitudes), k + 50)
            local = np.concatenate([amplitudes[lo:max(0,k-6)], amplitudes[min(len(amplitudes),k+5):hi]])
            if len(local) < 5:
                local = amplitudes[lo:hi]
            med = np.median(local)
            mad = np.median(np.abs(local - med)) * 1.4826
            snr = (A - med) / mad if mad > 0 else 0
            snrs.append(snr)
            if snr > 2:
                hits += 1

        results_halves.append({
            'label': label,
            'n_primes': len(p_slice),
            'n_gaps': N,
            'hits_snr2': hits,
            'mean_snr': float(np.mean(snrs)) if snrs else 0,
            'snrs': [float(s) for s in snrs],
        })

    return results_halves


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Per-Zero Hypothesis Test (GPU)")
    parser.add_argument('--limit', type=float, default=1e9, help="Prime sieve limit")
    parser.add_argument('--n-zeros', type=int, default=30, help="Number of zeros to test")
    parser.add_argument('--n-perms', type=int, default=1000, help="Permutation test iterations")
    parser.add_argument('--window', type=int, default=100, help="Local noise window (bins)")
    parser.add_argument('--no-perm', action='store_true', help="Skip permutation test")
    parser.add_argument('--no-split', action='store_true', help="Skip split-half validation")
    parser.add_argument('--outdir', type=str, default='results', help="Output directory")
    args = parser.parse_args()

    limit_str = f"{args.limit:.0e}".replace('+', '')

    print("=" * 60)
    print(f"  PER-ZERO HYPOTHESIS TEST — {limit_str}")
    print("=" * 60)
    print(f"  GPU: {HAS_GPU}")
    print(f"  primesieve Python: {HAS_PRIMESIEVE_PY}")
    print(f"  Zeros: {args.n_zeros}")
    print(f"  Permutations: {0 if args.no_perm else args.n_perms}")
    print()

    # ── Stage 1: Generate primes ─────────────────────────────
    print("▸ Stage 1: Prime generation")
    t_total = time.time()
    primes = generate_primes(args.limit)
    gaps = np.diff(primes)
    N = len(gaps)
    print(f"  {N:,} gaps, mean={np.mean(gaps):.4f}, max={np.max(gaps)}")
    mem_gb = (primes.nbytes + gaps.nbytes) / 1e9
    print(f"  Memory: {mem_gb:.2f} GB (primes + gaps)")
    print()

    # ── Stage 2: FFT ─────────────────────────────────────────
    print("▸ Stage 2: FFT")
    amplitudes, spectrum = compute_fft(gaps)
    print()

    # ── Stage 3: Per-zero hypothesis test ────────────────────
    print("▸ Stage 3: Per-zero hypothesis test")
    results, meta = per_zero_analysis(amplitudes, primes, gaps,
                                       n_zeros=args.n_zeros, window=args.window)

    bonferroni = 0.05 / len(results)

    print()
    print(f"  {'Zero':>6} {'γ':>8} {'k':>7} {'A_obs':>11} {'Median':>11} "
          f"{'SNR':>7} {'p(Rayl)':>10} {'p(loc)':>8} {'Pk':>3} {'Δγ':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*7} {'─'*11} {'─'*11} "
          f"{'─'*7} {'─'*10} {'─'*8} {'─'*3} {'─'*8}")

    n_sig_rayleigh = 0
    n_sig_local = 0
    n_high_snr = 0
    top_candidates = []  # bins for permutation test

    for r in results:
        sig = ''
        if r['p_rayleigh'] < bonferroni:
            sig = '★'
            n_sig_rayleigh += 1
        elif r['snr'] > 3:
            sig = '⚡'
        elif r['p_local'] < 0.05:
            sig = '·'

        if r['snr'] > 3:
            n_high_snr += 1
        if r['p_local'] < bonferroni:
            n_sig_local += 1

        pk = '▲' if r['is_local_peak'] else ''

        print(f"  γ{r['zero_idx']:<4} {r['gamma_known']:>7.3f} {r['k_int']:>7} "
              f"{r['A_obs']:>11.8f} {r['local_median']:>11.8f} "
              f"{r['snr']:>7.2f} {r['p_rayleigh']:>10.2e} "
              f"{r['p_local']:>8.4f} {pk:>3} {r['delta_gamma']:>+8.4f} {sig}")

        if r['snr'] > 1.5 or r['p_local'] < 0.15:
            top_candidates.append(r['k_int'])

    print()
    print(f"  Bonferroni threshold: p < {bonferroni:.6f}")
    print(f"  Significant (Rayleigh+Bonf):  {n_sig_rayleigh}/{len(results)}")
    print(f"  Significant (Local+Bonf):     {n_sig_local}/{len(results)}")
    print(f"  High SNR (>3σ):               {n_high_snr}/{len(results)}")
    print()

    # ── Stage 4: Amplitude decay ─────────────────────────────
    print("▸ Stage 4: Amplitude decay (A vs 1/|ρ|)")
    decay = amplitude_decay_analysis(results)
    print(f"  Pearson r(A, 1/|ρ|):  {decay['pearson_r']:.4f} (p={decay['pearson_p']:.2e})")
    print(f"  Spearman ρ(A, 1/γ):   {decay['spearman_r']:.4f} (p={decay['spearman_p']:.2e})")
    print(f"  Ratio A/expected:      {decay['ratio_mean']:.4f} ± {decay['ratio_std']:.4f}")
    print()

    # ── Stage 5: V2 ladder comparison ────────────────────────
    print("▸ Stage 5: V2 ladder test (comparison with 10⁸ baseline)")
    ladder = v2_ladder_test(amplitudes, primes, gaps)
    print(f"  Hits ±0.5: {ladder['hits_05']}/15")
    print(f"  Hits ±0.3: {ladder['hits_03']}/15")
    print(f"  Ladder spacing: {ladder['ladder_spacing']:.4f}")
    print(f"  Monte Carlo: p = {ladder['p_value']:.6f} "
          f"(mean random = {ladder['mean_random']:.2f} ± {ladder['std_random']:.2f})")
    print(f"  Sigma above mean: {ladder['sigma_above']:.2f}σ")
    print(f"  SIGNIFICANT: {'YES' if ladder['p_value'] < 0.05 else 'NO'}")
    print()

    # ── Stage 6: GPU permutation test ────────────────────────
    perm_results = {}
    if not args.no_perm and HAS_GPU and top_candidates:
        print(f"▸ Stage 6: GPU permutation test ({len(top_candidates[:10])} bins, {args.n_perms} perms)")
        perm_pvals = gpu_permutation_test(gaps, top_candidates[:10], n_perms=args.n_perms)

        print(f"\n  Permutation p-values:")
        for k, p in sorted(perm_pvals.items()):
            for r in results:
                if r['k_int'] == k:
                    sig = '★' if p < bonferroni else ('·' if p < 0.05 else '')
                    print(f"    γ{r['zero_idx']} (k={k}): p_perm = {p:.4f} {sig}")
                    r['p_permutation'] = p
                    break
        perm_results = {str(k): float(v) for k, v in perm_pvals.items()}
        print()
    elif args.no_perm:
        print("▸ Stage 6: Permutation test SKIPPED (--no-perm)")
        print()

    # ── Stage 7: Split-half validation ───────────────────────
    split_results = []
    if not args.no_split:
        print("▸ Stage 7: Split-half validation")
        split_results = split_half_validation(primes, n_zeros=min(15, args.n_zeros))
        for sh in split_results:
            print(f"  {sh['label']}: {sh['n_gaps']:,} gaps, "
                  f"hits(SNR>2)={sh['hits_snr2']}/15, mean SNR={sh['mean_snr']:.2f}")
        # Correlation between halves
        if len(split_results) == 2:
            s1 = np.array(split_results[0]['snrs'])
            s2 = np.array(split_results[1]['snrs'])
            n_common = min(len(s1), len(s2))
            if n_common > 3:
                r_split, p_split = pearsonr(s1[:n_common], s2[:n_common])
                print(f"  Half-half SNR correlation: r={r_split:.3f}, p={p_split:.4f}")
                print(f"  Consistent signal: {'YES' if r_split > 0.3 and p_split < 0.05 else 'NO'}")
        print()

    # ═══ SUMMARY ═════════════════════════════════════════════
    dt_total = time.time() - t_total

    print("=" * 60)
    print(f"  SUMMARY — {limit_str}")
    print("=" * 60)
    print(f"  Scale: {N:,} gaps")
    print(f"  Significant zeros (Rayleigh+Bonf): {n_sig_rayleigh}/{len(results)}")
    print(f"  High-SNR zeros (>3σ):              {n_high_snr}/{len(results)}")
    print(f"  Ladder test: {ladder['hits_05']}/15 hits, p={ladder['p_value']:.4f}")
    print(f"  Amplitude decay: r={decay['pearson_r']:.3f} (p={decay['pearson_p']:.2e})")
    if results:
        best_snr_idx = int(np.argmax([r['snr'] for r in results]))
        print(f"  Best SNR: γ{results[best_snr_idx]['zero_idx']} "
              f"(SNR={results[best_snr_idx]['snr']:.2f})")
    print(f"  Total time: {dt_total:.1f}s")
    print()

    # ── Save ─────────────────────────────────────────────────
    output = {
        'params': {
            'limit': args.limit,
            'limit_str': limit_str,
            'n_primes': int(len(primes)),
            'N_gaps': int(N),
            'n_zeros_tested': len(results),
            'window': args.window,
            'n_perms': 0 if args.no_perm else args.n_perms,
            'gpu': HAS_GPU,
            'total_time_s': dt_total,
        },
        'meta': meta,
        'bonferroni_threshold': float(bonferroni),
        'summary': {
            'n_sig_rayleigh': n_sig_rayleigh,
            'n_sig_local': n_sig_local,
            'n_high_snr': n_high_snr,
        },
        'ladder_test': ladder,
        'amplitude_decay': decay,
        'split_half': split_results,
        'permutation_pvalues': perm_results,
        'zeros': results,
    }

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f'per_zero_{limit_str}.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {outpath}")

    # Also save a compact human-readable summary
    summary_path = os.path.join(args.outdir, f'summary_{limit_str}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Per-Zero Hypothesis Test — {limit_str}\n")
        f.write(f"{'='*50}\n")
        f.write(f"N = {N:,} gaps\n")
        f.write(f"Significant (Rayleigh+Bonf): {n_sig_rayleigh}/{len(results)}\n")
        f.write(f"High SNR (>3σ): {n_high_snr}/{len(results)}\n")
        f.write(f"Ladder hits ±0.5: {ladder['hits_05']}/15 (p={ladder['p_value']:.4f})\n")
        f.write(f"Amplitude decay: r={decay['pearson_r']:.3f}\n")
        f.write(f"\nPer-zero results:\n")
        for r in results:
            f.write(f"  γ{r['zero_idx']}: k={r['k_int']}, SNR={r['snr']:.2f}, "
                    f"p_R={r['p_rayleigh']:.2e}, p_L={r['p_local']:.4f}, "
                    f"Δγ={r['delta_gamma']:+.4f}\n")
    print(f"  Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
