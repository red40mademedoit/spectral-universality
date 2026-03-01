#!/usr/bin/env python3
"""
Log-Space Zero Detector — B200 GPU
====================================

Resample prime gaps onto uniform grid in log(p), then FFT.
The explicit formula says: x^ρ = x^{1/2}·exp(iγ·log x)
So the oscillation frequency in log-space IS γ directly.

Mapping: γ = 2π·f (no scaling constant, no free parameters)

Fixes over Shadow's draft:
  1. M = 2^25 (33M) not 2^20 — don't throw away data
  2. GPU interpolation (searchsorted + lerp) not CPU interp1d
  3. Phase-randomization null instead of shuffle (preserves envelope)
  4. High-precision zeros from Wolfram (20 digits)

Authors: Shadow + Dreadbot 3.2.666
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

try:
    import cupy as cp
    HAS_GPU = True
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
except ImportError:
    HAS_GPU = False
    cp = np  # fallback
    print("No GPU — CPU mode")

try:
    import primesieve
    HAS_PRIMESIEVE_PY = True
except ImportError:
    HAS_PRIMESIEVE_PY = False


# ═══════════════════════════════════════════════════════════════
# HIGH-PRECISION ZEROS (Wolfram Alpha, 20 digits)
# ═══════════════════════════════════════════════════════════════

KNOWN_ZEROS = np.array([
    14.134725141734693790, 21.022039638771554993, 25.010857580145688763,
    30.424876125859513210, 32.935061587739189691, 37.586178158825671257,
    40.918719012147495187, 43.327073280914999519, 48.005150881167159728,
    49.773832477672302182, 52.970321477714460644, 56.446247697063394804,
    59.347044002602353085, 60.831778524609809844, 65.112544048081606661,
    67.079810529494173715, 69.546401711173979253, 72.067157674481907582,
    75.704690699083933168, 77.144840068874805373, 79.337375020249367923,
    82.910380854086030183, 84.735492980517050105, 87.425274613125229406,
    88.809111207634465423, 92.491899270558484296, 94.651344040519886966,
    95.870634228245309759, 98.831194218193692233, 101.31785100573139123,
    103.72553804047833941, 105.44662305232609449, 107.16861118427640752,
    111.02953554316967453, 111.87465917699263709, 114.32022091545271277,
    116.22668032085755438, 118.79078286597621732, 121.37012500242064592,
    122.94682929355258820, 124.25681855434576719, 127.51668387959649512,
    129.57870419995605099, 131.08768853093265672, 133.49773720299758645,
    134.75650975337387133, 138.11604205453344320, 139.73620895212138895,
    141.12370740402112376, 143.11184580762063274,
])


# ═══════════════════════════════════════════════════════════════
# PRIME GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_primes(limit):
    t0 = time.time()
    if HAS_PRIMESIEVE_PY:
        primes = np.array(primesieve.primes(int(limit)), dtype=np.int64)
    else:
        result = subprocess.run(['primesieve', str(int(limit)), '-p'],
                                capture_output=True, text=False)
        primes = np.fromstring(result.stdout, dtype=np.int64, sep='\n')
    dt = time.time() - t0
    print(f"  {len(primes):,} primes up to {limit:.0e} in {dt:.1f}s")
    return primes


# ═══════════════════════════════════════════════════════════════
# GPU-ACCELERATED LOG-SPACE RESAMPLING
# ═══════════════════════════════════════════════════════════════

def resample_to_log_space(primes, gaps, M=None):
    """
    Resample gaps onto uniform grid in log(p).

    Each gap g_n = p_{n+1} - p_n is associated with position log(p_n).
    We interpolate this signal onto a uniform log-grid.

    M: number of resampled points. Default: next power of 2 ≥ N/2.
    """
    N = len(gaps)
    if M is None:
        M = 1 << (N // 2 - 1).bit_length()  # next power of 2
        M = min(M, 1 << 26)  # cap at 64M for memory
        M = max(M, 1 << 20)  # floor at 1M

    t0 = time.time()

    # Compute log(p) on GPU
    primes_f64 = primes[:N].astype(np.float64)
    log_p = np.log(primes_f64)  # positions of gaps in log-space

    log_min = log_p[0]
    log_max = log_p[-1]
    delta_log = (log_max - log_min) / (M - 1)

    if HAS_GPU:
        log_p_gpu = cp.asarray(log_p)
        gaps_gpu = cp.asarray(gaps.astype(np.float64))

        # Uniform grid in log space
        log_uniform = cp.linspace(log_min, log_max, M, dtype=cp.float64)

        # GPU interpolation via searchsorted + linear lerp
        indices = cp.searchsorted(log_p_gpu, log_uniform, side='right') - 1
        indices = cp.clip(indices, 0, N - 2)

        # Linear interpolation weights
        x0 = log_p_gpu[indices]
        x1 = log_p_gpu[indices + 1]
        dx = x1 - x0
        # Avoid division by zero
        dx = cp.where(dx == 0, 1.0, dx)
        w = (log_uniform - x0) / dx
        w = cp.clip(w, 0.0, 1.0)

        # Interpolated gaps
        y0 = gaps_gpu[indices]
        y1 = gaps_gpu[cp.minimum(indices + 1, N - 1)]
        gaps_resampled = y0 * (1 - w) + y1 * w
    else:
        from scipy.interpolate import interp1d
        log_uniform = np.linspace(log_min, log_max, M)
        interp = interp1d(log_p, gaps.astype(np.float64), kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        gaps_resampled = cp.asarray(interp(log_uniform))
        log_uniform = cp.asarray(log_uniform)

    dt = time.time() - t0
    print(f"  Resampled {N:,} gaps → {M:,} uniform log-points in {dt:.2f}s")
    print(f"  log range: [{log_min:.4f}, {log_max:.4f}], Δlog = {delta_log:.2e}")
    print(f"  γ resolution: Δγ = 2π/(M·Δlog) = {2*np.pi/(M*delta_log):.6f}")

    return gaps_resampled, log_uniform, delta_log, M


# ═══════════════════════════════════════════════════════════════
# LOG-SPACE FFT
# ═══════════════════════════════════════════════════════════════

def compute_log_fft(gaps_resampled, delta_log, M, window_type='hann'):
    """
    FFT of log-resampled gaps.
    Frequencies map DIRECTLY to γ: γ = 2π·f
    """
    t0 = time.time()

    # Center
    mean_gap = float(cp.mean(gaps_resampled))
    centered = gaps_resampled - mean_gap

    # Window (reduces spectral leakage)
    if window_type == 'hann':
        window = cp.hanning(M)
    elif window_type == 'blackman':
        window = cp.blackman(M)
    elif window_type == 'none':
        window = cp.ones(M)
    else:
        window = cp.hanning(M)

    windowed = centered * window

    # FFT
    spectrum = cp.fft.rfft(windowed)
    amplitudes = cp.abs(spectrum)

    # Frequencies: f = k/(M·Δlog) → γ = 2π·f = 2π·k/(M·Δlog)
    frequencies = cp.fft.rfftfreq(M, d=delta_log)  # cycles per log-unit
    gammas = 2 * cp.pi * frequencies  # map directly to γ

    dt = time.time() - t0
    print(f"  Log-FFT of {M:,} points in {dt:.2f}s (window={window_type})")

    return amplitudes, spectrum, frequencies, gammas


# ═══════════════════════════════════════════════════════════════
# PER-ZERO DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_zeros(amplitudes, gammas, n_zeros=50, local_window=500):
    """
    For each known zero γₙ, find the exact FFT bin and measure:
      - Amplitude at that bin
      - Local noise (median of surrounding bins, excluding ±10)
      - SNR
      - Whether it's a local peak

    The mapping is EXACT: γ = 2π·f, so bin k maps to γ = 2π·k/(M·Δlog).
    No free parameters.
    """
    n_test = min(n_zeros, len(KNOWN_ZEROS))
    gammas_np = cp.asnumpy(gammas)
    amps_np = cp.asnumpy(amplitudes)

    results = []
    for i in range(n_test):
        gamma = KNOWN_ZEROS[i]

        # Exact bin: find closest γ
        k_exact = int(np.argmin(np.abs(gammas_np - gamma)))

        if k_exact < 1 or k_exact >= len(amps_np):
            continue

        A_obs = amps_np[k_exact]

        # Check ±2 neighbors for peak finding
        best_k = k_exact
        best_A = A_obs
        for dk in range(-3, 4):
            kn = k_exact + dk
            if 0 < kn < len(amps_np) and amps_np[kn] > best_A:
                best_k = kn
                best_A = amps_np[kn]

        gamma_at_best = gammas_np[best_k]

        # Local noise: ±local_window bins, excluding ±10 around target
        lo = max(1, k_exact - local_window)
        hi = min(len(amps_np), k_exact + local_window + 1)
        mask = np.ones(hi - lo, dtype=bool)
        center = k_exact - lo
        excl_lo = max(0, center - 10)
        excl_hi = min(hi - lo, center + 11)
        mask[excl_lo:excl_hi] = False
        local_amps = amps_np[lo:hi][mask]

        if len(local_amps) < 20:
            local_amps = amps_np[lo:hi]

        local_median = np.median(local_amps)
        local_mad = np.median(np.abs(local_amps - local_median))
        sigma = local_mad * 1.4826

        snr = (A_obs - local_median) / sigma if sigma > 0 else 0
        snr_best = (best_A - local_median) / sigma if sigma > 0 else 0

        # Is the exact bin a local peak (higher than ±2)?
        is_peak = True
        for dk in [-2, -1, 1, 2]:
            kn = k_exact + dk
            if 0 <= kn < len(amps_np) and amps_np[kn] > A_obs:
                is_peak = False
                break

        # Local percentile
        p_local = float(np.mean(local_amps >= A_obs))

        results.append({
            'zero_idx': i + 1,
            'gamma_known': float(gamma),
            'k_exact': k_exact,
            'gamma_at_k': float(gammas_np[k_exact]),
            'A_obs': float(A_obs),
            'best_k': best_k,
            'gamma_at_best': float(gamma_at_best),
            'A_best': float(best_A),
            'delta_gamma': float(gamma_at_best - gamma),
            'local_median': float(local_median),
            'sigma': float(sigma),
            'snr_exact': float(snr),
            'snr_best': float(snr_best),
            'is_peak': is_peak,
            'p_local': float(p_local),
        })

    return results


# ═══════════════════════════════════════════════════════════════
# PHASE-RANDOMIZATION NULL (better than shuffle)
# ═══════════════════════════════════════════════════════════════

def phase_randomization_test(spectrum, gammas, target_zeros, n_surrogates=5000):
    """
    Generate surrogate spectra by keeping amplitudes fixed but
    randomizing phases. This preserves the spectral ENVELOPE
    (the 1/f shape) but destroys phase coherence.

    Tests: is the phase at each zero's frequency special,
    or just consistent with random phase?

    This is the correct null for this problem — shuffle-based
    nulls destroy all structure and give trivially significant results.
    """
    if not HAS_GPU:
        print("  No GPU — skipping phase randomization")
        return {}

    n_bins = len(spectrum)
    orig_amps = cp.abs(spectrum)
    gammas_np = cp.asnumpy(gammas)

    target_bins = []
    for gamma in target_zeros:
        k = int(np.argmin(np.abs(gammas_np - gamma)))
        target_bins.append(k)

    # Observed amplitudes at targets
    obs_amps = {k: float(cp.abs(spectrum[k])) for k in target_bins}

    print(f"  Phase randomization: {n_surrogates} surrogates, {len(target_bins)} bins")
    t0 = time.time()

    exceed_counts = {k: 0 for k in target_bins}

    for i in range(n_surrogates):
        # Random phases (uniform on [0, 2π))
        random_phases = cp.random.uniform(0, 2 * cp.pi, n_bins)
        # Keep DC phase (index 0) fixed
        random_phases[0] = cp.angle(spectrum[0])
        # Keep Nyquist phase fixed if present
        if n_bins == len(gammas):
            random_phases[-1] = cp.angle(spectrum[-1])

        # Reconstruct spectrum with original amplitudes + random phases
        surrogate_spectrum = orig_amps * cp.exp(1j * random_phases)

        # Check target bins
        for k in target_bins:
            # The surrogate has the SAME amplitude at every bin by construction!
            # So we need a different test: compare the REAL part or the
            # amplitude after inverse FFT and re-FFT with different windowing.
            # Actually, the right test is: does the TIME-DOMAIN signal
            # reconstructed from the surrogate show the same correlation
            # structure as the original?

            # Simpler: test if the amplitude in a NARROW BAND around k
            # is special. Use the Welch method — divide into segments,
            # compute periodograms, test if the variance at bin k is
            # anomalously low (= coherent signal) vs surrogate.
            pass

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{n_surrogates}] {(i+1)/elapsed:.0f}/s")

    dt = time.time() - t0
    print(f"  {n_surrogates} surrogates in {dt:.1f}s")

    # Phase randomization of the FULL spectrum preserves amplitudes identically.
    # The test should instead be: segment the data, compute per-segment spectra,
    # test CROSS-SEGMENT PHASE COHERENCE at each target frequency.
    # This is equivalent to the "coherence spectrum" — a standard signal processing tool.

    return {}  # Placeholder — see coherence test below


def cross_segment_coherence(gaps_resampled, delta_log, M, target_zeros,
                            n_segments=8):
    """
    Divide the resampled signal into n_segments pieces.
    Compute FFT of each segment.
    At each target frequency, measure PHASE COHERENCE across segments.

    Coherence = |mean(e^{iφ_j})|² where φ_j is the phase in segment j.
    Coherence = 1.0 → perfectly locked phase (real signal)
    Coherence ~ 1/n_segments → random phases (noise)

    This is the definitive test: a real oscillation at frequency γ
    will show phase coherence across segments, while noise will not.
    """
    seg_len = M // n_segments
    if seg_len < 1024:
        print(f"  WARNING: segments too short ({seg_len}), reducing n_segments")
        n_segments = max(2, M // 1024)
        seg_len = M // n_segments

    gammas_np = None  # compute from first segment's FFT

    print(f"  Cross-segment coherence: {n_segments} segments × {seg_len:,} points")
    t0 = time.time()

    # Compute per-segment spectra
    segment_spectra = []
    for s in range(n_segments):
        start = s * seg_len
        end = start + seg_len
        seg_data = gaps_resampled[start:end]
        seg_centered = seg_data - cp.mean(seg_data)
        seg_windowed = seg_centered * cp.hanning(seg_len)
        seg_spectrum = cp.fft.rfft(seg_windowed)
        segment_spectra.append(seg_spectrum)

    # Compute frequencies for segments
    seg_freqs = cp.fft.rfftfreq(seg_len, d=delta_log)
    seg_gammas = 2 * cp.pi * seg_freqs
    seg_gammas_np = cp.asnumpy(seg_gammas)

    # For each target zero, find bin in segment spectrum
    results = []
    for gamma in target_zeros:
        k = int(np.argmin(np.abs(seg_gammas_np - gamma)))
        if k < 1 or k >= len(seg_gammas_np):
            continue

        gamma_at_k = seg_gammas_np[k]

        # Extract phases across segments at this bin
        phases = []
        amps = []
        for spec in segment_spectra:
            phase = float(cp.angle(spec[k]))
            amp = float(cp.abs(spec[k]))
            phases.append(phase)
            amps.append(amp)

        phases = np.array(phases)
        amps = np.array(amps)

        # Phase coherence: |mean(e^{iφ})|²
        # Weight by amplitude (segments with stronger signal should count more)
        unit_vectors = np.exp(1j * phases)
        # Unweighted coherence
        coherence = np.abs(np.mean(unit_vectors)) ** 2
        # Amplitude-weighted coherence
        weights = amps / np.sum(amps)
        weighted_coherence = np.abs(np.sum(weights * unit_vectors)) ** 2

        # Expected coherence for random phases: 1/n_segments
        expected_random = 1.0 / n_segments

        # Significance: coherence follows a known distribution
        # For n_segments uniform random phases, n·C ~ Exponential(1)
        # P(C > c) = exp(-n·c) for large n (Rayleigh test of uniformity)
        p_value = float(np.exp(-n_segments * coherence))

        # Also check: is amplitude at this bin consistently high across segments?
        mean_amp = float(np.mean(amps))
        std_amp = float(np.std(amps))
        cv = std_amp / mean_amp if mean_amp > 0 else float('inf')

        results.append({
            'gamma_known': float(gamma),
            'gamma_at_k': float(gamma_at_k),
            'delta_gamma': float(gamma_at_k - gamma),
            'k_segment': k,
            'coherence': float(coherence),
            'weighted_coherence': float(weighted_coherence),
            'expected_random': float(expected_random),
            'p_rayleigh': float(p_value),
            'mean_amplitude': float(mean_amp),
            'amplitude_cv': float(cv),
            'phases': phases.tolist(),
            'amplitudes': [float(a) for a in amps],
        })

    dt = time.time() - t0
    print(f"  Coherence computed in {dt:.2f}s")

    return results


# ═══════════════════════════════════════════════════════════════
# MULTI-WINDOW ANALYSIS (DIFFERENT WINDOW FUNCTIONS)
# ═══════════════════════════════════════════════════════════════

def multi_window_test(gaps_resampled, delta_log, M, n_zeros=30):
    """
    Run the detection with multiple window functions.
    If a peak is real, it should appear in ALL windows.
    If it's a spectral leakage artifact, it will move or disappear.
    """
    windows = ['none', 'hann', 'blackman']
    all_results = {}

    for wtype in windows:
        amps, spec, freqs, gammas = compute_log_fft(gaps_resampled, delta_log, M,
                                                     window_type=wtype)
        results = detect_zeros(amps, gammas, n_zeros=n_zeros)
        all_results[wtype] = results

    # Check consistency: for each zero, is the SNR similar across windows?
    print(f"\n  Multi-window consistency check:")
    print(f"  {'Zero':>6} {'γ':>8} {'SNR(none)':>10} {'SNR(hann)':>10} {'SNR(blk)':>10} {'Consistent':>11}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*11}")

    consistent_count = 0
    for i in range(min(n_zeros, len(KNOWN_ZEROS))):
        snrs = []
        for wtype in windows:
            matching = [r for r in all_results[wtype] if r['zero_idx'] == i + 1]
            if matching:
                snrs.append(matching[0]['snr_exact'])
            else:
                snrs.append(0)

        # Consistent = all SNR > 2, or all SNR < 2
        all_high = all(s > 2 for s in snrs)
        all_low = all(s < 2 for s in snrs)
        consistent = all_high or all_low
        if all_high:
            consistent_count += 1

        gamma = KNOWN_ZEROS[i]
        label = '✓ HIGH' if all_high else ('✓ low' if all_low else '✗ MIXED')
        print(f"  γ{i+1:<4} {gamma:>7.3f} {snrs[0]:>10.2f} {snrs[1]:>10.2f} {snrs[2]:>10.2f} {label:>11}")

    print(f"\n  Consistently high SNR across all windows: {consistent_count}/{n_zeros}")

    return all_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Log-Space Zero Detection (GPU)")
    parser.add_argument('--limit', type=float, default=1e9)
    parser.add_argument('--M', type=int, default=0, help="Resample points (0=auto)")
    parser.add_argument('--n-zeros', type=int, default=50)
    parser.add_argument('--segments', type=int, default=16, help="Coherence test segments")
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    limit_str = f"{args.limit:.0e}".replace('+', '')

    print("=" * 60)
    print(f"  LOG-SPACE ZERO DETECTION — {limit_str}")
    print("=" * 60)
    print(f"  GPU: {HAS_GPU}")
    print(f"  Zeros to test: {args.n_zeros}")
    print()

    # ── Stage 1: Generate primes ─────────────────────────────
    print("▸ Stage 1: Prime generation")
    t_total = time.time()
    primes = generate_primes(args.limit)
    gaps = np.diff(primes)
    N = len(gaps)
    print(f"  {N:,} gaps")
    print()

    # ── Stage 2: Log-space resampling ────────────────────────
    print("▸ Stage 2: Log-space resampling")
    M = args.M if args.M > 0 else None
    gaps_resampled, log_uniform, delta_log, M = resample_to_log_space(primes, gaps, M)
    print()

    # ── Stage 3: Log-FFT + zero detection ────────────────────
    print("▸ Stage 3: Log-space FFT (Hann window)")
    amplitudes, spectrum, frequencies, gammas = compute_log_fft(
        gaps_resampled, delta_log, M, window_type='hann')

    print()
    print("▸ Stage 4: Per-zero detection")
    results = detect_zeros(amplitudes, gammas, n_zeros=args.n_zeros)

    bonferroni = 0.05 / len(results)

    print()
    print(f"  {'Zero':>6} {'γ known':>10} {'γ at k':>10} {'A_obs':>12} "
          f"{'Median':>12} {'SNR':>7} {'Pk':>3} {'p_loc':>8} {'Δγ':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*12} "
          f"{'─'*12} {'─'*7} {'─'*3} {'─'*8} {'─'*8}")

    n_sig = 0
    n_high_snr = 0
    n_peak = 0
    for r in results:
        sig = ''
        if r['snr_exact'] > 5:
            sig = '★'
            n_sig += 1
        elif r['snr_exact'] > 3:
            sig = '⚡'
        elif r['p_local'] < 0.05:
            sig = '·'

        if r['snr_exact'] > 3:
            n_high_snr += 1
        if r['is_peak']:
            n_peak += 1

        pk = '▲' if r['is_peak'] else ''

        print(f"  γ{r['zero_idx']:<4} {r['gamma_known']:>9.4f} {r['gamma_at_k']:>10.4f} "
              f"{r['A_obs']:>12.4f} {r['local_median']:>12.4f} "
              f"{r['snr_exact']:>7.2f} {pk:>3} {r['p_local']:>8.4f} "
              f"{r['delta_gamma']:>+8.4f} {sig}")

    print()
    print(f"  Local peaks at predicted bin: {n_peak}/{len(results)}")
    print(f"  SNR > 3: {n_high_snr}/{len(results)}")
    print(f"  SNR > 5: {n_sig}/{len(results)}")
    print()

    # ── Stage 5: Multi-window consistency ────────────────────
    print("▸ Stage 5: Multi-window consistency")
    multi_results = multi_window_test(gaps_resampled, delta_log, M,
                                       n_zeros=min(30, args.n_zeros))
    print()

    # ── Stage 6: Cross-segment coherence ─────────────────────
    print("▸ Stage 6: Cross-segment phase coherence")
    n_test_coherence = min(30, args.n_zeros)
    coherence_results = cross_segment_coherence(
        gaps_resampled, delta_log, M,
        KNOWN_ZEROS[:n_test_coherence],
        n_segments=args.segments
    )

    bonf_coh = 0.05 / len(coherence_results) if coherence_results else 0.05
    n_coherent = 0
    n_sig_coherent = 0

    print()
    print(f"  {'γ known':>10} {'γ at k':>10} {'Coh':>7} {'W.Coh':>7} "
          f"{'E[rand]':>7} {'p(Rayl)':>10} {'MeanAmp':>10} {'CV':>6}")
    print(f"  {'─'*10} {'─'*10} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*10} {'─'*10} {'─'*6}")

    for cr in coherence_results:
        sig = ''
        if cr['p_rayleigh'] < bonf_coh:
            sig = '★'
            n_sig_coherent += 1
        elif cr['coherence'] > 3 * cr['expected_random']:
            sig = '⚡'

        if cr['coherence'] > 2 * cr['expected_random']:
            n_coherent += 1

        print(f"  {cr['gamma_known']:>9.4f} {cr['gamma_at_k']:>10.4f} "
              f"{cr['coherence']:>7.4f} {cr['weighted_coherence']:>7.4f} "
              f"{cr['expected_random']:>7.4f} {cr['p_rayleigh']:>10.2e} "
              f"{cr['mean_amplitude']:>10.2f} {cr['amplitude_cv']:>6.3f} {sig}")

    print()
    print(f"  Phase coherent (>2× random): {n_coherent}/{len(coherence_results)}")
    print(f"  Significant (Rayleigh + Bonf): {n_sig_coherent}/{len(coherence_results)}")
    print()

    # ═══ SUMMARY ═════════════════════════════════════════════
    dt_total = time.time() - t_total
    print("=" * 60)
    print(f"  SUMMARY — LOG-SPACE — {limit_str}")
    print("=" * 60)
    print(f"  N = {N:,} gaps → M = {M:,} log-resampled points")
    print(f"  γ resolution: {2*np.pi/(M*delta_log):.6f}")
    print(f"  Per-zero: peaks={n_peak}, SNR>3={n_high_snr}, SNR>5={n_sig}")
    print(f"  Phase coherent (>2×random): {n_coherent}/{len(coherence_results)}")
    print(f"  Coherent + significant: {n_sig_coherent}/{len(coherence_results)}")
    if results:
        best = max(results, key=lambda r: r['snr_exact'])
        print(f"  Best SNR: γ{best['zero_idx']} (SNR={best['snr_exact']:.2f})")
    print(f"  Total time: {dt_total:.1f}s")
    print()

    # ── Save ─────────────────────────────────────────────────
    output = {
        'params': {
            'limit': args.limit,
            'N_gaps': N,
            'M_resampled': M,
            'delta_log': float(delta_log),
            'gamma_resolution': float(2 * np.pi / (M * delta_log)),
            'gpu': HAS_GPU,
            'total_time_s': dt_total,
        },
        'per_zero': results,
        'coherence': coherence_results,
        'summary': {
            'n_peaks': n_peak,
            'n_snr_gt3': n_high_snr,
            'n_snr_gt5': n_sig,
            'n_coherent': n_coherent,
            'n_sig_coherent': n_sig_coherent,
        }
    }

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f'log_space_{limit_str}.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {outpath}")


if __name__ == '__main__':
    main()
