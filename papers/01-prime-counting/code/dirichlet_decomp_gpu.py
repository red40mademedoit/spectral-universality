#!/usr/bin/env python3
"""
Dirichlet Decomposition of Prime Counting Function — B200 GPU
===============================================================

Instead of analyzing π(x) globally, decompose by arithmetic progression:
  π(x; q, a) = count of primes p ≤ x with p ≡ a (mod q)

By Dirichlet's theorem, each progression has its own L-function zeros.
The spectral signature of π(x; q, a) - Li(x)/φ(q) should reveal
L(s, χ) zeros for characters χ mod q.

Moduli tested:
  q=3:  a ∈ {1, 2}        — 2 non-trivial characters
  q=4:  a ∈ {1, 3}        — the real character χ₄
  q=5:  a ∈ {1, 2, 3, 4}  — 4 characters
  q=6:  a ∈ {1, 5}         — same as q=3 essentially
  q=8:  a ∈ {1, 3, 5, 7}  — 4 characters
  q=12: a ∈ {1, 5, 7, 11} — 4 characters

For each (q, a), compute:
  f(t; q, a) = (π(eᵗ; q, a) - Li(eᵗ)/φ(q)) · e^{-t/2} · t

FFT of f → peaks at γ values of L(s, χ) zeros.

Then: do different characters produce different spectral signatures?
This is the universality question — Katz-Sarnak predicts GUE for all.

Authors: Shadow + Dreadbot 3.2.666
Date: 2026-02-28
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.special import expi
from scipy.stats import pearsonr

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = np

try:
    import primesieve
    import primesieve.numpy
    HAS_PRIMESIEVE = True
except ImportError:
    HAS_PRIMESIEVE = False


ZETA_ZEROS = np.array([
    14.134725141734693790, 21.022039638771554993, 25.010857580145688763,
    30.424876125859513210, 32.935061587739189691, 37.586178158825671257,
    40.918719012147495187, 43.327073280914999519, 48.005150881167159728,
    49.773832477672302182, 52.970321477714460644, 56.446247697063394804,
    59.347044002602353085, 60.831778524609809844, 65.112544048081606661,
    67.079810529494173715, 69.546401711173979253, 72.067157674481907582,
    75.704690699083933168, 77.144840068874805373,
])


def euler_phi(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def coprime_residues(q):
    return [a for a in range(1, q) if math.gcd(a, q) == 1]


def generate_primes(limit):
    t0 = time.time()
    if HAS_PRIMESIEVE:
        primes = primesieve.numpy.primes(int(limit))
    else:
        result = subprocess.run(['primesieve', str(int(limit)), '-p'],
                                capture_output=True, text=False)
        primes = np.fromstring(result.stdout, dtype=np.int64, sep='\n')
    dt = time.time() - t0
    print(f"  {len(primes):,} primes in {dt:.1f}s")
    return primes


def analyze_progression(primes, q, a, M, t_min, t_max):
    """
    Analyze primes in the progression p ≡ a (mod q).
    Returns the detrended spectral signal and its FFT.
    """
    # Filter primes by residue class
    mask = (primes % q) == a
    primes_qa = primes[mask]
    n_qa = len(primes_qa)

    if n_qa < 100:
        return None

    phi_q = euler_phi(q)

    # Uniform grid in t = log(x)
    t = np.linspace(t_min, t_max, M, dtype=np.float64)
    x = np.exp(t)
    dt_spacing = t[1] - t[0]

    # π(x; q, a) via searchsorted
    primes_qa_f64 = primes_qa.astype(np.float64)
    pi_qa = np.searchsorted(primes_qa_f64, x, side='right').astype(np.float64)

    # Expected: Li(x) / φ(q)
    li_values = (expi(t) - expi(np.log(2.0))) / phi_q

    # Detrend: (π(x;q,a) - Li(x)/φ(q)) · e^{-t/2} · t
    diff = pi_qa - li_values
    weight = np.exp(-t / 2) * t
    signal = diff * weight
    signal = signal - np.mean(signal)

    # FFT on GPU
    if HAS_GPU:
        sig_gpu = cp.asarray(signal)
        window = cp.hanning(M)
        windowed = sig_gpu * window
        spectrum = cp.fft.rfft(windowed)
        amplitudes = cp.asnumpy(cp.abs(spectrum))
        del sig_gpu, windowed, spectrum
        cp.get_default_memory_pool().free_all_blocks()
    else:
        window = np.hanning(M)
        windowed = signal * window
        spectrum = np.fft.rfft(windowed)
        amplitudes = np.abs(spectrum)

    frequencies = np.fft.rfftfreq(M, d=dt_spacing)
    gammas = 2 * np.pi * frequencies

    return {
        'n_primes': n_qa,
        'amplitudes': amplitudes,
        'gammas': gammas,
        'signal_rms': float(np.std(signal)),
    }


def detect_zeros(amplitudes, gammas, known_zeros, local_window=500):
    """Per-zero SNR detection."""
    results = []
    for i, gamma in enumerate(known_zeros):
        k = int(np.argmin(np.abs(gammas - gamma)))
        if k < 1 or k >= len(amplitudes):
            continue

        A_obs = amplitudes[k]

        # Local noise
        lo = max(1, k - local_window)
        hi = min(len(amplitudes), k + local_window + 1)
        mask_arr = np.ones(hi - lo, dtype=bool)
        center = k - lo
        excl_lo = max(0, center - 10)
        excl_hi = min(hi - lo, center + 11)
        mask_arr[excl_lo:excl_hi] = False
        local_amps = amplitudes[lo:hi][mask_arr]

        if len(local_amps) < 20:
            local_amps = amplitudes[lo:hi]

        local_median = np.median(local_amps)
        local_mad = np.median(np.abs(local_amps - local_median))
        sigma = local_mad * 1.4826

        snr = (A_obs - local_median) / sigma if sigma > 0 else 0

        is_peak = all(
            amplitudes[k + dk] <= A_obs
            for dk in [-2, -1, 1, 2]
            if 0 <= k + dk < len(amplitudes)
        )

        results.append({
            'zero_idx': i + 1,
            'gamma': float(gamma),
            'snr': float(snr),
            'is_peak': is_peak,
            'A_obs': float(A_obs),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=float, default=1e9)
    parser.add_argument('--M', type=int, default=1 << 20)
    parser.add_argument('--moduli', type=int, nargs='+', default=[3, 4, 5, 8, 12])
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    limit_str = f"{args.limit:.0e}".replace('+', '')

    print("=" * 65)
    print(f"  DIRICHLET DECOMPOSITION — {limit_str}")
    print("=" * 65)
    print(f"  Moduli: {args.moduli}")
    print(f"  M = {args.M:,}")
    print()

    # Generate primes
    print("▸ Generating primes")
    primes = generate_primes(args.limit)
    t_min = np.log(2.0)
    t_max = np.log(float(args.limit))
    print()

    # Global analysis (all primes) for comparison
    print("▸ Global (all primes)")
    global_result = analyze_progression(primes, 1, 0, args.M, t_min, t_max)
    # For q=1, a=0, every prime matches, φ(1)=1
    # Actually need to handle this differently — just use all primes
    # Recompute with all primes directly
    t = np.linspace(t_min, t_max, args.M, dtype=np.float64)
    x = np.exp(t)
    dt_spacing = t[1] - t[0]
    primes_f64 = primes.astype(np.float64)
    pi_all = np.searchsorted(primes_f64, x, side='right').astype(np.float64)
    li_all = expi(t) - expi(np.log(2.0))
    diff_all = pi_all - li_all
    weight_all = np.exp(-t / 2) * t
    signal_all = diff_all * weight_all
    signal_all = signal_all - np.mean(signal_all)

    if HAS_GPU:
        sig_gpu = cp.asarray(signal_all)
        spectrum_all = cp.fft.rfft(sig_gpu * cp.hanning(args.M))
        amps_all = cp.asnumpy(cp.abs(spectrum_all))
        del sig_gpu, spectrum_all
        cp.get_default_memory_pool().free_all_blocks()
    else:
        spectrum_all = np.fft.rfft(signal_all * np.hanning(args.M))
        amps_all = np.abs(spectrum_all)

    gammas_all = 2 * np.pi * np.fft.rfftfreq(args.M, d=dt_spacing)
    global_zeros = detect_zeros(amps_all, gammas_all, ZETA_ZEROS)
    global_snrs = {r['zero_idx']: r['snr'] for r in global_zeros}

    n_global_gt3 = sum(1 for r in global_zeros if r['snr'] > 3)
    print(f"  SNR>3: {n_global_gt3}/{len(global_zeros)}")
    print()

    # Per-modulus, per-residue analysis
    all_output = {'global': {'snrs': global_snrs}}

    for q in args.moduli:
        residues = coprime_residues(q)
        phi_q = euler_phi(q)
        print(f"▸ q = {q}, φ(q) = {phi_q}, residues = {residues}")

        for a in residues:
            result = analyze_progression(primes, q, a, args.M, t_min, t_max)
            if result is None:
                print(f"  a={a}: too few primes, skipping")
                continue

            zeros = detect_zeros(result['amplitudes'], result['gammas'], ZETA_ZEROS)
            n_primes = result['n_primes']
            n_gt3 = sum(1 for z in zeros if z['snr'] > 3)
            n_gt5 = sum(1 for z in zeros if z['snr'] > 5)
            n_peak = sum(1 for z in zeros if z['is_peak'])

            # Top 5 by SNR
            top5 = sorted(zeros, key=lambda z: z['snr'], reverse=True)[:5]
            top_str = ", ".join(f"γ{z['zero_idx']}={z['snr']:.1f}" for z in top5)

            print(f"  a≡{a} (mod {q}): {n_primes:,} primes, "
                  f"SNR>3={n_gt3}/{len(zeros)}, SNR>5={n_gt5}/{len(zeros)}, "
                  f"peaks={n_peak}/{len(zeros)}")
            print(f"    Top: {top_str}")

            key = f"q{q}_a{a}"
            all_output[key] = {
                'q': q, 'a': a, 'phi_q': phi_q,
                'n_primes': n_primes,
                'n_snr3': n_gt3, 'n_snr5': n_gt5, 'n_peak': n_peak,
                'signal_rms': result['signal_rms'],
                'zeros': zeros,
            }

        print()

    # ── Summary comparison ───────────────────────────────────
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Progression':<20} {'Primes':>12} {'SNR>3':>7} {'SNR>5':>7} {'Peaks':>7} {'Top SNR':>10}")
    print(f"  {'─'*20} {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")

    print(f"  {'Global':<20} {len(primes):>12,} {n_global_gt3:>5}/20 {'':>7} {'':>7} "
          f"{max(r['snr'] for r in global_zeros):>10.1f}")

    for key in sorted(all_output):
        if key == 'global':
            continue
        d = all_output[key]
        top_snr = max(z['snr'] for z in d['zeros']) if d['zeros'] else 0
        prog_label = f"p≡{d['a']} (mod {d['q']})"
        print(f"  {prog_label:20} {d['n_primes']:>12,} "
              f"{d['n_snr3']:>5}/20 {d['n_snr5']:>5}/20 {d['n_peak']:>5}/20 "
              f"{top_snr:>10.1f}")

    # ── Cross-residue comparison ─────────────────────────────
    # For each modulus, compare SNR patterns across residue classes
    print()
    print("▸ Cross-residue SNR comparison (do different residues see different zeros?)")

    for q in args.moduli:
        residues = coprime_residues(q)
        keys = [f"q{q}_a{a}" for a in residues if f"q{q}_a{a}" in all_output]
        if len(keys) < 2:
            continue

        print(f"\n  mod {q}:")
        # Compare SNR vectors across residue classes
        snr_vectors = {}
        for key in keys:
            snrs = [0.0] * 20
            for z in all_output[key]['zeros']:
                if z['zero_idx'] <= 20:
                    snrs[z['zero_idx'] - 1] = z['snr']
            snr_vectors[key] = np.array(snrs)

        # Pairwise correlation of SNR patterns
        key_list = list(snr_vectors.keys())
        for i in range(len(key_list)):
            for j in range(i + 1, len(key_list)):
                k1, k2 = key_list[i], key_list[j]
                r, p = pearsonr(snr_vectors[k1], snr_vectors[k2])
                a1 = all_output[k1]['a']
                a2 = all_output[k2]['a']
                print(f"    a={a1} vs a={a2}: r={r:.4f}, p={p:.4f} "
                      f"{'(DIFFERENT)' if p < 0.05 and r < 0.5 else '(similar)'}")

    # Save
    # Strip numpy arrays from output
    output_clean = {}
    for key, val in all_output.items():
        if isinstance(val, dict):
            output_clean[key] = {k: v for k, v in val.items()
                                  if not isinstance(v, np.ndarray)}

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f'dirichlet_{limit_str}.json')
    with open(outpath, 'w') as f:
        json.dump(output_clean, f, indent=2, default=str)
    print(f"\n  Saved to {outpath}")


if __name__ == '__main__':
    main()
