#!/usr/bin/env python3
"""
Counting Function Zero Detector — B200 GPU
============================================

Analyze the oscillatory part of the prime counting function:

Signal A: f_π(t) = (π(eᵗ) - Li(eᵗ)) · e^{-t/2} · t
Signal B: f_ψ(t) = (ψ(eᵗ) - eᵗ) · e^{-t/2}

Both should be sums of pure sinusoids cos(γₙt + φₙ)/|ρₙ|
with NO amplitude modulation by gap size.

γ = 2π·f  (direct mapping, zero free parameters)

Key insight (Dreadbot): gaps fail because the explicit formula
contributes ~ g_n · p_n^{-1/2} · e^{iγ·log p_n} / ρ — the gap
itself modulates the amplitude, smearing spectral peaks. Working
with π(x) or ψ(x) directly eliminates this.

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
from scipy.stats import pearsonr

try:
    import cupy as cp
    HAS_GPU = True
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"GPU: {gpu_name}")
except ImportError:
    HAS_GPU = False
    cp = np
    print("No GPU — CPU mode")

try:
    import primesieve
    import primesieve.numpy
    HAS_PRIMESIEVE = True
except ImportError:
    HAS_PRIMESIEVE = False

# ═══════════════════════════════════════════════════════════════
# HIGH-PRECISION ZEROS (Wolfram, 20 digits)
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


def li(x):
    """Li(x) = Ei(ln(x))"""
    return expi(np.log(x))


# ═══════════════════════════════════════════════════════════════
# COMPUTE π(x) AND ψ(x)
# ═══════════════════════════════════════════════════════════════

def compute_signals(max_x, M):
    """
    Compute π(eᵗ), Li(eᵗ), and ψ(eᵗ) on a uniform grid in t = log(x).

    π(x) via searchsorted on prime array.
    ψ(x) via cumsum(log(p)) + prime power corrections.
    Li(x) via scipy expi.
    """
    t_min = np.log(2.0)
    t_max = np.log(float(max_x))
    t = np.linspace(t_min, t_max, M, dtype=np.float64)
    x = np.exp(t)
    dt = t[1] - t[0]

    print(f"  t range: [{t_min:.4f}, {t_max:.4f}], dt = {dt:.2e}")
    print(f"  γ resolution: 2π/T = {2*np.pi/(t_max - t_min):.6f}")

    # ── Generate all primes ──────────────────────────────────
    t0 = time.time()
    if HAS_PRIMESIEVE:
        primes = primesieve.numpy.primes(int(max_x))
    else:
        result = subprocess.run(['primesieve', str(int(max_x)), '-p'],
                                capture_output=True, text=False)
        primes = np.fromstring(result.stdout, dtype=np.int64, sep='\n')
    dt_gen = time.time() - t0
    n_primes = len(primes)
    print(f"  Generated {n_primes:,} primes in {dt_gen:.1f}s")

    primes_f64 = primes.astype(np.float64)

    # ── π(x) via searchsorted ────────────────────────────────
    t0 = time.time()
    pi_values = np.searchsorted(primes_f64, x, side='right').astype(np.float64)
    dt_pi = time.time() - t0
    print(f"  π(x) computed in {dt_pi:.2f}s, π({max_x:.0e}) = {int(pi_values[-1])}")

    # ── Li(x) via expi ───────────────────────────────────────
    t0 = time.time()
    li_values = expi(t)  # Li(eᵗ) = Ei(t) ... wait, Li(x) = Ei(ln(x)) = Ei(t)
    # Actually Li(x) = li(x) = Ei(ln(x)), but standard convention subtracts li(2)
    # π(x) ~ Li(x) = Ei(ln(x)) - Ei(ln(2)) ≈ Ei(t) - 1.0451
    li_offset = expi(np.log(2.0))  # Ei(ln(2)) ≈ 1.0451
    li_values = li_values - li_offset
    dt_li = time.time() - t0
    print(f"  Li(x) computed in {dt_li:.2f}s, Li({max_x:.0e}) = {li_values[-1]:.2f}")

    # ── ψ(x) via cumsum(log(p)) + prime power corrections ───
    t0 = time.time()
    log_primes = np.log(primes_f64)
    cumsum_logp = np.cumsum(log_primes)
    # Prepend 0 for searchsorted index = 0 case
    cumsum_logp_ext = np.concatenate([[0.0], cumsum_logp])

    # ψ₁(x) = Σ_{p≤x} log(p)  (prime contribution)
    pi_idx = np.searchsorted(primes_f64, x, side='right')
    psi_primes = cumsum_logp_ext[pi_idx]

    # ψ₂(x) = Σ_{p²≤x} log(p) = ψ₁(√x)
    sqrt_x = np.sqrt(x)
    pi_sqrt = np.searchsorted(primes_f64, sqrt_x, side='right')
    psi_power2 = cumsum_logp_ext[pi_sqrt]

    # ψ₃(x) = Σ_{p³≤x} log(p) = ψ₁(x^{1/3})
    cbrt_x = np.cbrt(x)
    pi_cbrt = np.searchsorted(primes_f64, cbrt_x, side='right')
    psi_power3 = cumsum_logp_ext[pi_cbrt]

    # Higher powers negligible for x > 1000
    psi_values = psi_primes + psi_power2 + psi_power3
    dt_psi = time.time() - t0
    print(f"  ψ(x) computed in {dt_psi:.2f}s, ψ({max_x:.0e}) = {psi_values[-1]:.2f}")
    print(f"  Chebyshev: ψ/x = {psi_values[-1]/x[-1]:.6f} (should → 1.0)")

    return t, x, dt, pi_values, li_values, psi_values


# ═══════════════════════════════════════════════════════════════
# DETREND AND FFT
# ═══════════════════════════════════════════════════════════════

def analyze_signal(t, signal, dt_spacing, label, n_zeros=50,
                   window_type='hann', local_window=500):
    """
    FFT a detrended signal on uniform t-grid.
    Returns per-zero detection results.
    """
    M = len(signal)

    # Move to GPU
    if HAS_GPU:
        sig_gpu = cp.asarray(signal)
    else:
        sig_gpu = signal.copy()

    # Center
    sig_gpu = sig_gpu - cp.mean(sig_gpu)

    # Window
    if window_type == 'hann':
        window = cp.hanning(M)
    elif window_type == 'blackman':
        window = cp.blackman(M)
    else:
        window = cp.ones(M)

    windowed = sig_gpu * window

    # FFT
    t0 = time.time()
    spectrum = cp.fft.rfft(windowed)
    amplitudes = cp.abs(spectrum)
    frequencies = cp.fft.rfftfreq(M, d=dt_spacing)
    gammas = 2 * cp.pi * frequencies
    dt_fft = time.time() - t0

    amps_np = cp.asnumpy(amplitudes) if HAS_GPU else amplitudes
    gammas_np = cp.asnumpy(gammas) if HAS_GPU else gammas

    print(f"  [{label}] FFT of {M:,} points in {dt_fft:.3f}s")

    # Per-zero detection
    results = []
    n_test = min(n_zeros, len(KNOWN_ZEROS))

    for i in range(n_test):
        gamma = KNOWN_ZEROS[i]
        k = int(np.argmin(np.abs(gammas_np - gamma)))

        if k < 1 or k >= len(amps_np):
            continue

        A_obs = amps_np[k]
        gamma_at_k = gammas_np[k]

        # Best in ±3 neighborhood
        best_k = k
        best_A = A_obs
        for dk in range(-3, 4):
            kn = k + dk
            if 0 < kn < len(amps_np) and amps_np[kn] > best_A:
                best_k = kn
                best_A = amps_np[kn]

        # Local noise
        lo = max(1, k - local_window)
        hi = min(len(amps_np), k + local_window + 1)
        mask = np.ones(hi - lo, dtype=bool)
        center = k - lo
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

        # Is exact bin a local peak?
        is_peak = all(
            amps_np[k + dk] <= A_obs
            for dk in [-2, -1, 1, 2]
            if 0 <= k + dk < len(amps_np)
        )

        p_local = float(np.mean(local_amps >= A_obs))

        results.append({
            'zero_idx': i + 1,
            'gamma_known': float(gamma),
            'k': k,
            'gamma_at_k': float(gamma_at_k),
            'A_obs': float(A_obs),
            'best_k': best_k,
            'best_gamma': float(gammas_np[best_k]),
            'best_A': float(best_A),
            'delta_gamma': float(gammas_np[best_k] - gamma),
            'local_median': float(local_median),
            'sigma': float(sigma),
            'snr': float(snr),
            'snr_best': float(snr_best),
            'is_peak': is_peak,
            'p_local': float(p_local),
        })

    # Clean up GPU
    if HAS_GPU:
        del sig_gpu, windowed, spectrum, amplitudes, frequencies, gammas
        cp.get_default_memory_pool().free_all_blocks()

    return results, amps_np, gammas_np


def cross_segment_coherence(signal, dt_spacing, target_zeros, n_segments=8):
    """
    Phase coherence test across segments.
    """
    M = len(signal)
    seg_len = M // n_segments

    if HAS_GPU:
        sig_gpu = cp.asarray(signal)
    else:
        sig_gpu = signal.copy()

    # Per-segment FFT
    segment_spectra = []
    for s in range(n_segments):
        start = s * seg_len
        end = start + seg_len
        seg = sig_gpu[start:end]
        seg = seg - cp.mean(seg)
        seg = seg * cp.hanning(seg_len)
        spec = cp.fft.rfft(seg)
        segment_spectra.append(spec)

    seg_freqs = cp.fft.rfftfreq(seg_len, d=dt_spacing)
    seg_gammas = 2 * cp.pi * seg_freqs
    seg_gammas_np = cp.asnumpy(seg_gammas) if HAS_GPU else seg_gammas

    results = []
    for gamma in target_zeros:
        k = int(np.argmin(np.abs(seg_gammas_np - gamma)))
        if k < 1 or k >= len(seg_gammas_np):
            continue

        gamma_at_k = seg_gammas_np[k]

        phases = []
        amps = []
        for spec in segment_spectra:
            val = spec[k]
            if HAS_GPU:
                val = val.get()
            phases.append(float(np.angle(val)))
            amps.append(float(np.abs(val)))

        phases = np.array(phases)
        amps = np.array(amps)

        # Rayleigh test of phase uniformity
        unit_vectors = np.exp(1j * phases)
        coherence = np.abs(np.mean(unit_vectors)) ** 2
        expected_random = 1.0 / n_segments
        p_rayleigh = float(np.exp(-n_segments * coherence))

        results.append({
            'gamma_known': float(gamma),
            'gamma_at_k': float(gamma_at_k),
            'coherence': float(coherence),
            'expected_random': float(expected_random),
            'p_rayleigh': float(p_rayleigh),
            'mean_amp': float(np.mean(amps)),
            'amp_cv': float(np.std(amps) / np.mean(amps)) if np.mean(amps) > 0 else 0,
        })

    if HAS_GPU:
        del sig_gpu
        for s in segment_spectra:
            del s
        cp.get_default_memory_pool().free_all_blocks()

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=float, default=1e9)
    parser.add_argument('--M', type=int, default=1<<20, help="Sample points (default 2^20)")
    parser.add_argument('--n-zeros', type=int, default=50)
    parser.add_argument('--segments', type=int, default=8)
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    limit_str = f"{args.limit:.0e}".replace('+', '')
    M = args.M

    print("=" * 65)
    print(f"  COUNTING FUNCTION ZERO DETECTOR — {limit_str}")
    print("=" * 65)
    print(f"  M = {M:,} sample points")
    print()

    # ── Compute signals ──────────────────────────────────────
    print("▸ Stage 1: Compute π(x), Li(x), ψ(x)")
    t, x, dt_spacing, pi_val, li_val, psi_val = compute_signals(args.limit, M)
    print()

    # ── Detrend ──────────────────────────────────────────────
    print("▸ Stage 2: Detrend")

    # Signal A: (π(eᵗ) - Li(eᵗ)) · e^{-t/2} · t
    diff_pi = pi_val - li_val
    weight_pi = np.exp(-t / 2) * t
    signal_A = diff_pi * weight_pi
    print(f"  Signal A: (π-Li)·e^{{-t/2}}·t")
    print(f"    RMS = {np.std(signal_A):.4f}")
    print(f"    Range = [{np.min(signal_A):.2f}, {np.max(signal_A):.2f}]")

    # Signal B: (ψ(eᵗ) - eᵗ) · e^{-t/2}
    diff_psi = psi_val - x
    weight_psi = np.exp(-t / 2)
    signal_B = diff_psi * weight_psi
    print(f"  Signal B: (ψ-x)·e^{{-t/2}}")
    print(f"    RMS = {np.std(signal_B):.4f}")
    print(f"    Range = [{np.min(signal_B):.2f}, {np.max(signal_B):.2f}]")

    # Signal C: (ψ(eᵗ) - eᵗ) · e^{-t/2} · t  (extra log weighting)
    signal_C = diff_psi * weight_psi * t
    print(f"  Signal C: (ψ-x)·e^{{-t/2}}·t")
    print(f"    RMS = {np.std(signal_C):.4f}")
    print()

    # ── FFT all signals ──────────────────────────────────────
    print("▸ Stage 3: FFT + per-zero detection")

    all_results = {}
    all_amps = {}
    all_gammas = {}

    for label, signal in [("π-Li", signal_A), ("ψ-x", signal_B), ("ψ-x·t", signal_C)]:
        results, amps, gammas = analyze_signal(
            t, signal, dt_spacing, label,
            n_zeros=args.n_zeros, window_type='hann'
        )
        all_results[label] = results
        all_amps[label] = amps
        all_gammas[label] = gammas

    # ── Print comparison table ───────────────────────────────
    print()
    print("▸ Stage 4: Signal comparison")
    print()

    # Header
    print(f"  {'Zero':>6} {'γ':>8} "
          f"{'SNR(π-Li)':>10} {'SNR(ψ-x)':>10} {'SNR(ψt)':>10} "
          f"{'Pk(π)':>6} {'Pk(ψ)':>6} {'Pk(ψt)':>7}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*6} {'─'*7}")

    n_show = min(30, args.n_zeros)
    counts = {sig: {'snr3': 0, 'snr5': 0, 'peak': 0} for sig in all_results}

    for i in range(n_show):
        gamma = KNOWN_ZEROS[i]
        row = f"  γ{i+1:<4} {gamma:>7.3f} "

        for label in ["π-Li", "ψ-x", "ψ-x·t"]:
            matching = [r for r in all_results[label] if r['zero_idx'] == i + 1]
            if matching:
                r = matching[0]
                snr = r['snr']
                pk = '▲' if r['is_peak'] else ''
                sig = ''
                if snr > 5:
                    sig = '★'
                    counts[label]['snr5'] += 1
                    counts[label]['snr3'] += 1
                elif snr > 3:
                    sig = '⚡'
                    counts[label]['snr3'] += 1
                if r['is_peak']:
                    counts[label]['peak'] += 1
                row += f" {snr:>8.2f}{sig:1} "
            else:
                row += f" {'N/A':>9} "

        # Peaks column
        for label in ["π-Li", "ψ-x", "ψ-x·t"]:
            matching = [r for r in all_results[label] if r['zero_idx'] == i + 1]
            if matching:
                pk = '▲' if matching[0]['is_peak'] else ' '
                row += f"  {pk:>4} "
            else:
                row += f"  {'':>4} "

        print(row)

    print()
    print(f"  {'Signal':<12} {'SNR>3':>7} {'SNR>5':>7} {'Peaks':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7}")
    for label in ["π-Li", "ψ-x", "ψ-x·t"]:
        c = counts[label]
        print(f"  {label:<12} {c['snr3']:>5}/{n_show}  {c['snr5']:>5}/{n_show}  {c['peak']:>5}/{n_show}")
    print()

    # ── Best signal: detailed analysis ───────────────────────
    # Pick whichever has most SNR>3 hits
    best_label = max(counts, key=lambda k: counts[k]['snr3'])
    best_results = all_results[best_label]
    print(f"  Best signal: {best_label}")
    print()

    # Δγ analysis for best signal
    print(f"  Detailed {best_label} results:")
    print(f"  {'Zero':>6} {'γ':>10} {'Δγ':>8} {'SNR':>7} {'p_loc':>8} {'Pk':>3}")
    print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*7} {'─'*8} {'─'*3}")
    for r in best_results[:30]:
        pk = '▲' if r['is_peak'] else ''
        sig = ''
        if r['snr'] > 5:
            sig = ' ★'
        elif r['snr'] > 3:
            sig = ' ⚡'
        print(f"  γ{r['zero_idx']:<4} {r['gamma_known']:>9.4f} {r['delta_gamma']:>+8.4f} "
              f"{r['snr']:>7.2f} {r['p_local']:>8.4f} {pk:>3}{sig}")

    # Amplitude decay
    if best_results:
        gammas_known = np.array([r['gamma_known'] for r in best_results])
        amps_obs = np.array([r['A_obs'] for r in best_results])
        expected = 1.0 / np.sqrt(0.25 + gammas_known**2)
        scale = amps_obs[0] / expected[0] if expected[0] > 0 else 1
        expected_scaled = expected * scale
        r_corr, p_corr = pearsonr(amps_obs, expected_scaled)
        print(f"\n  Amplitude decay: r(A, 1/|ρ|) = {r_corr:.4f} (p = {p_corr:.2e})")
    print()

    # ── Phase coherence on best signal ───────────────────────
    print(f"▸ Stage 5: Phase coherence ({best_label}, {args.segments} segments)")
    best_signal = {"π-Li": signal_A, "ψ-x": signal_B, "ψ-x·t": signal_C}[best_label]
    coh_results = cross_segment_coherence(
        best_signal, dt_spacing, KNOWN_ZEROS[:30], n_segments=args.segments
    )

    bonf_coh = 0.05 / len(coh_results) if coh_results else 0.05
    n_coherent = 0
    n_sig_coh = 0

    print()
    print(f"  {'γ':>8} {'Coh':>7} {'E[rand]':>7} {'p(Rayl)':>10} {'Amp':>10}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*10} {'─'*10}")

    for cr in coh_results:
        sig = ''
        if cr['p_rayleigh'] < bonf_coh:
            sig = ' ★'
            n_sig_coh += 1
        if cr['coherence'] > 2 * cr['expected_random']:
            n_coherent += 1
        print(f"  {cr['gamma_known']:>7.3f} {cr['coherence']:>7.4f} "
              f"{cr['expected_random']:>7.4f} {cr['p_rayleigh']:>10.2e} "
              f"{cr['mean_amp']:>10.2f}{sig}")

    print()
    print(f"  Coherent (>2×random): {n_coherent}/{len(coh_results)}")
    print(f"  Significant (Bonf):   {n_sig_coh}/{len(coh_results)}")
    print()

    # ═══ SUMMARY ═════════════════════════════════════════════
    print("=" * 65)
    print(f"  SUMMARY — COUNTING FUNCTION — {limit_str}")
    print("=" * 65)
    for label in ["π-Li", "ψ-x", "ψ-x·t"]:
        c = counts[label]
        print(f"  {label:<12}: SNR>3={c['snr3']}, SNR>5={c['snr5']}, "
              f"peaks={c['peak']}")
    print(f"  Best signal: {best_label}")
    if best_results:
        best_snr = max(best_results, key=lambda r: r['snr'])
        print(f"  Best SNR: γ{best_snr['zero_idx']} = {best_snr['snr']:.2f}")
    print(f"  Phase coherent: {n_coherent}/{len(coh_results)}")
    print(f"  Amplitude decay: r = {r_corr:.3f}")

    # ── Save ─────────────────────────────────────────────────
    output = {
        'params': {
            'limit': args.limit,
            'M': M,
            'gamma_resolution': float(2 * np.pi / (t[-1] - t[0])),
            'dt': float(dt_spacing),
            'gpu': HAS_GPU,
        },
        'signals': {},
        'coherence': coh_results,
    }
    for label in all_results:
        output['signals'][label] = {
            'zeros': all_results[label],
            'n_snr3': counts[label]['snr3'],
            'n_snr5': counts[label]['snr5'],
            'n_peaks': counts[label]['peak'],
        }

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f'counting_fn_{limit_str}.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {outpath}")


if __name__ == '__main__':
    main()
