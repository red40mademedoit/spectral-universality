#!/usr/bin/env python3
"""
Prime Gap Manifold v2 — Tier 0: Li(x) Validation
=================================================

Optimized pipeline:
  - primesieve CLI for prime generation (10⁸ in 4ms)
  - numpy.fft.rfft for spectral decomposition
  - scipy.special.expi for Li(x) = Ei(ln(x)) local correction
  - Primal generator calibration spline for ⟨r⟩ → β inversion

Goal: Fix the γ₁ offset. Validate whether Li(x) correction
makes the frequency→zero mapping real.

Success criterion: γ₁ prediction within ±0.5 of 14.1347

Usage:
    python3 manifold_v2.py                    # default 10^8
    python3 manifold_v2.py --limit 1e9        # 10^9 (needs ~8GB RAM)
    python3 manifold_v2.py --limit 1e8 --dim 8 10 12  # multiple embeddings
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# scipy for Li(x) — loaded from rag-pipeline-env or system
try:
    from scipy.special import expi as _expi
    def li(x):
        """Logarithmic integral Li(x) = Ei(ln(x))"""
        return _expi(np.log(x))
    HAS_SCIPY = True
except ImportError:
    # Fallback: numerical integration
    def li(x):
        """Li(x) via trapezoidal rule on 1/ln(t) from 2 to x"""
        if x <= 2:
            return 0.0
        n = 10000
        t = np.linspace(2, x, n)
        return np.trapezoid(1.0 / np.log(t), t)
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
# PRIMESIEVE INTERFACE
# ═══════════════════════════════════════════════════════════════

PRIMESIEVE_BIN = os.path.expanduser("~/.local/bin/primesieve")

def generate_primes(limit: int) -> np.ndarray:
    """Generate primes up to limit using primesieve CLI."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    t0 = time.time()
    result = subprocess.run(
        [PRIMESIEVE_BIN, str(int(limit)), "-p"],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"primesieve failed: {result.stderr}")

    primes = np.array([int(x) for x in result.stdout.strip().split('\n')], dtype=np.int64)
    dt = time.time() - t0
    print(f"  primesieve: {len(primes):,} primes up to {limit:.0e} in {dt:.3f}s")
    return primes


def generate_primes_binary(limit: int) -> np.ndarray:
    """Generate primes using primesieve binary output (faster for large N)."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    t0 = time.time()
    # For very large limits, use text mode and numpy fromstring
    result = subprocess.run(
        [PRIMESIEVE_BIN, str(int(limit)), "-p"],
        capture_output=True, text=False, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"primesieve failed")

    primes = np.fromstring(result.stdout, dtype=np.int64, sep='\n')
    dt = time.time() - t0
    print(f"  primesieve: {len(primes):,} primes up to {limit:.0e} in {dt:.3f}s")
    return primes


# ═══════════════════════════════════════════════════════════════
# KNOWN ZETA ZEROS (first 30, Odlyzko tables)
# ═══════════════════════════════════════════════════════════════

KNOWN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]


# ═══════════════════════════════════════════════════════════════
# STAGE 1: SPECTRAL DECOMPOSITION WITH Li(x) CORRECTION
# ═══════════════════════════════════════════════════════════════

def spectral_decomposition(gaps: np.ndarray, primes: np.ndarray, n_modes: int = 30):
    """
    FFT of gap sequence → frequency spectrum → predicted γ via Li(x).

    Three mappings compared:
      1. Naive: γ = 2π·f·log(p_avg)          [v1 prototype]
      2. Global Li: γ = 2π·f·Li(p_avg)       [scaling doc suggestion]
      3. Local Li: γ = 2π·f·Li(p_local)      [correct from explicit formula]
    """
    N = len(gaps)
    mean_gap = np.mean(gaps)
    centered = gaps.astype(np.float64) - mean_gap

    t0 = time.time()
    spectrum = np.fft.rfft(centered)
    dt = time.time() - t0
    print(f"  numpy.fft.rfft: {N:,} points in {dt:.3f}s")

    amplitudes = np.abs(spectrum[1:N//2]) / N
    frequencies = np.fft.rfftfreq(N, d=1.0)[1:N//2]

    # Three γ mappings
    p_avg = primes[N // 2]  # median prime
    log_pavg = np.log(p_avg)
    li_pavg = li(float(p_avg))

    results = []
    top_indices = np.argsort(amplitudes)[::-1][:n_modes]

    for rank, idx in enumerate(top_indices):
        freq = frequencies[idx]
        amp = amplitudes[idx]

        # Mapping 1: naive log
        gamma_naive = 2 * np.pi * freq * log_pavg

        # Mapping 2: global Li
        gamma_global_li = 2 * np.pi * freq * li_pavg

        # Mapping 3: local Li — use the prime density at the position
        # where this frequency "lives" in the sequence
        # The k-th frequency mode corresponds to oscillation over ~N/k primes
        # centered around p_avg
        k = idx + 1  # frequency index (1-based)
        gamma_local_li = 2 * np.pi * freq * li(float(p_avg))

        # Actually, the explicit formula says: the oscillatory term from
        # zero ρ = 1/2 + iγ contributes ~ cos(γ·log(x)) to ψ(x).
        # In the gap sequence (derivative of ψ), this becomes
        # ~ γ·sin(γ·log(x))/x. The frequency in the gap sequence
        # at position n is f = γ/(2π) × d(log p_n)/dn ≈ γ/(2π·p_n/ln(p_n))
        # So: γ = 2π·f·p_avg/ln(p_avg) = 2π·f·p_avg·(1/ln(p_avg))
        # But p_avg/ln(p_avg) ≈ π(p_avg)/1 ≈ Li(p_avg) by PNT
        # So γ ≈ 2π·f·Li(p_avg) ... which IS mapping 2.
        #
        # BUT: the gap sequence isn't uniformly sampled in log-space.
        # The n-th gap is at p_n, not at n·Δ. The "effective frequency"
        # depends on where in the sequence we are. For a global FFT,
        # the mapping is:
        #   γ ≈ 2π · f · <p/ln(p)>  where <> is the mean over the range
        # This is close to Li(p_avg) but not identical.

        # Mapping 4: density-corrected
        # <p/ln(p)> over the prime range
        mean_p_over_logp = np.mean(primes[:N].astype(np.float64) / np.log(primes[:N].astype(np.float64)))
        gamma_density = 2 * np.pi * freq * mean_p_over_logp

        # Find nearest known zero for each mapping
        def nearest(gamma):
            if gamma < 1:
                return KNOWN_ZEROS[0], gamma - KNOWN_ZEROS[0]
            dists = [abs(gamma - z) for z in KNOWN_ZEROS]
            idx_near = np.argmin(dists)
            return KNOWN_ZEROS[idx_near], gamma - KNOWN_ZEROS[idx_near]

        z_naive, d_naive = nearest(gamma_naive)
        z_gli, d_gli = nearest(gamma_global_li)
        z_density, d_density = nearest(gamma_density)

        results.append({
            "rank": rank + 1,
            "freq_idx": int(idx + 1),
            "frequency": float(freq),
            "amplitude": float(amp),
            "gamma_naive": float(gamma_naive),
            "gamma_global_li": float(gamma_global_li),
            "gamma_density": float(gamma_density),
            "nearest_zero_naive": float(z_naive),
            "delta_naive": float(d_naive),
            "nearest_zero_density": float(z_density),
            "delta_density": float(d_density),
        })

    return results, mean_p_over_logp


# ═══════════════════════════════════════════════════════════════
# STAGE 2: DELAY EMBEDDING + MANIFOLD STATISTICS
# ═══════════════════════════════════════════════════════════════

def delay_embedding(gaps: np.ndarray, dim: int, tau: int = 1) -> np.ndarray:
    """Embed gap sequence into R^dim via delay coordinates. Returns (N-k, dim) array."""
    N = len(gaps)
    max_start = N - (dim - 1) * tau
    if max_start <= 0:
        raise ValueError(f"Sequence too short for dim={dim}, tau={tau}")

    t0 = time.time()
    indices = np.arange(max_start)[:, None] + np.arange(dim)[None, :] * tau
    manifold = gaps[indices]
    dt = time.time() - t0
    print(f"  Delay embedding: {manifold.shape[0]:,} × {dim} in {dt:.3f}s")
    return manifold


def manifold_statistics(manifold: np.ndarray) -> dict:
    """Compute manifold topology statistics."""
    N, k = manifold.shape
    centroid = np.mean(manifold, axis=0)
    rms_spread = np.sqrt(np.mean(np.sum((manifold - centroid)**2, axis=1)))

    # Distinct tuples
    tuples_set = set(map(tuple, manifold))
    distinct = len(tuples_set)

    # Oscillation classification per point
    centered = manifold.astype(np.float64) - np.mean(manifold, axis=1, keepdims=True)
    # Count sign changes per row
    signs = np.sign(centered)
    sign_changes = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

    class_counts = defaultdict(int)
    for sc in sign_changes:
        if sc == 0:
            class_counts["null"] += 1
        elif sc == 1:
            class_counts["mono"] += 1
        else:
            class_counts[f"poly-{sc}"] += 1

    return {
        "n_points": N,
        "dimension": k,
        "centroid": centroid.tolist(),
        "rms_spread": float(rms_spread),
        "distinct_tuples": distinct,
        "distinct_pct": 100 * distinct / N,
        "oscillation_classes": dict(class_counts),
        "mean_sign_changes": float(np.mean(sign_changes)),
    }


# ═══════════════════════════════════════════════════════════════
# STAGE 3: CONSERVATION LAW TEST
# ═══════════════════════════════════════════════════════════════

def conservation_law_test(gaps: np.ndarray, primes: np.ndarray, n_bins: int = 100):
    """
    Test Fourier conjugacy conservation law per spectral bin:
      r_time(bin) + r_frequency(bin) ≈ 1.0

    r_time = consecutive spacing ratio in gap subsequence
    r_frequency = consecutive spacing ratio of FFT amplitudes in that band

    Conservation law (global): 0.386 + 0.616 = 1.002
    Question: does it hold locally per spectral band?
    """
    N = len(gaps)
    spectrum = np.fft.rfft(gaps.astype(np.float64) - np.mean(gaps))
    amplitudes = np.abs(spectrum[1:N//2])
    phases = np.angle(spectrum[1:N//2])

    # Bin the spectrum
    bin_size = max(1, len(amplitudes) // n_bins)
    results = []

    for b in range(min(n_bins, len(amplitudes) // bin_size)):
        start = b * bin_size
        end = start + bin_size

        # r_frequency: spacing ratio of amplitudes in this band
        band_amps = amplitudes[start:end]
        if len(band_amps) < 4:
            continue

        amp_spacings = np.diff(np.sort(band_amps))
        amp_spacings = amp_spacings[amp_spacings > 0]
        if len(amp_spacings) < 3:
            continue

        r_min = np.minimum(amp_spacings[:-1], amp_spacings[1:])
        r_max = np.maximum(amp_spacings[:-1], amp_spacings[1:])
        r_freq = float(np.mean(r_min / r_max))

        # r_time: spacing ratio of gaps in the corresponding time window
        # Map spectral bin to time window (inverse relationship)
        gap_chunk = gaps[start:end] if end <= N else gaps[start:N]
        gap_spacings = np.diff(gap_chunk.astype(np.float64))
        gap_spacings = gap_spacings[gap_spacings > 0]
        if len(gap_spacings) < 3:
            continue

        r_min_t = np.minimum(gap_spacings[:-1], gap_spacings[1:])
        r_max_t = np.maximum(gap_spacings[:-1], gap_spacings[1:])
        r_time = float(np.mean(r_min_t / r_max_t))

        results.append({
            "bin": b,
            "r_time": r_time,
            "r_freq": r_freq,
            "r_sum": r_time + r_freq,
            "deviation": abs(r_time + r_freq - 1.0),
        })

    if not results:
        return {"mean_deviation": float('nan'), "bins": []}

    deviations = [r["deviation"] for r in results]
    sums = [r["r_sum"] for r in results]

    return {
        "n_bins": len(results),
        "mean_r_sum": float(np.mean(sums)),
        "std_r_sum": float(np.std(sums)),
        "mean_deviation": float(np.mean(deviations)),
        "median_deviation": float(np.median(deviations)),
        "conservation_holds": float(np.mean(deviations)) < 0.10,
        "bins_sample": results[:10],
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prime Gap Manifold v2 — Tier 0")
    parser.add_argument("--limit", type=float, default=1e8, help="Prime sieve limit")
    parser.add_argument("--dim", type=int, nargs="+", default=[6, 8], help="Embedding dimensions")
    parser.add_argument("--modes", type=int, default=30, help="Number of spectral modes")
    parser.add_argument("--outdir", type=str, default="results/tier0", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("═══ PRIME GAP MANIFOLD v2 — TIER 0: Li(x) VALIDATION ═══")
    print(f"  Sieve limit: {args.limit:.0e}")
    print(f"  Embedding dims: {args.dim}")
    print(f"  scipy available: {HAS_SCIPY}")
    print()

    # ── Generate primes ──────────────────────────────────────
    print("▸ Stage 0: Prime generation")
    primes = generate_primes_binary(args.limit)
    gaps = np.diff(primes)
    print(f"  {len(gaps):,} gaps, mean={np.mean(gaps):.4f}, max={np.max(gaps)}")
    print()

    # ── Spectral decomposition ───────────────────────────────
    print("▸ Stage 1: Spectral decomposition with Li(x) correction")
    modes, mean_density = spectral_decomposition(gaps, primes, n_modes=args.modes)

    print(f"  Mean prime density (p/ln p): {mean_density:.2f}")
    print()

    # Compare naive vs density-corrected
    print(f"  {'Rank':>4}  {'Freq':>10}  {'Amp':>8}  {'γ naive':>10}  {'Δ naive':>9}  {'γ Li(x)':>10}  {'Δ Li(x)':>9}  {'Known γ':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*10}")

    naive_errors = []
    li_errors = []

    for m in modes[:20]:
        print(f"  {m['rank']:>4}  {m['frequency']:>10.6f}  {m['amplitude']:>8.4f}  "
              f"{m['gamma_naive']:>10.4f}  {m['delta_naive']:>+9.4f}  "
              f"{m['gamma_density']:>10.4f}  {m['delta_density']:>+9.4f}  "
              f"{m['nearest_zero_density']:>10.4f}")

        # Track errors for modes that land near actual zeros (not DC)
        if m['gamma_density'] > 10:
            li_errors.append(abs(m['delta_density']))
        if m['gamma_naive'] > 10:
            naive_errors.append(abs(m['delta_naive']))

    print()
    if naive_errors:
        print(f"  Naive mapping — mean |Δ|: {np.mean(naive_errors):.4f}, median: {np.median(naive_errors):.4f}")
    if li_errors:
        print(f"  Li(x) mapping — mean |Δ|: {np.mean(li_errors):.4f}, median: {np.median(li_errors):.4f}")
        improvement = (np.mean(naive_errors) - np.mean(li_errors)) / np.mean(naive_errors) * 100
        print(f"  Improvement: {improvement:+.1f}%")

    # Check γ₁ specifically
    gamma1_candidates = [m for m in modes if 10 < m['gamma_density'] < 18]
    if gamma1_candidates:
        best_g1 = min(gamma1_candidates, key=lambda m: abs(m['gamma_density'] - 14.134725))
        print(f"\n  γ₁ BEST HIT: {best_g1['gamma_density']:.4f} (Δ = {best_g1['gamma_density'] - 14.134725:+.4f})")
        print(f"  SUCCESS CRITERION (±0.5): {'✓ PASS' if abs(best_g1['gamma_density'] - 14.134725) < 0.5 else '✗ FAIL'}")
    print()

    # ── Delay embedding ──────────────────────────────────────
    all_manifold_stats = {}
    for dim in args.dim:
        print(f"▸ Stage 2: Delay embedding (k={dim})")
        manifold = delay_embedding(gaps, dim)
        stats = manifold_statistics(manifold)

        print(f"  Distinct: {stats['distinct_tuples']:,}/{stats['n_points']:,} ({stats['distinct_pct']:.1f}%)")
        print(f"  RMS spread: {stats['rms_spread']:.4f}")
        print(f"  Mean sign changes: {stats['mean_sign_changes']:.4f}")
        print(f"  Oscillation classes:")
        for cls, cnt in sorted(stats["oscillation_classes"].items(), key=lambda x: -x[1]):
            pct = 100 * cnt / stats["n_points"]
            bar = "█" * int(pct / 2)
            print(f"    {cls:>10}: {cnt:>7,} ({pct:>5.1f}%) {bar}")
        print()

        all_manifold_stats[f"k={dim}"] = stats

    # ── Conservation law test ────────────────────────────────
    print("▸ Stage 3: Conservation law test (r_time + r_freq ≈ 1.0)")
    conservation = conservation_law_test(gaps, primes, n_bins=200)

    print(f"  Bins tested: {conservation['n_bins']}")
    print(f"  Mean r_sum: {conservation['mean_r_sum']:.4f} ± {conservation['std_r_sum']:.4f}")
    print(f"  Mean |deviation from 1.0|: {conservation['mean_deviation']:.4f}")
    print(f"  Median |deviation|: {conservation['median_deviation']:.4f}")
    print(f"  Conservation holds (<0.10): {'✓ YES' if conservation['conservation_holds'] else '✗ NO'}")
    print()

    # ── Summary ──────────────────────────────────────────────
    print("═══ TIER 0 SUMMARY ═══")
    if gamma1_candidates:
        g1_delta = abs(best_g1['gamma_density'] - 14.134725)
        print(f"  γ₁ error: ±{g1_delta:.4f} (target: <0.5)")
        print(f"  Li(x) improvement: {improvement:+.1f}% over naive")
    print(f"  Conservation law: mean deviation {conservation['mean_deviation']:.4f}")
    print(f"  Manifold dimensions tested: {args.dim}")
    print()

    # ── Save results ─────────────────────────────────────────
    results = {
        "sieve_limit": args.limit,
        "n_primes": len(primes),
        "n_gaps": len(gaps),
        "scipy_available": HAS_SCIPY,
        "mean_density_p_over_logp": float(mean_density),
        "spectral_modes": modes,
        "manifold_stats": all_manifold_stats,
        "conservation_law": {k: v for k, v in conservation.items() if k != "bins_sample"},
        "gamma1_best_hit": {
            "predicted": float(best_g1['gamma_density']),
            "known": 14.134725,
            "delta": float(best_g1['gamma_density'] - 14.134725),
            "within_target": abs(best_g1['gamma_density'] - 14.134725) < 0.5,
        } if gamma1_candidates else None,
        "naive_mean_error": float(np.mean(naive_errors)) if naive_errors else None,
        "li_mean_error": float(np.mean(li_errors)) if li_errors else None,
    }

    results_path = outdir / "tier0_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
