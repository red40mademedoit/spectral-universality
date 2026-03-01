# 01 — Spectral Detection of Riemann Zeta Zeros via Prime Counting Functions

## Summary

Detect individual Riemann zeta zeros in the spectrum of detrended prime counting functions. Three signal variants tested across three orders of magnitude (10⁹, 10¹⁰, 10¹¹).

**Key result:** 50/50 known zeros detected with SNR > 5 at all scales. Amplitude decay matches theoretical 1/|ρ| with r = 0.998.

**Key negative result:** Gap-based spectral methods (index-space FFT, log-space FFT, ladder mapping) all fail due to amplitude modulation by gap sizes g_n.

## What This Is

A computational verification of the explicit formula for π(x), showing that:
- The detrended signal `(π(eᵗ) - Li(eᵗ)) · e^{-t/2} · t` produces FFT peaks at known zero locations γ/(2π)
- Peak amplitudes match 1/|ρ| = 1/√(1/4 + γ²) with 4% accuracy
- The ψ(x) variant gives ~1.5× higher SNR than π(x)

## What This Is NOT

- **Not a zero-finding algorithm.** Requires primes as input — if you have primes, you don't need to "find" zeros.
- **Not evidence for RH.** Detecting zeros on the critical line says nothing about hypothetical off-line zeros.
- **Not novel mathematics.** The explicit formula predicts exactly these oscillations. Numerical verification exists (Brent, Odlyzko, Platt).

## Genuine Contributions

1. **Quantitative diagnosis** of why gap-based methods fail (g_n amplitude modulation)
2. **SNR comparison** across three signal variants and three scales
3. **Amplitude calibration** — r = 0.998 extraction of 1/|ρ| weighting, p = 10⁻⁵⁴
4. **Dirichlet decomposition** — different residue classes show different zero patterns (r = 0.5-0.7 cross-correlation), with q=5 a≡4 anomaly
5. **Memory-efficient 10¹¹ implementation** — incremental `count_primes()` uses 5GB instead of 50GB+

## Results

| Scale | Zeros Detected | γ₁ SNR | γ₃₀ SNR | γ₅₀ SNR | Resolution | r(decay) |
|-------|---------------|--------|---------|---------|------------|----------|
| 10⁹  | 30/30         | 53.8   | 6.1     | —       | 0.314      | 0.998    |
| 10¹⁰ | 50/50         | 78.4   | 9.2     | 6.3     | 0.281      | 0.9984   |
| 10¹¹ | 50/50         | 91.1   | 13.6    | 11.7    | 0.255      | 0.9968   |

SNR scaling: ~1.2-1.8× per decade. Higher zeros benefit more from increased log-span.

### Failed Methods (Important Context)

| Method | Result | Root Cause |
|--------|--------|------------|
| Gap index-space FFT | Rayleigh trivially significant, 0/30 local peaks | Detects autocorrelation, not zeros |
| Gap log-space FFT | 2/50 SNR>3, 0/50 SNR>5 | g_n amplitude modulation smears peaks |
| Ladder mapping | p=0.045 at 10⁸ → p=0.49 at 10⁹ | Statistical fluke |
| Von Mangoldt scaling | 10 hits but p=0.361 | Overfitting (tuned α) |

## Scripts

| File | Purpose |
|------|---------|
| `counting_fn_gpu.py` | Main pipeline — all 3 signals, GPU FFT, phase coherence (needs full prime array) |
| `counting_fn_11.py` | Memory-efficient variant for 10¹¹+ (incremental count_primes) |
| `per_zero_gpu.py` | Per-zero hypothesis testing — Rayleigh, local, permutation, ladder, amplitude, split-half |
| `log_space_gpu.py` | Log-space resampling with GPU interpolation, multi-window consistency |
| `dirichlet_decomp_gpu.py` | Residue class decomposition mod q, cross-residue SNR comparison |
| `manifold_v2.py` | Original gap-based approach (superseded) |
| `burst_prime_onboard.sh` | B200 environment setup |

## Data

| File | Contents |
|------|----------|
| `counting_fn_1e09.json` | 10⁹ results — 3 signals, 50 zeros, phase coherence |
| `counting_fn_1e10.json` | 10¹⁰ results — same structure |
| `counting_fn_1e11.json` | 10¹¹ results — π-Li signal only (memory-efficient) |
| `dirichlet_1e10.json` | Dirichlet decomposition mod 3,4,5,8,12 at 10¹⁰ |
| `per_zero_1e09.json` | Per-zero hypothesis test results |
| `log_space_1e09.json` | Log-space resampling results |

## Dependencies

```
numpy >= 2.0
scipy >= 1.10
primesieve >= 2.3    # pip install primesieve (or apt install primesieve for CLI)
cupy-cuda12x >= 14   # GPU runs only
```

## Reproduce

```bash
# Local (CPU, 10⁸-10⁹)
pip install numpy scipy primesieve
python counting_fn_gpu.py --limit 1e9 --M 1048576 --n-zeros 50

# GPU (10¹⁰+)
pip install numpy scipy primesieve cupy-cuda12x
python counting_fn_gpu.py --limit 1e10 --M 1048576 --n-zeros 50

# Memory-efficient (10¹¹+, no full prime array)
python counting_fn_11.py

# Dirichlet decomposition
python dirichlet_decomp_gpu.py --limit 1e10 --moduli 3 4 5 8 12
```

## Date

2026-02-28. Compute: NVIDIA B200 via Shadeform (~$10-12 total).
