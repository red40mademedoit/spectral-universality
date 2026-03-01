#!/usr/bin/env python3
"""10^11 counting function — memory-efficient (no full prime array)"""
import primesieve, time, numpy as np, json, os
from scipy.special import expi
import cupy as cp

max_x = 1e11
M = 1 << 20
t_min, t_max = np.log(2), np.log(max_x)
t = np.linspace(t_min, t_max, M)
x = np.exp(t)
dt = t[1] - t[0]
print(f"10^11 counting fn, M={M:,}, gamma res={2*np.pi/(t_max-t_min):.4f}")

# Incremental pi(x) — no need to store 4.1B primes
print("Computing pi(x)...")
t0 = time.time()
pi_values = np.zeros(M, dtype=np.float64)
cumulative = 0
prev_x = 0
for i in range(M):
    xi = int(x[i])
    if xi > prev_x:
        cumulative += primesieve.count_primes(prev_x + 1, xi)
        prev_x = xi
    pi_values[i] = cumulative
    if (i+1) % 200000 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (M - i - 1)
        print(f"  [{i+1}/{M}] pi={cumulative:,}, {elapsed:.0f}s, ETA {eta:.0f}s")

print(f"  pi(1e11) = {int(pi_values[-1]):,} in {time.time()-t0:.1f}s")

# Li and detrend
li_values = expi(t) - expi(np.log(2.0))
diff = pi_values - li_values
signal = (diff * np.exp(-t/2) * t)
signal -= np.mean(signal)
print(f"  Signal RMS = {np.std(signal):.4f}")

# GPU FFT
sig_gpu = cp.asarray(signal)
spectrum = cp.fft.rfft(sig_gpu * cp.hanning(M))
amps = cp.asnumpy(cp.abs(spectrum))
gammas = 2 * np.pi * np.fft.rfftfreq(M, d=dt)
print(f"  FFT done")

ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320221, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087689, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846,
]

print()
header = f"  {'Zero':>6} {'gamma':>8} {'SNR':>7} {'Pk':>3} {'delta':>8}"
print(header)
print("  " + "-"*40)

results = []
n_snr3 = 0
n_snr5 = 0
n_peak = 0

for i, gamma in enumerate(ZEROS):
    k = int(np.argmin(np.abs(gammas - gamma)))
    if k < 1 or k >= len(amps):
        continue
    A = amps[k]
    lo, hi = max(1, k-500), min(len(amps), k+501)
    mask = np.ones(hi-lo, dtype=bool)
    c = k - lo
    mask[max(0, c-10):min(hi-lo, c+11)] = False
    local = amps[lo:hi][mask]
    med = np.median(local)
    mad = np.median(np.abs(local - med)) * 1.4826
    snr = (A - med) / mad if mad > 0 else 0

    pk = all(amps[k+d] <= A for d in [-2,-1,1,2] if 0 <= k+d < len(amps))

    best_k = k
    for d in range(-3, 4):
        kn = k + d
        if 0 < kn < len(amps) and amps[kn] > amps[best_k]:
            best_k = kn
    delta = gammas[best_k] - gamma

    pm = "^" if pk else " "
    sig = " **" if snr > 5 else (" !" if snr > 3 else "")
    if snr > 3: n_snr3 += 1
    if snr > 5: n_snr5 += 1
    if pk: n_peak += 1

    print(f"  g{i+1:<4} {gamma:>7.3f} {snr:>7.2f} {pm:>3} {delta:>+8.4f}{sig}")

    results.append({
        'zero_idx': i+1, 'gamma': gamma, 'snr': float(snr),
        'is_peak': pk, 'delta_gamma': float(delta), 'A_obs': float(A),
    })

print()
print(f"  SNR>3: {n_snr3}/50, SNR>5: {n_snr5}/50, Peaks: {n_peak}/50")
print(f"  Total: {time.time()-t0:.1f}s")

# Amplitude decay
gammas_known = np.array([r['gamma'] for r in results])
A_obs = np.array([r['A_obs'] for r in results])
expected = 1.0 / np.sqrt(0.25 + gammas_known**2)
scale = A_obs[0] / expected[0]
from scipy.stats import pearsonr
r_corr, p_corr = pearsonr(A_obs, expected * scale)
print(f"  Amplitude decay: r={r_corr:.4f}, p={p_corr:.2e}")

# Save
os.makedirs('results', exist_ok=True)
output = {
    'limit': 1e11, 'M': M,
    'gamma_resolution': float(2*np.pi/(t_max-t_min)),
    'n_snr3': n_snr3, 'n_snr5': n_snr5, 'n_peak': n_peak,
    'amplitude_decay_r': float(r_corr),
    'zeros': results,
}
with open('results/counting_fn_1e11.json', 'w') as f:
    json.dump(output, f, indent=2)
print("  Saved to results/counting_fn_1e11.json")
