#!/usr/bin/env python3
"""
N-Dependence Curve for α
=========================
Embeds the full 10K arXiv corpus once, then subsamples at different N
to characterize how α depends on sample size.

Critical for paper: global α on 10K = 9.59, per-category on 200 = ~2.0.
This test maps the transition.

Output: n_dependence_results.json + eigenvalue .npy files
"""

import json
import time
import os
import numpy as np
from scipy import stats
from pathlib import Path

# ─── Config ───
CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
OUTPUT_DIR = Path("results/n_dependence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_VALUES = [50, 100, 200, 400, 800, 1600, 3200, 5000, 7500, 10000]
TAIL_FRACTIONS = [0.10, 0.20, 0.30]
N_BOOTSTRAP = 100
SEED = 42

rng = np.random.default_rng(SEED)


def alpha_ols(eigs, frac=0.30):
    """OLS log-log slope on top fraction of eigenvalues."""
    eigs = np.sort(np.abs(eigs))[::-1]
    eigs = eigs[eigs > 0]
    k = max(int(len(eigs) * frac), 5)
    tail = eigs[:k]
    log_r = np.log(np.arange(1, len(tail) + 1))
    log_e = np.log(tail)
    slope, _, r_value, _, _ = stats.linregress(log_e, log_r)
    return abs(slope), r_value ** 2


def alpha_mle(eigs, frac=0.30):
    """MLE power-law fit via powerlaw package."""
    import powerlaw
    eigs = np.sort(np.abs(eigs))[::-1]
    eigs = eigs[eigs > 0]
    k = max(int(len(eigs) * frac), 5)
    tail = eigs[:k]
    try:
        fit = powerlaw.Fit(tail, discrete=False, verbose=False)
        return fit.power_law.alpha, fit.power_law.D
    except:
        return None, None


def spacing_ratio(eigs):
    """Mean spacing ratio ⟨r⟩."""
    eigs = np.sort(np.real(eigs))
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return None
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(ratios))


def mp_edge(N, D, sigma2=1.0):
    """Marchenko-Pastur upper edge."""
    gamma = N / D
    return sigma2 * (1 + np.sqrt(gamma)) ** 2


# ─── Load corpus ───
print("Loading corpus...")
with open(CORPUS_PATH) as f:
    data = json.load(f)
docs = data["documents"]
print(f"Loaded {len(docs)} documents")

# ─── Embed full corpus ───
print("Loading Nomic Embed v2 MoE...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
print(f"Model loaded in {time.time()-t0:.1f}s")

texts = [f"search_document: {d['text']}" for d in docs]
print(f"Embedding {len(texts)} documents...")
t0 = time.time()
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True)
print(f"Embedded in {time.time()-t0:.1f}s, shape={embeddings.shape}")

D = embeddings.shape[1]
categories = [d["primary_category"] for d in docs]
cat_list = sorted(set(categories))

# Save full embeddings for reuse
np.save(OUTPUT_DIR / "full_embeddings.npy", embeddings)

# ─── N-dependence curve ───
print("\n" + "=" * 60)
print("N-DEPENDENCE CURVE")
print("=" * 60)

results = {"D": D, "categories": cat_list, "n_values": []}

for N in N_VALUES:
    print(f"\n--- N = {N} ---")

    if N >= len(docs):
        idx = np.arange(len(docs))
    else:
        # Stratified subsample
        idx = []
        per_cat = N // len(cat_list)
        remainder = N % len(cat_list)
        for i, cat in enumerate(cat_list):
            cat_idx = [j for j, c in enumerate(categories) if c == cat]
            n_take = per_cat + (1 if i < remainder else 0)
            n_take = min(n_take, len(cat_idx))
            chosen = rng.choice(cat_idx, size=n_take, replace=False)
            idx.extend(chosen)
        idx = np.array(idx)
        rng.shuffle(idx)

    X = embeddings[idx]
    N_actual = len(X)

    # Center and row-normalize
    X_c = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X_c = X_c / norms

    # Gram matrix
    G = X_c @ X_c.T
    eigs = np.linalg.eigvalsh(G)[::-1]

    # Save eigenvalues
    np.save(OUTPUT_DIR / f"eigs_N{N_actual}.npy", eigs)

    # MP edge
    lp = mp_edge(N_actual, D)
    signal_eigs = eigs[eigs > lp]
    n_signal = len(signal_eigs)

    # ⟨r⟩
    r_signal = spacing_ratio(signal_eigs) if n_signal > 3 else None
    r_full = spacing_ratio(eigs)

    # α at multiple tail fractions
    alpha_results = {}
    for frac in TAIL_FRACTIONS:
        a_ols, r2 = alpha_ols(eigs, frac)
        a_mle, ks = alpha_mle(eigs, frac)
        alpha_results[f"frac_{frac}"] = {
            "alpha_ols": float(a_ols),
            "r2": float(r2),
            "alpha_mle": float(a_mle) if a_mle else None,
            "ks_D": float(ks) if ks else None,
        }

    # Bootstrap α at 30%
    boot_alphas = []
    for _ in range(N_BOOTSTRAP):
        boot_idx = rng.choice(N_actual, size=N_actual, replace=True)
        G_boot = X_c[boot_idx] @ X_c[boot_idx].T
        eigs_boot = np.linalg.eigvalsh(G_boot)[::-1]
        a_boot, _ = alpha_ols(eigs_boot, 0.30)
        boot_alphas.append(a_boot)

    entry = {
        "N": N_actual,
        "D": D,
        "gamma": N_actual / D,
        "mp_edge": float(lp),
        "n_signal": n_signal,
        "r_signal": r_signal,
        "r_full": r_full,
        "dominant_eigenvalue": float(eigs[0]),
        "alpha": alpha_results,
        "bootstrap_alpha_30pct": {
            "mean": float(np.mean(boot_alphas)),
            "std": float(np.std(boot_alphas)),
            "ci_lo": float(np.percentile(boot_alphas, 2.5)),
            "ci_hi": float(np.percentile(boot_alphas, 97.5)),
        },
    }
    results["n_values"].append(entry)

    a30 = alpha_results["frac_0.3"]["alpha_ols"]
    r2_30 = alpha_results["frac_0.3"]["r2"]
    r_str = f"{r_signal:.4f}" if r_signal else "N/A"
    print(f"  N={N_actual:>5}  α(30%)={a30:.4f}  R²={r2_30:.4f}  "
          f"⟨r⟩_signal={r_str:>8}  "
          f"n_signal={n_signal}  λ₁={eigs[0]:.1f}")

# ─── Per-category α at fixed N=200 and N=400 ───
print("\n" + "=" * 60)
print("PER-CATEGORY α AT MATCHED N")
print("=" * 60)

for n_per_cat in [200, 400]:
    print(f"\n--- N per category = {n_per_cat} ---")
    cat_results = {}
    for cat in cat_list:
        cat_idx = [j for j, c in enumerate(categories) if c == cat]
        if len(cat_idx) < n_per_cat:
            n_take = len(cat_idx)
        else:
            n_take = n_per_cat
        chosen = rng.choice(cat_idx, size=n_take, replace=False)
        X_cat = embeddings[chosen]

        X_c = X_cat - X_cat.mean(axis=0)
        norms = np.linalg.norm(X_c, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        X_c = X_c / norms

        G = X_c @ X_c.T
        eigs = np.linalg.eigvalsh(G)[::-1]
        np.save(OUTPUT_DIR / f"eigs_cat_{cat}_N{n_take}.npy", eigs)

        a_ols, r2 = alpha_ols(eigs, 0.30)
        lp = mp_edge(n_take, D)
        sig_eigs = eigs[eigs > lp]
        r_sig = spacing_ratio(sig_eigs) if len(sig_eigs) > 3 else None

        cat_results[cat] = {
            "N": n_take,
            "alpha_ols_30pct": float(a_ols),
            "r2": float(r2),
            "r_signal": r_sig,
            "n_signal": len(sig_eigs),
        }
        r_s = f"{r_sig:.3f}" if r_sig else "N/A"
        print(f"  {cat:<22} N={n_take:>3}  α={a_ols:.3f}  R²={r2:.3f}  ⟨r⟩={r_s}")

    results[f"per_category_N{n_per_cat}"] = cat_results

# ─── Save ───
with open(OUTPUT_DIR / "n_dependence_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\nResults saved to {OUTPUT_DIR / 'n_dependence_results.json'}")
print("Done!")
