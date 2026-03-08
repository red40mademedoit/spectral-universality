#!/usr/bin/env python3
"""
Clauset-Shalizi-Newman KS p-values via Semi-Parametric Bootstrap
=================================================================
The `powerlaw` package gives KS distance D but not proper p-values.
CSN (2009) recommends: generate N_boot synthetic power-law samples,
fit each, compute KS distance, p-value = fraction with D >= D_observed.

This gives actual p-values for the power-law hypothesis, not just
model comparison (which is what distribution_compare does).

Uses the embeddings from 02_n_dependence.py (full_embeddings.npy).

Output: ks_pvalues.json
"""

import json
import time
import os
import numpy as np
from scipy import stats
from pathlib import Path
import powerlaw

OUTPUT_DIR = Path("results/ks_pvalues")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
EMBED_PATH = Path("results/n_dependence/full_embeddings.npy")
N_BOOTSTRAP = 500  # CSN recommends 2500+ but 500 is decent for a first pass
TAIL_FRAC = 0.30
SEED = 42


def alpha_ols(eigs, frac=0.30):
    eigs = np.sort(np.abs(eigs))[::-1]
    eigs = eigs[eigs > 0]
    k = max(int(len(eigs) * frac), 5)
    tail = eigs[:k]
    log_r = np.log(np.arange(1, len(tail) + 1))
    log_e = np.log(tail)
    slope, _, r_value, _, _ = stats.linregress(log_e, log_r)
    return abs(slope), r_value ** 2


# ─── Load ───
print("Loading corpus metadata...")
with open(CORPUS_PATH) as f:
    data = json.load(f)
docs = data["documents"]
categories_list = [d["primary_category"] for d in docs]
cat_names = sorted(set(categories_list))

print("Loading embeddings...")
if EMBED_PATH.exists():
    embeddings = np.load(EMBED_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")
else:
    print(f"ERROR: {EMBED_PATH} not found. Run 02_n_dependence.py first.")
    print("Attempting to embed now...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    texts = [f"search_document: {d['text']}" for d in docs]
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=True)
    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_PATH, embeddings)

D = embeddings.shape[1]

# ─── Per-category CSN KS p-values ───
print("\n" + "=" * 70)
print(f"CSN BOOTSTRAP KS P-VALUES (N_boot={N_BOOTSTRAP})")
print("=" * 70)

results = {}
rng = np.random.default_rng(SEED)

for cat in cat_names:
    cat_idx = [j for j, c in enumerate(categories_list) if c == cat]
    X = embeddings[cat_idx]
    N = len(X)

    # Center and normalize
    X_c = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X_c = X_c / norms

    G = X_c @ X_c.T
    eigs = np.linalg.eigvalsh(G)[::-1]
    eigs_pos = eigs[eigs > 0]

    k = max(int(len(eigs_pos) * TAIL_FRAC), 5)
    tail = eigs_pos[:k]

    # Fit power law
    fit = powerlaw.Fit(tail, discrete=False, verbose=False)
    alpha_fit = fit.power_law.alpha
    xmin_fit = fit.power_law.xmin
    D_obs = fit.power_law.D

    # OLS for comparison
    a_ols, r2 = alpha_ols(eigs_pos)

    # Compare vs lognormal and exponential
    R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R_tpl, p_tpl = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)

    # CSN bootstrap p-value
    # Generate synthetic power-law data with fitted params, compute KS each time
    n_exceed = 0
    for _ in range(N_BOOTSTRAP):
        # Generate synthetic power-law sample
        synthetic = powerlaw.Power_Law(xmin=xmin_fit, parameters=[alpha_fit]).generate_random(k)
        try:
            syn_fit = powerlaw.Fit(synthetic, discrete=False, verbose=False)
            if syn_fit.power_law.D >= D_obs:
                n_exceed += 1
        except:
            pass

    p_ks = n_exceed / N_BOOTSTRAP

    results[cat] = {
        "N_docs": N,
        "N_eigenvalues": len(eigs_pos),
        "tail_k": k,
        "alpha_mle": float(alpha_fit),
        "alpha_ols": float(a_ols),
        "r2_ols": float(r2),
        "xmin": float(xmin_fit),
        "ks_D": float(D_obs),
        "ks_p_csn": float(p_ks),
        "power_law_plausible": p_ks >= 0.10,  # CSN criterion
        "vs_lognormal": {"R": float(R_ln), "p": float(p_ln),
                         "winner": "power_law" if R_ln > 0 else "lognormal"},
        "vs_exponential": {"R": float(R_exp), "p": float(p_exp),
                           "winner": "power_law" if R_exp > 0 else "exponential"},
        "vs_truncated_pl": {"R": float(R_tpl), "p": float(p_tpl),
                            "winner": "power_law" if R_tpl > 0 else "truncated_pl"},
    }

    verdict = "✓ PLAUSIBLE" if p_ks >= 0.10 else "✗ REJECTED"
    ln_win = "PL" if R_ln > 0 else "LN"
    print(f"  {cat:<22} α_mle={alpha_fit:.3f}  KS_D={D_obs:.4f}  "
          f"p_CSN={p_ks:.3f} {verdict}  vs_LN: {ln_win}(p={p_ln:.3f})")

# ─── Summary ───
plausible = [c for c, v in results.items() if v["power_law_plausible"]]
rejected = [c for c, v in results.items() if not v["power_law_plausible"]]
print(f"\nPower-law plausible (p≥0.10): {len(plausible)}/{len(results)}")
print(f"  {plausible}")
print(f"Power-law rejected (p<0.10): {len(rejected)}/{len(results)}")
print(f"  {rejected}")

results["_summary"] = {
    "n_plausible": len(plausible),
    "n_rejected": len(rejected),
    "plausible_categories": plausible,
    "rejected_categories": rejected,
    "n_bootstrap": N_BOOTSTRAP,
    "tail_fraction": TAIL_FRAC,
    "criterion": "CSN (2009): p >= 0.10",
}

with open(OUTPUT_DIR / "ks_pvalues.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'ks_pvalues.json'}")
print("Done!")
