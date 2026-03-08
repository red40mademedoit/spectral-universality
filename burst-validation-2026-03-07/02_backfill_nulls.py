#!/usr/bin/env python3
"""
02_backfill_nulls.py — Burst validation for spectral paper
==========================================================

Addresses reviewer criticisms 1-9 from external review.
Runs on H200 burst instance. ~60-90 min total.

Tests:
  A. α under null controls (shuffled text) — make-or-break
  B. Bootstrap CIs for α (1000 resamples)
  C. Tail fraction sensitivity sweep (5%, 10%, 15%, 20%, 25%, 30%, 40%)
  D. Empirical MP edge validation (normalized random vectors)
  E. MLE vs OLS α comparison (Clauset-Shalizi-Newman)
  F. Gram (XXᵀ) vs Covariance (XᵀX) comparison
  G. Matryoshka α sweep (128/256/512/768d)
  H. Per-category R² (already have these, recompute with all methods)
  I. Average subword token count per document per model

Data requirements (uploaded by 01_upload_data.sh):
  ~/data/arxiv_spectral_corpus_10k.json   — 10K arXiv docs
  ~/data/controls/*.embed.jsonl           — shuffled texts (3 × 10K)

Outputs:
  ~/results/eigenvalues/     — raw .npy eigenvalue arrays
  ~/results/alpha_nulls/     — α under null controls
  ~/results/bootstrap/       — bootstrap CIs
  ~/results/tail_sweep/      — tail fraction sensitivity
  ~/results/matryoshka/      — per-dimension α
  ~/results/gram_vs_cov/     — XXᵀ vs XᵀX comparison
  ~/results/mp_validation/   — empirical vs theoretical MP edge
  ~/results/token_counts/    — subword counts per model
  ~/results/summary.json     — everything in one file
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import linalg
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# §0 — Configuration
# ══════════════════════════════════════════════════════════════

CORPUS_PATH = os.path.expanduser("~/data/arxiv_spectral_corpus_10k.json")
CONTROLS_DIR = os.path.expanduser("~/data/controls")
RESULTS_DIR = os.path.expanduser("~/results")

# Use 5K subset (200 per category) for speed — still sufficient
N_PER_CATEGORY = 200
SEED = 42
BOOTSTRAP_N = 1000
TAIL_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
MATRYOSHKA_DIMS = [128, 256, 512, 768]
LABEL_SHUFFLE_PERMUTATIONS = 1000

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════
# §1 — Core RMT Functions
# ══════════════════════════════════════════════════════════════

def spacing_ratios(eigenvalues):
    """Consecutive spacing ratios r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})."""
    eigs = np.sort(eigenvalues)
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return np.array([])
    r = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return r


def marchenko_pastur_edge(N, D, sigma_sq=1.0):
    """Theoretical MP upper edge: λ₊ = σ²(1 + √γ)² where γ = N/D."""
    gamma = N / D
    return sigma_sq * (1 + np.sqrt(gamma)) ** 2


def compute_gram_matrix(X):
    """Centered cosine Gram matrix G = X̃X̃ᵀ where X̃ is row-normalized and centered."""
    # Row-normalize (unit vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    # Center
    X_centered = X_norm - X_norm.mean(axis=0, keepdims=True)
    # Gram matrix
    G = X_centered @ X_centered.T
    return G


def compute_covariance_matrix(X):
    """Centered covariance matrix C = X̃ᵀX̃ / (N-1)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    X_centered = X_norm - X_norm.mean(axis=0, keepdims=True)
    C = (X_centered.T @ X_centered) / max(X_centered.shape[0] - 1, 1)
    return C


def alpha_ols(eigenvalues, tail_fraction=0.20):
    """
    OLS log-log fit on top tail_fraction of eigenvalues.
    Returns (alpha, r_squared).
    
    This is the method used in the Mar 6 burst (reviewer criticism #1).
    """
    eigs = np.sort(np.abs(eigenvalues))[::-1]
    n = len(eigs)
    k = max(int(n * tail_fraction), 5)
    
    sorted_eig = eigs[:k]
    ranks = np.arange(1, k + 1)
    
    log_r = np.log(ranks)
    log_e = np.log(sorted_eig)
    
    valid = np.isfinite(log_r) & np.isfinite(log_e)
    if valid.sum() < 3:
        return 0.0, 0.0
    
    # OLS: log(rank) = α⁻¹ · log(λ) + const
    # So slope = α⁻¹, alpha = 1/slope (if fitting log_r ~ log_e)
    # But original code fits: slope = polyfit(log_e, log_r, 1)[0] → α = |slope|
    coeffs = np.polyfit(log_e[valid], log_r[valid], 1)
    slope = coeffs[0]
    alpha = float(abs(slope))
    
    # R²
    predicted = np.polyval(coeffs, log_e[valid])
    ss_res = np.sum((log_r[valid] - predicted) ** 2)
    ss_tot = np.sum((log_r[valid] - np.mean(log_r[valid])) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return alpha, r_squared


def alpha_mle(eigenvalues, tail_fraction=0.20):
    """
    MLE power-law exponent using the Clauset-Shalizi-Newman method.
    Uses the `powerlaw` package if available, falls back to Hill estimator.
    Returns (alpha, xmin, ks_distance).
    """
    eigs = np.sort(np.abs(eigenvalues))[::-1]
    n = len(eigs)
    k = max(int(n * tail_fraction), 5)
    tail = eigs[:k]
    tail = tail[tail > 0]
    
    if len(tail) < 5:
        return 0.0, 0.0, 1.0
    
    try:
        import powerlaw
        fit = powerlaw.Fit(tail, discrete=False, xmin=float(tail[-1]), verbose=False)
        a = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)
        ks = float(fit.power_law.D)
        return a, xmin, ks
    except Exception:
        return alpha_hill(tail), float(tail[-1]), -1.0


def alpha_hill(eigenvalues, k=None):
    """Hill estimator for tail index."""
    sorted_eig = np.sort(np.abs(eigenvalues))[::-1]
    if k is None:
        k = max(len(sorted_eig) // 5, 5)
    if len(sorted_eig) < k + 1:
        k = len(sorted_eig) - 1
    if k < 2:
        return 0.0
    top_k = sorted_eig[:k]
    thresh = sorted_eig[k]
    if thresh <= 0:
        return 0.0
    logs = np.log(top_k / thresh)
    return float(k / np.sum(logs)) if np.sum(logs) > 0 else 0.0


def full_spectral_analysis(X, N=None, D=None, tail_fraction=0.20):
    """
    Complete spectral analysis on embedding matrix X (N×D).
    Returns dict with eigenvalues, ⟨r⟩, α (OLS + MLE + Hill), MP edge, etc.
    """
    if N is None:
        N = X.shape[0]
    if D is None:
        D = X.shape[1]
    
    G = compute_gram_matrix(X)
    eigenvalues = np.linalg.eigvalsh(G)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # MP edge
    gamma = N / D
    # Estimate σ² from bulk
    sigma_sq = np.median(eigenvalues) / (1 + gamma) if gamma > 0 else 1.0
    mp_edge = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    
    signal_eigs = eigenvalues[eigenvalues > mp_edge]
    n_signal = len(signal_eigs)
    
    # Spacing ratios
    r_all = spacing_ratios(eigenvalues)
    r_signal = spacing_ratios(signal_eigs) if n_signal > 2 else np.array([])
    
    # α — three methods
    a_ols, a_ols_r2 = alpha_ols(eigenvalues, tail_fraction)
    a_mle, a_xmin, a_ks = alpha_mle(eigenvalues, tail_fraction)
    a_hill = alpha_hill(eigenvalues)
    
    return {
        "N": N,
        "D": D,
        "gamma": gamma,
        "eigenvalues": eigenvalues,  # raw — save as .npy
        "n_signal": n_signal,
        "lambda_1": float(eigenvalues[0]),
        "mp_edge_theoretical": float(mp_edge),
        "r_mean": float(np.mean(r_all)) if len(r_all) > 0 else None,
        "r_signal_mean": float(np.mean(r_signal)) if len(r_signal) > 0 else None,
        "r_ci": bootstrap_ci(r_all) if len(r_all) > 10 else None,
        "alpha_ols": a_ols,
        "alpha_ols_r2": a_ols_r2,
        "alpha_mle": a_mle,
        "alpha_mle_xmin": a_xmin,
        "alpha_mle_ks": a_ks,
        "alpha_hill": a_hill,
        "tail_fraction": tail_fraction,
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    values = np.asarray(values)
    if len(values) < 3:
        return None
    boot_means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    lo = (1 - ci) / 2
    hi = 1 - lo
    return [float(np.percentile(boot_means, lo * 100)),
            float(np.percentile(boot_means, hi * 100))]


# ══════════════════════════════════════════════════════════════
# §2 — Data Loading
# ══════════════════════════════════════════════════════════════

def load_corpus(path, n_per_category=200):
    """Load arxiv corpus, subsample to n_per_category per category."""
    print(f"Loading corpus from {path}...")
    with open(path) as f:
        data = json.load(f)
    
    docs = data["documents"]
    print(f"  Total documents: {len(docs)}")
    
    # Group by category
    by_cat = {}
    for d in docs:
        cat = d["primary_category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(d)
    
    # Subsample
    sampled = {}
    for cat, cat_docs in sorted(by_cat.items()):
        if len(cat_docs) >= n_per_category:
            np.random.shuffle(cat_docs)
            sampled[cat] = cat_docs[:n_per_category]
        else:
            sampled[cat] = cat_docs
        print(f"  {cat}: {len(sampled[cat])}/{len(cat_docs)}")
    
    return sampled, data.get("metadata", {})


def load_control_texts(controls_dir):
    """Load shuffled control texts."""
    controls = {}
    for name in ["control_cross_shuffle", "control_intra_shuffle", "control_word_shuffle"]:
        path = os.path.join(controls_dir, f"{name}.embed.jsonl")
        if os.path.exists(path):
            docs = []
            with open(path) as f:
                for line in f:
                    docs.append(json.loads(line))
            controls[name] = docs
            print(f"  {name}: {len(docs)} texts")
    return controls


# ══════════════════════════════════════════════════════════════
# §3 — Embedding
# ══════════════════════════════════════════════════════════════

def get_embedder(dim=768):
    """Load Nomic v2 MoE embedder with specified Matryoshka dimension."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v2-moe",
        trust_remote_code=True,
    )
    # Matryoshka: truncate to desired dim
    model.truncate_dim = dim
    return model


def embed_texts(model, texts, batch_size=64, prefix="search_document: "):
    """Embed a list of texts. Returns (N, D) numpy array."""
    # Nomic requires task prefix
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# §4 — Test A: α Under Null Controls (THE MAKE-OR-BREAK)
# ══════════════════════════════════════════════════════════════

def test_alpha_nulls(model, corpus_by_cat, controls, results_dir):
    """
    Reviewer #6: Show α is NOT invariant to shuffling.
    
    For each null control:
      1. Embed the shuffled texts
      2. Compute per-category α
      3. Compare to real α
    
    If α is also invariant → central claim is dead.
    If α diverges → strong positive evidence.
    """
    print("\n" + "=" * 70)
    print("TEST A: α under null controls")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "alpha_nulls")
    os.makedirs(out_dir, exist_ok=True)
    
    # First: compute real α per category
    print("\n--- Computing real α per category ---")
    real_results = {}
    all_real_texts = []
    all_real_cats = []
    
    for cat, docs in sorted(corpus_by_cat.items()):
        texts = [d["text"] for d in docs]
        all_real_texts.extend(texts)
        all_real_cats.extend([cat] * len(texts))
    
    print(f"Embedding {len(all_real_texts)} real texts...")
    real_embeddings = embed_texts(model, all_real_texts)
    
    # Per-category analysis
    categories = sorted(corpus_by_cat.keys())
    cat_indices = {}
    idx = 0
    for cat in categories:
        n = len(corpus_by_cat[cat])
        cat_indices[cat] = (idx, idx + n)
        idx += n
    
    for cat in categories:
        start, end = cat_indices[cat]
        X_cat = real_embeddings[start:end]
        result = full_spectral_analysis(X_cat)
        real_results[cat] = {
            "alpha_ols": result["alpha_ols"],
            "alpha_ols_r2": result["alpha_ols_r2"],
            "alpha_mle": result["alpha_mle"],
            "alpha_hill": result["alpha_hill"],
            "r_signal_mean": result["r_signal_mean"],
            "n_signal": result["n_signal"],
            "lambda_1": result["lambda_1"],
        }
        # Save eigenvalues
        np.save(os.path.join(results_dir, "eigenvalues", f"real_{cat}.npy"),
                result["eigenvalues"])
    
    # Global real
    result_global = full_spectral_analysis(real_embeddings)
    real_results["_global"] = {
        "alpha_ols": result_global["alpha_ols"],
        "alpha_ols_r2": result_global["alpha_ols_r2"],
        "alpha_mle": result_global["alpha_mle"],
        "alpha_hill": result_global["alpha_hill"],
        "r_signal_mean": result_global["r_signal_mean"],
    }
    np.save(os.path.join(results_dir, "eigenvalues", "real_global.npy"),
            result_global["eigenvalues"])
    
    print(f"\nReal global α (OLS): {result_global['alpha_ols']:.4f}")
    print(f"Real global α (MLE): {result_global['alpha_mle']:.4f}")
    print(f"Real global α (Hill): {result_global['alpha_hill']:.4f}")
    
    # Now: each null control
    null_results = {}
    for control_name, control_docs in controls.items():
        print(f"\n--- {control_name} ---")
        
        # Match texts to categories (same ordering as real)
        ctrl_by_cat = {}
        for d in control_docs:
            cat = d["category"]
            if cat not in ctrl_by_cat:
                ctrl_by_cat[cat] = []
            ctrl_by_cat[cat].append(d["text"])
        
        # Subsample to match real corpus size
        ctrl_texts = []
        ctrl_cats = []
        for cat in categories:
            if cat in ctrl_by_cat:
                n_needed = cat_indices[cat][1] - cat_indices[cat][0]
                available = ctrl_by_cat[cat][:n_needed]
                ctrl_texts.extend(available)
                ctrl_cats.extend([cat] * len(available))
        
        print(f"  Embedding {len(ctrl_texts)} shuffled texts...")
        ctrl_embeddings = embed_texts(model, ctrl_texts)
        
        # Per-category
        ctrl_result = {}
        idx = 0
        for cat in categories:
            n = cat_indices[cat][1] - cat_indices[cat][0]
            if idx + n > len(ctrl_embeddings):
                break
            X_cat = ctrl_embeddings[idx:idx + n]
            result = full_spectral_analysis(X_cat)
            ctrl_result[cat] = {
                "alpha_ols": result["alpha_ols"],
                "alpha_ols_r2": result["alpha_ols_r2"],
                "alpha_mle": result["alpha_mle"],
                "alpha_hill": result["alpha_hill"],
                "r_signal_mean": result["r_signal_mean"],
                "n_signal": result["n_signal"],
                "lambda_1": result["lambda_1"],
            }
            np.save(os.path.join(results_dir, "eigenvalues", f"{control_name}_{cat}.npy"),
                    result["eigenvalues"])
            idx += n
        
        # Global
        result_global_ctrl = full_spectral_analysis(ctrl_embeddings)
        ctrl_result["_global"] = {
            "alpha_ols": result_global_ctrl["alpha_ols"],
            "alpha_ols_r2": result_global_ctrl["alpha_ols_r2"],
            "alpha_mle": result_global_ctrl["alpha_mle"],
            "alpha_hill": result_global_ctrl["alpha_hill"],
            "r_signal_mean": result_global_ctrl["r_signal_mean"],
        }
        np.save(os.path.join(results_dir, "eigenvalues", f"{control_name}_global.npy"),
                result_global_ctrl["eigenvalues"])
        
        null_results[control_name] = ctrl_result
        
        print(f"  {control_name} global α (OLS): {result_global_ctrl['alpha_ols']:.4f}")
        print(f"  {control_name} global α (MLE): {result_global_ctrl['alpha_mle']:.4f}")
    
    # Label shuffle null (no re-embedding — just permute category labels)
    print(f"\n--- Label shuffle null ({LABEL_SHUFFLE_PERMUTATIONS} permutations) ---")
    label_shuffle_alphas = []
    for perm_i in tqdm(range(LABEL_SHUFFLE_PERMUTATIONS), desc="Label shuffle"):
        # Shuffle category assignments, keep embeddings fixed
        shuffled_cats = np.random.permutation(all_real_cats)
        
        # Pick one random category and compute its α
        target_cat = categories[perm_i % len(categories)]
        mask = shuffled_cats == target_cat
        if mask.sum() < 10:
            continue
        X_perm = real_embeddings[mask]
        a, _ = alpha_ols(np.linalg.eigvalsh(compute_gram_matrix(X_perm)))
        label_shuffle_alphas.append({"category": target_cat, "alpha": a})
    
    null_results["label_shuffle"] = {
        "permutations": LABEL_SHUFFLE_PERMUTATIONS,
        "alphas": label_shuffle_alphas,
        "mean_alpha": float(np.mean([x["alpha"] for x in label_shuffle_alphas])),
        "std_alpha": float(np.std([x["alpha"] for x in label_shuffle_alphas])),
    }
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("TEST A RESULTS: α comparison (real vs nulls)")
    print("=" * 70)
    print(f"{'Category':<25} {'Real α':>10} {'CrossShuf':>10} {'IntraShuf':>10} {'WordShuf':>10}")
    print("-" * 70)
    for cat in categories:
        real_a = real_results[cat]["alpha_ols"]
        cross_a = null_results.get("control_cross_shuffle", {}).get(cat, {}).get("alpha_ols", "—")
        intra_a = null_results.get("control_intra_shuffle", {}).get(cat, {}).get("alpha_ols", "—")
        word_a = null_results.get("control_word_shuffle", {}).get(cat, {}).get("alpha_ols", "—")
        print(f"{cat:<25} {real_a:>10.4f} {cross_a:>10.4f} {intra_a:>10.4f} {word_a:>10.4f}")
    
    # Save
    output = {
        "real": real_results,
        "nulls": null_results,
        "verdict": "PLACEHOLDER — check if real α ≠ null α"
    }
    with open(os.path.join(out_dir, "alpha_null_comparison.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved to {out_dir}/alpha_null_comparison.json")
    return real_results, null_results, real_embeddings, cat_indices


# ══════════════════════════════════════════════════════════════
# §5 — Test B: Bootstrap CIs for α
# ══════════════════════════════════════════════════════════════

def test_bootstrap_alpha(real_embeddings, cat_indices, categories, results_dir):
    """
    Reviewer #3: α needs error bars. Run bootstrap on eigenvalues.
    """
    print("\n" + "=" * 70)
    print(f"TEST B: Bootstrap CIs for α ({BOOTSTRAP_N} resamples)")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for cat in tqdm(categories, desc="Bootstrap per category"):
        start, end = cat_indices[cat]
        X_cat = real_embeddings[start:end]
        N_cat = X_cat.shape[0]
        
        boot_alphas_ols = []
        boot_alphas_hill = []
        boot_r_means = []
        
        for _ in range(BOOTSTRAP_N):
            idx = np.random.randint(0, N_cat, size=N_cat)
            X_boot = X_cat[idx]
            G = compute_gram_matrix(X_boot)
            eigs = np.linalg.eigvalsh(G)
            
            a_ols, _ = alpha_ols(eigs)
            a_hill = alpha_hill(eigs)
            r = spacing_ratios(eigs)
            
            boot_alphas_ols.append(a_ols)
            boot_alphas_hill.append(a_hill)
            boot_r_means.append(float(np.mean(r)) if len(r) > 0 else 0)
        
        results[cat] = {
            "alpha_ols_mean": float(np.mean(boot_alphas_ols)),
            "alpha_ols_ci": [float(np.percentile(boot_alphas_ols, 2.5)),
                             float(np.percentile(boot_alphas_ols, 97.5))],
            "alpha_ols_std": float(np.std(boot_alphas_ols)),
            "alpha_hill_mean": float(np.mean(boot_alphas_hill)),
            "alpha_hill_ci": [float(np.percentile(boot_alphas_hill, 2.5)),
                              float(np.percentile(boot_alphas_hill, 97.5))],
            "r_mean_mean": float(np.mean(boot_r_means)),
            "r_mean_ci": [float(np.percentile(boot_r_means, 2.5)),
                          float(np.percentile(boot_r_means, 97.5))],
        }
    
    # Print summary
    print(f"\n{'Category':<25} {'α (OLS)':<12} {'95% CI':<25} {'α (Hill)':<12}")
    print("-" * 75)
    for cat in categories:
        r = results[cat]
        ci = r["alpha_ols_ci"]
        print(f"{cat:<25} {r['alpha_ols_mean']:<12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]   {r['alpha_hill_mean']:<12.4f}")
    
    with open(os.path.join(out_dir, "bootstrap_cis.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved to {out_dir}/bootstrap_cis.json")
    return results


# ══════════════════════════════════════════════════════════════
# §6 — Test C: Tail Fraction Sensitivity
# ══════════════════════════════════════════════════════════════

def test_tail_fraction(real_embeddings, cat_indices, categories, results_dir):
    """
    Reviewer #4: How sensitive is α to the tail fraction choice?
    Sweep from 5% to 40%.
    """
    print("\n" + "=" * 70)
    print(f"TEST C: Tail fraction sensitivity sweep ({TAIL_FRACTIONS})")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "tail_sweep")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for cat in tqdm(categories, desc="Tail sweep"):
        start, end = cat_indices[cat]
        X_cat = real_embeddings[start:end]
        G = compute_gram_matrix(X_cat)
        eigs = np.linalg.eigvalsh(G)
        
        cat_results = {}
        for tf in TAIL_FRACTIONS:
            a_ols, r2 = alpha_ols(eigs, tail_fraction=tf)
            a_mle, _, ks = alpha_mle(eigs, tail_fraction=tf)
            cat_results[str(tf)] = {
                "alpha_ols": a_ols,
                "r2": r2,
                "alpha_mle": a_mle,
                "ks": ks,
            }
        results[cat] = cat_results
    
    # Print summary for a few interesting categories
    print(f"\n{'Tail%':<10}", end="")
    highlight_cats = ["math.NT", "math.SP", "cs.CL", "astro-ph.CO", "quant-ph"]
    highlight_cats = [c for c in highlight_cats if c in categories]
    for cat in highlight_cats:
        print(f"{cat:<15}", end="")
    print()
    print("-" * (10 + 15 * len(highlight_cats)))
    
    for tf in TAIL_FRACTIONS:
        print(f"{tf:<10.0%}", end="")
        for cat in highlight_cats:
            a = results[cat][str(tf)]["alpha_ols"]
            print(f"{a:<15.4f}", end="")
        print()
    
    with open(os.path.join(out_dir, "tail_fraction_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_dir}/tail_fraction_sweep.json")
    return results


# ══════════════════════════════════════════════════════════════
# §7 — Test D: Empirical MP Edge Validation
# ══════════════════════════════════════════════════════════════

def test_mp_validation(results_dir):
    """
    Reviewer #8: Validate MP edge with normalized random vectors.
    After row-normalization, entries are NOT iid — does the MP edge hold?
    """
    print("\n" + "=" * 70)
    print("TEST D: Empirical MP edge validation")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "mp_validation")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for N, D in [(200, 768), (400, 768), (5000, 768), (200, 128), (200, 256)]:
        print(f"\n  N={N}, D={D}, γ={N/D:.3f}")
        
        empirical_edges = []
        theoretical = marchenko_pastur_edge(N, D)
        
        for trial in range(100):
            # Generate random Gaussian, then normalize rows (same as pipeline)
            X = np.random.randn(N, D).astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = X / norms
            X_centered = X_norm - X_norm.mean(axis=0, keepdims=True)
            G = X_centered @ X_centered.T
            eigs = np.linalg.eigvalsh(G)
            empirical_edges.append(float(eigs[-1]))  # max eigenvalue
        
        empirical_mean = float(np.mean(empirical_edges))
        empirical_std = float(np.std(empirical_edges))
        
        # Also compute what the effective σ² is
        # For unit-norm vectors in D dims, each component ≈ 1/√D
        # After centering, variance changes
        effective_sigma_sq = empirical_mean / (1 + np.sqrt(N/D))**2
        corrected_edge = effective_sigma_sq * (1 + np.sqrt(N/D))**2
        
        results[f"N{N}_D{D}"] = {
            "N": N,
            "D": D,
            "gamma": N/D,
            "theoretical_edge": theoretical,
            "empirical_edge_mean": empirical_mean,
            "empirical_edge_std": empirical_std,
            "ratio": empirical_mean / theoretical if theoretical > 0 else None,
            "effective_sigma_sq": effective_sigma_sq,
        }
        
        print(f"    Theoretical λ₊: {theoretical:.4f}")
        print(f"    Empirical λ₊:   {empirical_mean:.4f} ± {empirical_std:.4f}")
        print(f"    Ratio:          {empirical_mean/theoretical:.4f}")
    
    with open(os.path.join(out_dir, "mp_edge_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_dir}/mp_edge_validation.json")
    return results


# ══════════════════════════════════════════════════════════════
# §8 — Test F: Gram (XXᵀ) vs Covariance (XᵀX)
# ══════════════════════════════════════════════════════════════

def test_gram_vs_cov(real_embeddings, cat_indices, categories, results_dir):
    """
    Reviewer #7: Compare Gram matrix vs covariance matrix results.
    For N < D (200 < 768), Gram is natural. For N > D, covariance is cheaper.
    Non-zero eigenvalues should be identical (up to scaling) but MP edges differ.
    """
    print("\n" + "=" * 70)
    print("TEST F: Gram (XXᵀ) vs Covariance (XᵀX)")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "gram_vs_cov")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for cat in tqdm(categories, desc="Gram vs Cov"):
        start, end = cat_indices[cat]
        X_cat = real_embeddings[start:end]
        
        # Gram: N×N
        G = compute_gram_matrix(X_cat)
        eigs_gram = np.sort(np.linalg.eigvalsh(G))[::-1]
        
        # Covariance: D×D
        C = compute_covariance_matrix(X_cat)
        eigs_cov = np.sort(np.linalg.eigvalsh(C))[::-1]
        
        a_gram_ols, r2_gram = alpha_ols(eigs_gram)
        a_cov_ols, r2_cov = alpha_ols(eigs_cov)
        
        r_gram = spacing_ratios(eigs_gram)
        r_cov = spacing_ratios(eigs_cov[eigs_cov > 1e-10])  # remove zero eigs from D>N
        
        results[cat] = {
            "gram_alpha_ols": a_gram_ols,
            "gram_r2": r2_gram,
            "gram_r_mean": float(np.mean(r_gram)) if len(r_gram) > 0 else None,
            "gram_n_eigs": len(eigs_gram),
            "cov_alpha_ols": a_cov_ols,
            "cov_r2": r2_cov,
            "cov_r_mean": float(np.mean(r_cov)) if len(r_cov) > 0 else None,
            "cov_n_eigs": int((eigs_cov > 1e-10).sum()),
        }
    
    # Print
    print(f"\n{'Category':<25} {'α(Gram)':<12} {'α(Cov)':<12} {'Δ':<10} {'⟨r⟩(G)':<10} {'⟨r⟩(C)':<10}")
    print("-" * 80)
    for cat in categories:
        r = results[cat]
        delta = abs(r["gram_alpha_ols"] - r["cov_alpha_ols"])
        print(f"{cat:<25} {r['gram_alpha_ols']:<12.4f} {r['cov_alpha_ols']:<12.4f} {delta:<10.4f} "
              f"{r['gram_r_mean'] or 0:<10.4f} {r['cov_r_mean'] or 0:<10.4f}")
    
    with open(os.path.join(out_dir, "gram_vs_cov.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_dir}/gram_vs_cov.json")
    return results


# ══════════════════════════════════════════════════════════════
# §9 — Test G: Matryoshka α Sweep
# ══════════════════════════════════════════════════════════════

def test_matryoshka(corpus_by_cat, categories, results_dir):
    """
    Matryoshka dimension sweep: how does α behave at 128/256/512/768d?
    Requires re-embedding at each dimension.
    """
    print("\n" + "=" * 70)
    print(f"TEST G: Matryoshka α sweep ({MATRYOSHKA_DIMS})")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "matryoshka")
    os.makedirs(out_dir, exist_ok=True)
    
    # Collect all texts in category order
    all_texts = []
    cat_ranges = {}
    idx = 0
    for cat in categories:
        docs = corpus_by_cat[cat]
        texts = [d["text"] for d in docs]
        all_texts.extend(texts)
        cat_ranges[cat] = (idx, idx + len(texts))
        idx += len(texts)
    
    results = {}
    
    for dim in MATRYOSHKA_DIMS:
        print(f"\n--- Dimension: {dim}d ---")
        model = get_embedder(dim=dim)
        embeddings = embed_texts(model, all_texts)
        print(f"  Embedded: {embeddings.shape}")
        
        dim_results = {}
        for cat in categories:
            start, end = cat_ranges[cat]
            X_cat = embeddings[start:end]
            result = full_spectral_analysis(X_cat)
            dim_results[cat] = {
                "alpha_ols": result["alpha_ols"],
                "alpha_ols_r2": result["alpha_ols_r2"],
                "alpha_mle": result["alpha_mle"],
                "alpha_hill": result["alpha_hill"],
                "r_signal_mean": result["r_signal_mean"],
                "n_signal": result["n_signal"],
                "lambda_1": result["lambda_1"],
            }
        
        # Global
        result_global = full_spectral_analysis(embeddings)
        dim_results["_global"] = {
            "alpha_ols": result_global["alpha_ols"],
            "alpha_mle": result_global["alpha_mle"],
            "r_signal_mean": result_global["r_signal_mean"],
        }
        
        results[str(dim)] = dim_results
        print(f"  Global α (OLS): {result_global['alpha_ols']:.4f}")
        
        del model  # free GPU memory
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except:
            pass
    
    with open(os.path.join(out_dir, "matryoshka_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_dir}/matryoshka_sweep.json")
    return results


# ══════════════════════════════════════════════════════════════
# §10 — Test I: Subword Token Counts
# ══════════════════════════════════════════════════════════════

def test_token_counts(corpus_by_cat, categories, results_dir):
    """
    Reviewer #10: Report average token count per document per model.
    Disambiguates tokenization granularity effects on α.
    """
    print("\n" + "=" * 70)
    print("TEST I: Subword token counts per model")
    print("=" * 70)
    
    out_dir = os.path.join(results_dir, "token_counts")
    os.makedirs(out_dir, exist_ok=True)
    
    from transformers import AutoTokenizer
    
    models = [
        ("nomic-ai/nomic-embed-text-v2-moe", "nomic_v2_moe"),
        ("sentence-transformers/all-MiniLM-L6-v2", "minilm"),
    ]
    
    # Collect texts per category
    results = {}
    
    for model_name, label in models:
        print(f"\n--- {label} ({model_name}) ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  Failed to load tokenizer: {e}")
            continue
        
        model_results = {}
        for cat in categories:
            texts = [d["text"] for d in corpus_by_cat[cat]]
            token_counts = [len(tokenizer.encode(t)) for t in texts]
            model_results[cat] = {
                "mean_tokens": float(np.mean(token_counts)),
                "std_tokens": float(np.std(token_counts)),
                "min_tokens": int(np.min(token_counts)),
                "max_tokens": int(np.max(token_counts)),
                "median_tokens": float(np.median(token_counts)),
            }
        
        results[label] = model_results
        
        # Print summary
        all_means = [model_results[c]["mean_tokens"] for c in categories]
        print(f"  Overall mean tokens: {np.mean(all_means):.0f}")
        print(f"  Range: {np.min(all_means):.0f} — {np.max(all_means):.0f}")
    
    with open(os.path.join(out_dir, "token_counts.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_dir}/token_counts.json")
    return results


# ══════════════════════════════════════════════════════════════
# §11 — Main Orchestrator
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    
    print("=" * 70)
    print("BURST VALIDATION — Spectral Paper Review Tests")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    corpus_by_cat, metadata = load_corpus(CORPUS_PATH, N_PER_CATEGORY)
    categories = sorted(corpus_by_cat.keys())
    # Remove empty categories
    categories = [c for c in categories if len(corpus_by_cat[c]) >= N_PER_CATEGORY]
    print(f"\n{len(categories)} categories with ≥{N_PER_CATEGORY} docs")
    
    controls = load_control_texts(CONTROLS_DIR)
    
    # Load 768d model for main tests
    print("\nLoading Nomic v2 MoE (768d)...")
    model = get_embedder(dim=768)
    
    # ── Test A: α under null controls (THE MAIN EVENT) ──────
    real_results, null_results, real_embeddings, cat_indices = \
        test_alpha_nulls(model, corpus_by_cat, controls, RESULTS_DIR)
    
    # Checkpoint
    print("\n✓ CHECKPOINT: Test A complete, saving...")
    with open(os.path.join(RESULTS_DIR, "checkpoint_A.json"), "w") as f:
        json.dump({"status": "complete", "time": time.time() - t0}, f)
    
    # ── Test B: Bootstrap CIs ───────────────────────────────
    bootstrap_results = test_bootstrap_alpha(
        real_embeddings, cat_indices, categories, RESULTS_DIR)
    
    print("\n✓ CHECKPOINT: Test B complete")
    
    # ── Test C: Tail fraction sweep ─────────────────────────
    tail_results = test_tail_fraction(
        real_embeddings, cat_indices, categories, RESULTS_DIR)
    
    print("\n✓ CHECKPOINT: Test C complete")
    
    # ── Test D: Empirical MP edge ───────────────────────────
    mp_results = test_mp_validation(RESULTS_DIR)
    
    print("\n✓ CHECKPOINT: Test D complete")
    
    # ── Test F: Gram vs Covariance ──────────────────────────
    gram_cov_results = test_gram_vs_cov(
        real_embeddings, cat_indices, categories, RESULTS_DIR)
    
    print("\n✓ CHECKPOINT: Test F complete")
    
    # Free main embeddings before Matryoshka (GPU memory)
    del real_embeddings, model
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except:
        pass
    
    # ── Test G: Matryoshka sweep ────────────────────────────
    matryoshka_results = test_matryoshka(corpus_by_cat, categories, RESULTS_DIR)
    
    print("\n✓ CHECKPOINT: Test G complete")
    
    # ── Test I: Token counts ────────────────────────────────
    token_results = test_token_counts(corpus_by_cat, categories, RESULTS_DIR)
    
    print("\n✓ CHECKPOINT: Test I complete")
    
    # ── Final summary ───────────────────────────────────────
    elapsed = time.time() - t0
    
    summary = {
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "n_categories": len(categories),
        "n_per_category": N_PER_CATEGORY,
        "total_docs": N_PER_CATEGORY * len(categories),
        "bootstrap_n": BOOTSTRAP_N,
        "tail_fractions": TAIL_FRACTIONS,
        "matryoshka_dims": MATRYOSHKA_DIMS,
        "tests_completed": ["A_alpha_nulls", "B_bootstrap", "C_tail_sweep",
                           "D_mp_validation", "F_gram_vs_cov",
                           "G_matryoshka", "I_token_counts"],
    }
    
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"ALL TESTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print("=" * 70)
    
    # List output files
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {os.path.relpath(path, RESULTS_DIR):<50} {size:>10,} bytes")


if __name__ == "__main__":
    main()
