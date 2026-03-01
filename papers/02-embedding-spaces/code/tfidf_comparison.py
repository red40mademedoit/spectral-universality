#!/usr/bin/env python3
"""
Encoding Invariance Test: TF-IDF vs Learned Embeddings
=======================================================

Core question: Does the RMT signature come from the content or the encoder?

Test: Apply TF-IDF (no learned semantics) to the same 747 arXiv abstracts
used in the Nomic Embed analysis. Compare per-cluster ⟨r⟩ values.

If TF-IDF produces same RMT classes → structure is in co-occurrence statistics
If TF-IDF produces different classes → structure is in learned representations

Controls for γ = N/D:
  - TF-IDF raw: D = vocab size (~5K-20K), γ << 1
  - TF-IDF + SVD(768): D = 768, γ ≈ 0.97 (matches Nomic)
  - TF-IDF + SVD(100): D = 100, γ ≈ 7.5 (very different regime)

Authors: Shadow + Dreadbot 3.2.666
Date: 2026-02-28
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

# ── RMT Utilities (inline to avoid import issues) ───────────

def spacing_ratios(eigs):
    """Spacing ratios from sorted-descending eigenvalues."""
    eigs = np.sort(eigs)  # ascending
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return np.array([])
    return np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])


def mean_r(eigs):
    r = spacing_ratios(eigs)
    return float(np.mean(r)) if len(r) > 0 else None


def classify(r_val):
    if r_val is None:
        return "N/A"
    if r_val < 0.45:
        return "Poisson"
    elif r_val < 0.50:
        return "~Poisson"
    elif r_val < 0.57:
        return "GOE"
    elif r_val < 0.60:
        return "~GUE"
    else:
        return "GUE"


# ── Nomic reference values from Paper 2 (K=10) ──────────────

NOMIC_K10 = {
    'C00': {'cat': 'hep-th (pure)',         'r': 0.627, 'regime': 'GUE'},
    'C01': {'cat': 'cs.LG',                 'r': 0.520, 'regime': 'GOE'},
    'C02': {'cat': 'quant-ph',              'r': 0.473, 'regime': 'Poisson'},
    'C03': {'cat': 'cond-mat.stat-mech',    'r': 0.548, 'regime': 'GOE'},
    'C04': {'cat': 'physics.bio-ph',        'r': 0.515, 'regime': 'GOE'},
    'C05': {'cat': 'nlin.CD',               'r': 0.510, 'regime': 'GOE'},
    'C06': {'cat': 'math-ph',               'r': 0.468, 'regime': 'Poisson'},
    'C07': {'cat': 'math.SP',               'r': 0.570, 'regime': 'GUE'},
    'C08': {'cat': 'math.NT/stat.ML',       'r': 0.487, 'regime': 'Poisson'},
    'C09': {'cat': 'hep-th/math-ph (mixed)','r': 0.500, 'regime': 'GOE'},
}


def run_spectral_analysis(X, docs, K, label=""):
    """Full spectral analysis pipeline on embedding matrix X."""
    N, D = X.shape
    gamma = N / D

    # Center and normalize
    X_c = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_c = X_c / norms

    # Gram matrix
    G = X_c @ X_c.T
    eigenvalues = np.linalg.eigvalsh(G)[::-1]

    # MP edge
    lambda_plus = (1 + np.sqrt(gamma))**2
    signal_eigs = eigenvalues[eigenvalues > lambda_plus]
    n_signal = len(signal_eigs)

    # Global ⟨r⟩
    r_global = mean_r(eigenvalues[:n_signal]) if n_signal > 2 else None

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  N={N}, D={D}, γ={gamma:.3f}")
    print(f"  MP edge λ+ = {lambda_plus:.3f}, signal eigenvalues: {n_signal}")
    print(f"  Global ⟨r⟩ (signal): {r_global:.4f}" if r_global else "  Global ⟨r⟩: N/A")

    # Spectral clustering
    eigvals_full, eigvecs_full = np.linalg.eigh(G)
    idx_sorted = np.argsort(eigvals_full)[::-1]
    top_K_vecs = eigvecs_full[:, idx_sorted[:K]]
    row_norms = np.linalg.norm(top_K_vecs, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    top_K_vecs = top_K_vecs / row_norms

    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = km.fit_predict(top_K_vecs)

    # Per-cluster analysis
    results = []
    print(f"\n  {'ID':<6} {'Top Category':<25} {'N':>4} {'Coh':>7} {'⟨r⟩':>7} {'Regime':<10}")
    print(f"  {'-'*65}")

    for c in range(K):
        mask = labels == c
        n_c = mask.sum()
        if n_c < 5:
            results.append({'cluster': c, 'n': n_c, 'r': None, 'coherence': None,
                            'top_cat': 'too_small', 'regime': 'N/A'})
            continue

        G_c = X_c[mask] @ X_c[mask].T
        eigs_c = np.linalg.eigvalsh(G_c)[::-1]

        # Coherence
        cosines = G_c[np.triu_indices(n_c, k=1)]
        coherence = float(np.mean(cosines))

        r_c = mean_r(eigs_c)
        regime = classify(r_c)

        cat_dist = Counter(docs[i]["query_category"] for i in np.where(mask)[0])
        top_cat = cat_dist.most_common(1)[0]

        results.append({
            'cluster': c, 'n': int(n_c), 'r': float(r_c) if r_c else None,
            'coherence': coherence, 'top_cat': top_cat[0],
            'top_cat_count': top_cat[1], 'regime': regime,
            'category_dist': dict(cat_dist),
        })

        print(f"  C{c:02d}   {top_cat[0]+'('+str(top_cat[1])+')':25} {n_c:4d} "
              f"{coherence:7.3f} {r_c:7.3f} {regime:<10}" if r_c else
              f"  C{c:02d}   {top_cat[0]:25} {n_c:4d}     N/A     N/A")

    return {
        'label': label,
        'N': N, 'D': D, 'gamma': gamma,
        'mp_edge': float(lambda_plus),
        'n_signal': n_signal,
        'r_global': r_global,
        'K': K,
        'clusters': results,
        'labels': labels.tolist(),
    }


def main():
    corpus_path = Path.home() / "papers" / "spectral-taxonomy" / "arxiv_corpus.json"
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        sys.exit(1)

    with open(corpus_path) as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents")

    texts = [d['text'] for d in docs]
    K = 10  # Match paper

    # ── Encoding 1: TF-IDF raw ───────────────────────────────
    print("\nBuilding TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,
    )
    X_tfidf_raw = tfidf.fit_transform(texts).toarray()
    print(f"  TF-IDF shape: {X_tfidf_raw.shape} (vocab={len(tfidf.vocabulary_)})")

    result_tfidf_raw = run_spectral_analysis(X_tfidf_raw, docs, K,
                                              label="TF-IDF (raw, D=vocab)")

    # ── Encoding 2: TF-IDF + SVD(768) — match Nomic D ───────
    print("\nReducing TF-IDF to 768 dimensions (match Nomic)...")
    svd_768 = TruncatedSVD(n_components=768, random_state=42)
    X_tfidf_768 = svd_768.fit_transform(tfidf.fit_transform(texts))
    var_explained = svd_768.explained_variance_ratio_.sum()
    print(f"  Shape: {X_tfidf_768.shape}, variance explained: {var_explained:.3f}")

    result_tfidf_768 = run_spectral_analysis(X_tfidf_768, docs, K,
                                              label="TF-IDF + SVD(768) [γ-matched to Nomic]")

    # ── Encoding 3: TF-IDF + SVD(100) — different γ regime ──
    print("\nReducing TF-IDF to 100 dimensions...")
    svd_100 = TruncatedSVD(n_components=100, random_state=42)
    X_tfidf_100 = svd_100.fit_transform(tfidf.fit_transform(texts))
    var_100 = svd_100.explained_variance_ratio_.sum()
    print(f"  Shape: {X_tfidf_100.shape}, variance explained: {var_100:.3f}")

    result_tfidf_100 = run_spectral_analysis(X_tfidf_100, docs, K,
                                              label="TF-IDF + SVD(100) [high-γ regime]")

    # ── Encoding 4: NMF ──────────────────────────────────────
    from sklearn.decomposition import NMF
    print("\nNon-negative Matrix Factorization (100 topics)...")
    X_tfidf_sparse = tfidf.fit_transform(texts)
    nmf = NMF(n_components=100, random_state=42, max_iter=500)
    X_nmf = nmf.fit_transform(X_tfidf_sparse)
    print(f"  Shape: {X_nmf.shape}")

    result_nmf = run_spectral_analysis(X_nmf, docs, K,
                                        label="NMF (100 topics)")

    # ── Encoding 5: Random projection baseline ───────────────
    print("\nRandom Gaussian projection (768D)...")
    rng = np.random.RandomState(42)
    R = rng.randn(X_tfidf_raw.shape[1], 768) / np.sqrt(768)
    X_random = X_tfidf_raw @ R
    print(f"  Shape: {X_random.shape}")

    result_random = run_spectral_analysis(X_random, docs, K,
                                           label="Random Projection (768D) [control]")

    # ── Cross-method comparison ──────────────────────────────
    print("\n" + "="*65)
    print("  CROSS-METHOD COMPARISON")
    print("="*65)

    all_results = {
        'TF-IDF raw': result_tfidf_raw,
        'TF-IDF+SVD(768)': result_tfidf_768,
        'TF-IDF+SVD(100)': result_tfidf_100,
        'NMF(100)': result_nmf,
        'Random(768)': result_random,
    }

    # Compare ⟨r⟩ distributions across methods
    print(f"\n  {'Method':<22} {'γ':>6} {'Sig':>4} {'⟨r⟩_global':>10} {'⟨r⟩ range':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Nomic Embed (ref)':<22} {'0.97':>6} {'36':>4} {'0.496':>10} {'0.468-0.627':>15}")

    for name, res in all_results.items():
        r_vals = [c['r'] for c in res['clusters'] if c['r'] is not None]
        r_range = f"{min(r_vals):.3f}-{max(r_vals):.3f}" if r_vals else "N/A"
        r_g = f"{res['r_global']:.3f}" if res['r_global'] else "N/A"
        print(f"  {name:<22} {res['gamma']:>6.2f} {res['n_signal']:>4} {r_g:>10} {r_range:>15}")

    # Regime classification comparison
    print(f"\n  Regime counts per method:")
    print(f"  {'Method':<22} {'Poisson':>8} {'~Poisson':>9} {'GOE':>5} {'~GUE':>5} {'GUE':>5}")
    print(f"  {'-'*55}")

    nomic_regimes = Counter(v['regime'] for v in NOMIC_K10.values())
    print(f"  {'Nomic (paper ref)':<22} {nomic_regimes.get('Poisson',0):>8} "
          f"{'':>9} {nomic_regimes.get('GOE',0):>5} {'':>5} {nomic_regimes.get('GUE',0):>5}")

    for name, res in all_results.items():
        regimes = Counter(c['regime'] for c in res['clusters'])
        print(f"  {name:<22} {regimes.get('Poisson',0):>8} {regimes.get('~Poisson',0):>9} "
              f"{regimes.get('GOE',0):>5} {regimes.get('~GUE',0):>5} {regimes.get('GUE',0):>5}")

    # Cluster assignment agreement (ARI)
    from sklearn.metrics import adjusted_rand_score
    print(f"\n  Adjusted Rand Index (cluster agreement):")
    method_names = list(all_results.keys())
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            ari = adjusted_rand_score(
                all_results[method_names[i]]['labels'],
                all_results[method_names[j]]['labels']
            )
            print(f"    {method_names[i]} vs {method_names[j]}: ARI = {ari:.3f}")

    # Save everything
    outdir = Path(__file__).parent.parent / "data"
    outdir.mkdir(exist_ok=True)

    # Strip labels (large) for JSON
    for res in all_results.values():
        del res['labels']

    output = {
        'corpus': 'arXiv 747 abstracts (same as Paper 2)',
        'corpus_path': str(corpus_path),
        'nomic_reference': NOMIC_K10,
        'results': {k: v for k, v in all_results.items()},
    }

    outpath = outdir / "tfidf_comparison.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {outpath}")


if __name__ == '__main__':
    main()
