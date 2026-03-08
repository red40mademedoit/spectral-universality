#!/usr/bin/env python3
"""
02b_greek_validation.py — Greek dialect-quarantined spectral validation
=======================================================================

Runs on H200 burst instance. Designed to run AFTER or INSTEAD OF 02_backfill_nulls.py.
Shares the same core functions.

Tests:
  G1. Koine-only author comparison (drop non-Koine authors)
  G2. DDB temporal split — Ptolemaic vs Roman vs Late Antique
  G3. Register comparison within Koine — documentary vs literary papyri vs literary authors
  G4. α null controls on Greek (shuffle within dialect)
  G5. Bootstrap CIs on per-author α

Data requirements (uploaded by 01b_upload_greek.sh):
  ~/data/greek/ddb/           — extracted DDB papyri texts (.txt)
  ~/data/greek/dclp/          — extracted DCLP papyri texts (.txt)
  ~/data/greek/metadata_index.json  — text metadata with TM IDs
  ~/data/greek/hgv_metadata.jsonl   — HGV dates/provenance
  ~/data/greek/literary/      — literary author texts (from canonical-greekLit or pre-chunked)

Outputs:
  ~/results/greek/koine_authors/       — within-Koine author α comparison
  ~/results/greek/temporal_split/      — DDB by period
  ~/results/greek/register_comparison/ — documentary vs literary
  ~/results/greek/null_controls/       — shuffle tests
  ~/results/greek/bootstrap/           — per-author CIs
  ~/results/greek/eigenvalues/         — raw .npy files
"""

import json
import os
import sys
import time
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import linalg
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# §0 — Configuration
# ══════════════════════════════════════════════════════════════

DATA_DIR = os.path.expanduser("~/data/greek")
RESULTS_DIR = os.path.expanduser("~/results/greek")

SEED = 42
BOOTSTRAP_N = 1000
TAIL_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]
LABEL_SHUFFLE_PERMUTATIONS = 1000

# Minimum texts per group for spectral analysis
# Need at least ~5 for a meaningful Gram matrix eigenvalue spectrum
MIN_TEXTS = 5

# Max texts per group (subsample larger groups to keep Gram matrices tractable)
MAX_TEXTS_PER_GROUP = 2000

# Koine authors — auto-discovered from literary/ directory at runtime.
# These are the known non-Koine authors to quarantine (for labeled comparison only).
KOINE_AUTHORS = None  # populated at runtime from literary/ dir

NON_KOINE_AUTHORS = {
    "hippocrates": "Ionic",
    "herodotus": "Ionic",
    "demosthenes": "Attic",
    "lysias": "Attic",
    "aristotle": "Attic",
    "plato": "Attic",
    "xenophon": "Attic",
    "septuaginta": "Translation",
    "homer": "Epic",
    "hesiod": "Epic",
}

# Temporal period boundaries for DDB
PERIODS = {
    "ptolemaic": (-300, 0),
    "roman": (1, 300),
    "late_antique": (301, 650),
}

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════
# §1 — Core RMT Functions (shared with 02_backfill_nulls.py)
# ══════════════════════════════════════════════════════════════

def spacing_ratios(eigenvalues):
    eigs = np.sort(eigenvalues)
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 0]
    if len(spacings) < 2:
        return np.array([])
    return np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])


def compute_gram_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    X_centered = X_norm - X_norm.mean(axis=0, keepdims=True)
    return X_centered @ X_centered.T


def alpha_ols(eigenvalues, tail_fraction=0.20):
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
    coeffs = np.polyfit(log_e[valid], log_r[valid], 1)
    slope = coeffs[0]
    alpha = float(abs(slope))
    predicted = np.polyval(coeffs, log_e[valid])
    ss_res = np.sum((log_r[valid] - predicted) ** 2)
    ss_tot = np.sum((log_r[valid] - np.mean(log_r[valid])) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, r_squared


def alpha_mle(eigenvalues, tail_fraction=0.20):
    eigs = np.sort(np.abs(eigenvalues))[::-1]
    n = len(eigs)
    k = max(int(n * tail_fraction), 5)
    tail = eigs[:k]
    tail = tail[tail > 0]
    if len(tail) < 5:
        return 0.0, 0.0, 1.0
    try:
        import powerlaw
        fit = powerlaw.Fit(tail, discrete=False, xmin=tail[-1], verbose=False)
        return float(fit.power_law.alpha), float(fit.power_law.xmin), float(fit.power_law.D)
    except ImportError:
        return alpha_hill(tail), float(tail[-1]), -1.0


def alpha_hill(eigenvalues, k=None):
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


def bootstrap_ci(values, n_boot=1000, ci=0.95):
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


def full_spectral_analysis(X, tail_fraction=0.20):
    N, D = X.shape
    G = compute_gram_matrix(X)
    eigenvalues = np.sort(np.linalg.eigvalsh(G))[::-1]

    gamma = N / D
    sigma_sq = np.median(eigenvalues) / (1 + gamma) if gamma > 0 else 1.0
    mp_edge = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    signal_eigs = eigenvalues[eigenvalues > mp_edge]
    n_signal = len(signal_eigs)

    r_all = spacing_ratios(eigenvalues)
    r_signal = spacing_ratios(signal_eigs) if n_signal > 2 else np.array([])

    a_ols, a_ols_r2 = alpha_ols(eigenvalues, tail_fraction)
    a_mle, a_xmin, a_ks = alpha_mle(eigenvalues, tail_fraction)
    a_hill = alpha_hill(eigenvalues)

    return {
        "N": N, "D": D, "gamma": gamma,
        "eigenvalues": eigenvalues,
        "n_signal": n_signal,
        "lambda_1": float(eigenvalues[0]),
        "mp_edge": float(mp_edge),
        "r_mean": float(np.mean(r_all)) if len(r_all) > 0 else None,
        "r_signal_mean": float(np.mean(r_signal)) if len(r_signal) > 0 else None,
        "r_ci": bootstrap_ci(r_all) if len(r_all) > 10 else None,
        "alpha_ols": a_ols, "alpha_ols_r2": a_ols_r2,
        "alpha_mle": a_mle, "alpha_mle_xmin": a_xmin, "alpha_mle_ks": a_ks,
        "alpha_hill": a_hill,
        "tail_fraction": tail_fraction,
    }


# ══════════════════════════════════════════════════════════════
# §2 — Data Loading
# ══════════════════════════════════════════════════════════════

def load_papyri_texts(data_dir):
    """Load extracted papyri texts with metadata cross-reference."""
    print("Loading papyri texts...")

    # Load metadata index
    idx_path = os.path.join(data_dir, "metadata_index.json")
    with open(idx_path) as f:
        meta_idx = json.load(f)
    print(f"  Metadata index: {len(meta_idx)} entries")

    # Build TM → date lookup
    hgv_path = os.path.join(data_dir, "hgv_metadata.jsonl")
    tm_dates = {}
    with open(hgv_path) as f:
        for line in f:
            d = json.loads(line)
            tm = d.get("tm", "")
            ds = d.get("date_start", "")
            de = d.get("date_end", "")
            if tm and (ds or de):
                tm_dates[tm] = (ds, de)
    print(f"  HGV date entries: {len(tm_dates)}")

    # Load texts with dates
    texts = []
    for entry in tqdm(meta_idx, desc="Loading texts"):
        filepath = os.path.join(data_dir, entry["file"])
        if not os.path.exists(filepath):
            continue

        try:
            with open(filepath, encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            continue

        if len(text) < 20:  # skip trivially short
            continue

        # Parse date
        tm_id = entry.get("tm_id", "")
        year = None
        if tm_id in tm_dates:
            ds, de = tm_dates[tm_id]
            year_str = ds if ds else de
            try:
                if year_str.startswith("-"):
                    year = -int(year_str.lstrip("-").split("-")[0])
                else:
                    year = int(year_str.split("-")[0])
            except (ValueError, IndexError):
                pass

        collection = entry.get("collection", "")
        is_ddb = collection.startswith("ddb")

        # Determine period
        period = None
        if year is not None:
            for pname, (lo, hi) in PERIODS.items():
                if lo <= year <= hi:
                    period = pname
                    break

        texts.append({
            "text": text,
            "file": entry["file"],
            "collection": "ddb" if is_ddb else "dclp",
            "year": year,
            "period": period,
            "greek_chars": entry.get("greek_chars", len(text)),
            "tm_id": tm_id,
        })

    print(f"  Loaded {len(texts)} texts with ≥20 chars")

    # Stats
    by_collection = Counter(t["collection"] for t in texts)
    by_period = Counter(t["period"] for t in texts if t["period"])
    print(f"  DDB: {by_collection['ddb']}, DCLP: {by_collection['dclp']}")
    print(f"  Ptolemaic: {by_period.get('ptolemaic', 0)}, "
          f"Roman: {by_period.get('roman', 0)}, "
          f"Late Antique: {by_period.get('late_antique', 0)}")

    return texts


def load_literary_texts(data_dir):
    """
    Load literary author texts. Expects pre-organized directory:
    ~/data/greek/literary/{author_name}/*.txt
    """
    print("Loading literary texts...")
    literary_dir = os.path.join(data_dir, "literary")
    if not os.path.exists(literary_dir):
        print(f"  WARNING: {literary_dir} not found. Skipping literary texts.")
        return {}

    by_author = {}
    for author_dir in sorted(Path(literary_dir).iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        texts = []
        for txt_file in sorted(author_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8").strip()
                if len(text) >= 20:
                    texts.append({"text": text, "file": str(txt_file)})
            except Exception:
                continue
        if texts:
            by_author[author] = texts
            print(f"  {author}: {len(texts)} texts")

    return by_author


# ══════════════════════════════════════════════════════════════
# §3 — Embedding
# ══════════════════════════════════════════════════════════════

_model_cache = {}

def get_embedder(dim=768):
    if dim in _model_cache:
        return _model_cache[dim]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v2-moe",
        trust_remote_code=True,
    )
    model.truncate_dim = dim
    _model_cache[dim] = model
    return model


def embed_texts(model, texts, batch_size=64, prefix="search_document: "):
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_group(model, text_list, max_n=MAX_TEXTS_PER_GROUP):
    """Embed a list of text dicts, subsampling if needed."""
    if len(text_list) > max_n:
        np.random.shuffle(text_list)
        text_list = text_list[:max_n]
    raw_texts = [t["text"] for t in text_list]
    return embed_texts(model, raw_texts), text_list


# ══════════════════════════════════════════════════════════════
# §4 — Test G1: Koine-Only Author Comparison
# ══════════════════════════════════════════════════════════════

def test_koine_authors(model, literary_by_author, results_dir):
    """
    Hold dialect constant (Koine), compare across authors.
    If α still differentiates authors within Koine, it's measuring
    content/style, not dialect.
    """
    print("\n" + "=" * 70)
    print("TEST G1: Koine-only author comparison")
    print("=" * 70)

    out_dir = os.path.join(results_dir, "koine_authors")
    os.makedirs(out_dir, exist_ok=True)
    eig_dir = os.path.join(results_dir, "eigenvalues")
    os.makedirs(eig_dir, exist_ok=True)

    results = {}

    # All authors in literary/ are Koine (pre-quarantined by extraction script)
    # Any non-Koine would be in the NON_KOINE_AUTHORS dict (not uploaded)
    koine_available = sorted([a for a in literary_by_author if a not in NON_KOINE_AUTHORS])
    non_koine_available = sorted([a for a in literary_by_author if a in NON_KOINE_AUTHORS])

    if not koine_available:
        print("  No Koine literary authors found. Skipping.")
        return {}

    print(f"  Koine authors ({len(koine_available)}): {koine_available}")
    print(f"  Non-Koine (for comparison): {non_koine_available}")

    # Embed and analyze each Koine author
    for author in koine_available:
        texts = literary_by_author[author]
        if len(texts) < MIN_TEXTS:
            print(f"  {author}: only {len(texts)} texts, below minimum {MIN_TEXTS}. Skipping.")
            continue

        embeddings, used_texts = embed_group(model, texts)
        result = full_spectral_analysis(embeddings)

        results[author] = {
            "dialect": "Koine",
            "N": len(used_texts),
            "alpha_ols": result["alpha_ols"],
            "alpha_ols_r2": result["alpha_ols_r2"],
            "alpha_mle": result["alpha_mle"],
            "alpha_hill": result["alpha_hill"],
            "r_signal_mean": result["r_signal_mean"],
            "r_ci": result["r_ci"],
            "n_signal": result["n_signal"],
            "lambda_1": result["lambda_1"],
        }
        np.save(os.path.join(eig_dir, f"koine_{author}.npy"), result["eigenvalues"])
        print(f"  {author:<20} N={len(used_texts):>4}  α(OLS)={result['alpha_ols']:.4f}  "
              f"⟨r⟩={result['r_signal_mean'] or 0:.4f}  R²={result['alpha_ols_r2']:.4f}")

    # Also run non-Koine for comparison (clearly labeled)
    for author in non_koine_available:
        texts = literary_by_author[author]
        embeddings, used_texts = embed_group(model, texts)
        result = full_spectral_analysis(embeddings)

        dialect = NON_KOINE_AUTHORS[author]
        results[author] = {
            "dialect": dialect,
            "N": len(used_texts),
            "alpha_ols": result["alpha_ols"],
            "alpha_ols_r2": result["alpha_ols_r2"],
            "alpha_mle": result["alpha_mle"],
            "alpha_hill": result["alpha_hill"],
            "r_signal_mean": result["r_signal_mean"],
            "r_ci": result["r_ci"],
            "n_signal": result["n_signal"],
            "lambda_1": result["lambda_1"],
        }
        np.save(os.path.join(eig_dir, f"nkoine_{dialect}_{author}.npy"), result["eigenvalues"])
        print(f"  {author:<20} N={len(used_texts):>4}  α(OLS)={result['alpha_ols']:.4f}  "
              f"⟨r⟩={result['r_signal_mean'] or 0:.4f}  [{dialect}]")

    # Pooled Koine literary
    all_koine_texts = []
    for a in koine_available:
        all_koine_texts.extend(literary_by_author[a])
    if len(all_koine_texts) >= MIN_TEXTS:
        embeddings, used = embed_group(model, all_koine_texts)
        result = full_spectral_analysis(embeddings)
        results["_pooled_koine_literary"] = {
            "dialect": "Koine",
            "N": len(used),
            "alpha_ols": result["alpha_ols"],
            "alpha_ols_r2": result["alpha_ols_r2"],
            "alpha_mle": result["alpha_mle"],
            "r_signal_mean": result["r_signal_mean"],
        }
        np.save(os.path.join(eig_dir, "pooled_koine_literary.npy"), result["eigenvalues"])
        print(f"\n  Pooled Koine literary: N={len(used)}, α={result['alpha_ols']:.4f}")

    with open(os.path.join(out_dir, "koine_author_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ══════════════════════════════════════════════════════════════
# §5 — Test G2: DDB Temporal Split
# ══════════════════════════════════════════════════════════════

def test_temporal_split(model, papyri_texts, results_dir):
    """
    Same register (DDB documentary), same dialect (Koine), different centuries.
    Does α drift with language change over time?
    """
    print("\n" + "=" * 70)
    print("TEST G2: DDB temporal split (Koine across centuries)")
    print("=" * 70)

    out_dir = os.path.join(results_dir, "temporal_split")
    os.makedirs(out_dir, exist_ok=True)
    eig_dir = os.path.join(results_dir, "eigenvalues")

    # Filter to DDB with known period
    ddb_by_period = defaultdict(list)
    for t in papyri_texts:
        if t["collection"] == "ddb" and t["period"]:
            ddb_by_period[t["period"]].append(t)

    results = {}

    for period in ["ptolemaic", "roman", "late_antique"]:
        texts = ddb_by_period.get(period, [])
        if len(texts) < MIN_TEXTS:
            print(f"  {period}: only {len(texts)} texts, skipping")
            continue

        print(f"\n--- {period} ({len(texts)} texts, subsampling to {min(len(texts), MAX_TEXTS_PER_GROUP)}) ---")
        embeddings, used = embed_group(model, texts)
        result = full_spectral_analysis(embeddings)

        results[period] = {
            "N": len(used),
            "alpha_ols": result["alpha_ols"],
            "alpha_ols_r2": result["alpha_ols_r2"],
            "alpha_mle": result["alpha_mle"],
            "alpha_hill": result["alpha_hill"],
            "r_signal_mean": result["r_signal_mean"],
            "r_ci": result["r_ci"],
            "n_signal": result["n_signal"],
            "lambda_1": result["lambda_1"],
        }

        # Also run tail fraction sweep for each period
        eigs = result["eigenvalues"]
        tail_sweep = {}
        for tf in TAIL_FRACTIONS:
            a, r2 = alpha_ols(eigs, tail_fraction=tf)
            tail_sweep[str(tf)] = {"alpha_ols": a, "r2": r2}
        results[period]["tail_sweep"] = tail_sweep

        np.save(os.path.join(eig_dir, f"ddb_{period}.npy"), result["eigenvalues"])
        print(f"  {period:<15} N={len(used):>5}  α(OLS)={result['alpha_ols']:.4f}  "
              f"⟨r⟩={result['r_signal_mean'] or 0:.4f}  R²={result['alpha_ols_r2']:.4f}")

    # Print comparison
    print("\n=== Temporal α comparison (same register, same dialect) ===")
    print(f"{'Period':<15} {'N':>6} {'α(OLS)':>10} {'α(MLE)':>10} {'α(Hill)':>10} {'⟨r⟩':>10}")
    print("-" * 65)
    for period in ["ptolemaic", "roman", "late_antique"]:
        if period in results:
            r = results[period]
            print(f"{period:<15} {r['N']:>6} {r['alpha_ols']:>10.4f} {r['alpha_mle']:>10.4f} "
                  f"{r['alpha_hill']:>10.4f} {r['r_signal_mean'] or 0:>10.4f}")

    with open(os.path.join(out_dir, "temporal_split.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ══════════════════════════════════════════════════════════════
# §6 — Test G3: Register Comparison Within Koine
# ══════════════════════════════════════════════════════════════

def test_register_comparison(model, papyri_texts, literary_by_author, results_dir):
    """
    All Koine. Different registers:
    - DDB documentary (contracts, receipts, letters)
    - DCLP literary papyri (literary fragments on papyrus)
    - Literary authors (full works)

    If α differentiates these, it's measuring content/register, not dialect.
    """
    print("\n" + "=" * 70)
    print("TEST G3: Register comparison within Koine")
    print("=" * 70)

    out_dir = os.path.join(results_dir, "register_comparison")
    os.makedirs(out_dir, exist_ok=True)
    eig_dir = os.path.join(results_dir, "eigenvalues")

    results = {}
    groups = {}

    # DDB documentary (subsample — it's huge)
    ddb_texts = [t for t in papyri_texts if t["collection"] == "ddb"]
    if ddb_texts:
        groups["ddb_documentary"] = ddb_texts
        print(f"  DDB documentary: {len(ddb_texts)} texts")

    # DCLP literary papyri
    dclp_texts = [t for t in papyri_texts if t["collection"] == "dclp"]
    if dclp_texts:
        groups["dclp_literary_papyri"] = dclp_texts
        print(f"  DCLP literary papyri: {len(dclp_texts)} texts")

    # Pooled Koine literary authors (all authors in literary/ are Koine)
    koine_literary = []
    for author in sorted(literary_by_author.keys()):
        if author not in NON_KOINE_AUTHORS:
            koine_literary.extend(literary_by_author[author])
    if koine_literary:
        groups["koine_literary_authors"] = koine_literary
        print(f"  Koine literary authors: {len(koine_literary)} texts")

    for group_name, texts in groups.items():
        if len(texts) < MIN_TEXTS:
            print(f"  {group_name}: only {len(texts)} texts, skipping")
            continue

        print(f"\n--- {group_name} ---")
        embeddings, used = embed_group(model, texts)
        result = full_spectral_analysis(embeddings)

        results[group_name] = {
            "N": len(used),
            "alpha_ols": result["alpha_ols"],
            "alpha_ols_r2": result["alpha_ols_r2"],
            "alpha_mle": result["alpha_mle"],
            "alpha_hill": result["alpha_hill"],
            "r_signal_mean": result["r_signal_mean"],
            "r_ci": result["r_ci"],
            "n_signal": result["n_signal"],
            "lambda_1": result["lambda_1"],
        }
        np.save(os.path.join(eig_dir, f"register_{group_name}.npy"), result["eigenvalues"])
        print(f"  α(OLS)={result['alpha_ols']:.4f}  ⟨r⟩={result['r_signal_mean'] or 0:.4f}  "
              f"R²={result['alpha_ols_r2']:.4f}  N={len(used)}")

    # Print comparison
    print("\n=== Register comparison (all Koine) ===")
    print(f"{'Register':<30} {'N':>6} {'α(OLS)':>10} {'α(MLE)':>10} {'⟨r⟩':>10}")
    print("-" * 70)
    for g in ["ddb_documentary", "dclp_literary_papyri", "koine_literary_authors"]:
        if g in results:
            r = results[g]
            print(f"{g:<30} {r['N']:>6} {r['alpha_ols']:>10.4f} {r['alpha_mle']:>10.4f} "
                  f"{r['r_signal_mean'] or 0:>10.4f}")

    with open(os.path.join(out_dir, "register_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ══════════════════════════════════════════════════════════════
# §7 — Test G4: α Null Controls on Greek
# ══════════════════════════════════════════════════════════════

def test_greek_null_controls(model, papyri_texts, literary_by_author, results_dir):
    """
    Shuffle tests within Koine:
    1. Cross-register shuffle: mix DDB + DCLP + literary labels, keep embeddings
    2. Within-DDB period shuffle: permute period labels on DDB texts
    3. Within-literary author shuffle: permute author labels
    4. Word-order shuffle on a subset: re-embed shuffled text (GPU)
    """
    print("\n" + "=" * 70)
    print("TEST G4: Greek null controls")
    print("=" * 70)

    out_dir = os.path.join(results_dir, "null_controls")
    os.makedirs(out_dir, exist_ok=True)
    eig_dir = os.path.join(results_dir, "eigenvalues")

    results = {}

    # ── 4a. Cross-register label shuffle ────────────────────
    print("\n--- G4a: Cross-register label shuffle ---")

    # Build combined Koine corpus with register labels
    combined = []
    ddb_texts_sub = [t for t in papyri_texts if t["collection"] == "ddb"]
    if len(ddb_texts_sub) > 1000:
        np.random.shuffle(ddb_texts_sub)
        ddb_texts_sub = ddb_texts_sub[:1000]
    for t in ddb_texts_sub:
        combined.append({"text": t["text"], "label": "documentary"})

    dclp_texts_sub = [t for t in papyri_texts if t["collection"] == "dclp"]
    for t in dclp_texts_sub:
        combined.append({"text": t["text"], "label": "literary_papyri"})

    for author in sorted(literary_by_author.keys()):
        if author not in NON_KOINE_AUTHORS:
            for t in literary_by_author[author]:
                combined.append({"text": t["text"], "label": f"literary_{author}"})

    if len(combined) >= MIN_TEXTS:
        print(f"  Combined Koine corpus: {len(combined)} texts")
        all_texts_raw = [c["text"] for c in combined]
        all_labels = [c["label"] for c in combined]
        embeddings = embed_texts(model, all_texts_raw)

        # Real α per register
        label_set = sorted(set(all_labels))
        real_alphas = {}
        for lab in label_set:
            mask = np.array([l == lab for l in all_labels])
            if mask.sum() < 20:
                continue
            X = embeddings[mask]
            eigs = np.linalg.eigvalsh(compute_gram_matrix(X))
            a, r2 = alpha_ols(eigs)
            real_alphas[lab] = {"alpha": a, "r2": r2, "N": int(mask.sum())}

        # Shuffle labels N times, recompute α
        shuffle_alphas = defaultdict(list)
        for _ in tqdm(range(LABEL_SHUFFLE_PERMUTATIONS), desc="Label shuffle"):
            shuffled = np.random.permutation(all_labels)
            for lab in label_set:
                mask = shuffled == lab
                if mask.sum() < 20:
                    continue
                X = embeddings[mask]
                eigs = np.linalg.eigvalsh(compute_gram_matrix(X))
                a, _ = alpha_ols(eigs)
                shuffle_alphas[lab].append(a)

        # P-values
        pvalues = {}
        for lab in label_set:
            if lab in real_alphas and lab in shuffle_alphas:
                real_a = real_alphas[lab]["alpha"]
                null_dist = np.array(shuffle_alphas[lab])
                # Two-sided p-value
                p = float(np.mean(np.abs(null_dist - np.mean(null_dist)) >=
                                  np.abs(real_a - np.mean(null_dist))))
                pvalues[lab] = p
                print(f"  {lab:<25} real α={real_a:.4f}  null μ={np.mean(null_dist):.4f} "
                      f"± {np.std(null_dist):.4f}  p={p:.4f}")

        results["cross_register_shuffle"] = {
            "real": real_alphas,
            "null_mean": {k: float(np.mean(v)) for k, v in shuffle_alphas.items()},
            "null_std": {k: float(np.std(v)) for k, v in shuffle_alphas.items()},
            "p_values": pvalues,
            "n_permutations": LABEL_SHUFFLE_PERMUTATIONS,
        }
    else:
        print("  Not enough combined texts. Skipping.")

    # ── 4b. Word-order shuffle (re-embed, GPU) ──────────────
    print("\n--- G4b: Word-order shuffle (re-embed) ---")

    # Take a manageable subset: 500 DDB + available DCLP + Koine literary
    word_shuffle_texts = []
    for t in ddb_texts_sub[:500]:
        word_shuffle_texts.append({"text": t["text"], "label": "documentary"})
    for t in dclp_texts_sub:
        word_shuffle_texts.append({"text": t["text"], "label": "literary_papyri"})
    for author in sorted(literary_by_author.keys()):
        if author not in NON_KOINE_AUTHORS:
            for t in literary_by_author[author]:
                word_shuffle_texts.append({"text": t["text"], "label": f"literary_{author}"})

    if len(word_shuffle_texts) >= MIN_TEXTS:
        original_texts = [t["text"] for t in word_shuffle_texts]
        ws_labels = [t["label"] for t in word_shuffle_texts]

        # Shuffle words within each text
        shuffled_texts = []
        for text in original_texts:
            words = text.split()
            np.random.shuffle(words)
            shuffled_texts.append(" ".join(words))

        # Embed both
        print(f"  Embedding {len(original_texts)} original texts...")
        orig_emb = embed_texts(model, original_texts)
        print(f"  Embedding {len(shuffled_texts)} word-shuffled texts...")
        shuf_emb = embed_texts(model, shuffled_texts)

        # Compare per-register α
        ws_results = {"original": {}, "word_shuffled": {}}
        ws_label_set = sorted(set(ws_labels))

        for lab in ws_label_set:
            mask = np.array([l == lab for l in ws_labels])
            if mask.sum() < 20:
                continue

            eigs_orig = np.linalg.eigvalsh(compute_gram_matrix(orig_emb[mask]))
            eigs_shuf = np.linalg.eigvalsh(compute_gram_matrix(shuf_emb[mask]))

            a_orig, r2_orig = alpha_ols(eigs_orig)
            a_shuf, r2_shuf = alpha_ols(eigs_shuf)

            ws_results["original"][lab] = {"alpha": a_orig, "r2": r2_orig, "N": int(mask.sum())}
            ws_results["word_shuffled"][lab] = {"alpha": a_shuf, "r2": r2_shuf, "N": int(mask.sum())}

            print(f"  {lab:<25} orig α={a_orig:.4f}  shuf α={a_shuf:.4f}  Δ={abs(a_orig-a_shuf):.4f}")

        results["word_order_shuffle"] = ws_results
    else:
        print("  Not enough texts. Skipping.")

    with open(os.path.join(out_dir, "greek_null_controls.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ══════════════════════════════════════════════════════════════
# §8 — Test G5: Bootstrap CIs on Per-Author α
# ══════════════════════════════════════════════════════════════

def test_bootstrap_authors(model, literary_by_author, results_dir):
    """
    Bootstrap CIs for α on each author. Critical for small N authors.
    """
    print("\n" + "=" * 70)
    print(f"TEST G5: Bootstrap CIs for per-author α ({BOOTSTRAP_N} resamples)")
    print("=" * 70)

    out_dir = os.path.join(results_dir, "bootstrap")
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for author, texts in sorted(literary_by_author.items()):
        if len(texts) < 20:
            print(f"  {author}: only {len(texts)} texts, skipping bootstrap")
            continue

        embeddings, used = embed_group(model, texts)
        N = embeddings.shape[0]

        dialect = NON_KOINE_AUTHORS.get(author, "Koine")

        boot_alpha_ols = []
        boot_alpha_hill = []
        boot_r = []

        for _ in tqdm(range(BOOTSTRAP_N), desc=f"Bootstrap {author}", leave=False):
            idx = np.random.randint(0, N, size=N)
            X_boot = embeddings[idx]
            G = compute_gram_matrix(X_boot)
            eigs = np.linalg.eigvalsh(G)

            a, _ = alpha_ols(eigs)
            boot_alpha_ols.append(a)
            boot_alpha_hill.append(alpha_hill(eigs))
            r = spacing_ratios(eigs)
            boot_r.append(float(np.mean(r)) if len(r) > 0 else 0)

        results[author] = {
            "dialect": dialect,
            "N": N,
            "alpha_ols_mean": float(np.mean(boot_alpha_ols)),
            "alpha_ols_ci": [float(np.percentile(boot_alpha_ols, 2.5)),
                             float(np.percentile(boot_alpha_ols, 97.5))],
            "alpha_ols_std": float(np.std(boot_alpha_ols)),
            "alpha_hill_mean": float(np.mean(boot_alpha_hill)),
            "alpha_hill_ci": [float(np.percentile(boot_alpha_hill, 2.5)),
                              float(np.percentile(boot_alpha_hill, 97.5))],
            "r_mean_mean": float(np.mean(boot_r)),
            "r_mean_ci": [float(np.percentile(boot_r, 2.5)),
                          float(np.percentile(boot_r, 97.5))],
        }

        ci = results[author]["alpha_ols_ci"]
        print(f"  {author:<20} [{dialect:<12}] N={N:>4}  α={results[author]['alpha_ols_mean']:.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    with open(os.path.join(out_dir, "author_bootstrap_cis.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════
# §9 — Main
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 70)
    print("GREEK DIALECT-QUARANTINED SPECTRAL VALIDATION")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "eigenvalues"), exist_ok=True)

    # Load data
    papyri_texts = load_papyri_texts(DATA_DIR)
    literary_by_author = load_literary_texts(DATA_DIR)

    # Load model
    print("\nLoading Nomic v2 MoE (768d)...")
    model = get_embedder(dim=768)

    # ── G1: Koine author comparison ─────────────────────────
    koine_results = test_koine_authors(model, literary_by_author, RESULTS_DIR)
    print("\n✓ CHECKPOINT: G1 complete")
    with open(os.path.join(RESULTS_DIR, "checkpoint_G1.json"), "w") as f:
        json.dump({"status": "complete", "elapsed": time.time() - t0}, f)

    # ── G2: DDB temporal split ──────────────────────────────
    temporal_results = test_temporal_split(model, papyri_texts, RESULTS_DIR)
    print("\n✓ CHECKPOINT: G2 complete")
    with open(os.path.join(RESULTS_DIR, "checkpoint_G2.json"), "w") as f:
        json.dump({"status": "complete", "elapsed": time.time() - t0}, f)

    # ── G3: Register comparison ─────────────────────────────
    register_results = test_register_comparison(model, papyri_texts, literary_by_author, RESULTS_DIR)
    print("\n✓ CHECKPOINT: G3 complete")
    with open(os.path.join(RESULTS_DIR, "checkpoint_G3.json"), "w") as f:
        json.dump({"status": "complete", "elapsed": time.time() - t0}, f)

    # ── G4: Null controls ───────────────────────────────────
    null_results = test_greek_null_controls(model, papyri_texts, literary_by_author, RESULTS_DIR)
    print("\n✓ CHECKPOINT: G4 complete")
    with open(os.path.join(RESULTS_DIR, "checkpoint_G4.json"), "w") as f:
        json.dump({"status": "complete", "elapsed": time.time() - t0}, f)

    # ── G5: Bootstrap CIs ──────────────────────────────────
    bootstrap_results = test_bootstrap_authors(model, literary_by_author, RESULTS_DIR)
    print("\n✓ CHECKPOINT: G5 complete")

    # ── Summary ─────────────────────────────────────────────
    elapsed = time.time() - t0

    summary = {
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "papyri_texts_loaded": len(papyri_texts),
        "literary_authors": list(literary_by_author.keys()),
        "koine_authors_tested": [a for a in sorted(literary_by_author.keys()) if a not in NON_KOINE_AUTHORS],
        "tests_completed": ["G1_koine_authors", "G2_temporal_split",
                           "G3_register_comparison", "G4_null_controls",
                           "G5_bootstrap"],
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL GREEK TESTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print("=" * 70)

    for root, dirs, files in os.walk(RESULTS_DIR):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            size = os.path.getsize(path)
            print(f"  {os.path.relpath(path, RESULTS_DIR):<50} {size:>10,} bytes")


if __name__ == "__main__":
    main()
