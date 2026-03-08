#!/usr/bin/env python3
"""
Per-Category Word-Order Shuffle for arXiv
==========================================
For each of 25 categories:
  1. Compute α on original embeddings
  2. Shuffle words within each document (preserving vocabulary)
  3. Re-embed shuffled texts
  4. Compute α on shuffled embeddings
  5. Compare Δα

Also computes token counts before/after shuffle to test BPE hypothesis
(word reordering creates novel subword boundaries → inflated token count
→ potentially different embedding distribution → α change).

Output: word_shuffle_arxiv.json
"""

import json
import time
import os
import numpy as np
from scipy import stats
from pathlib import Path
from collections import Counter

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
OUTPUT_DIR = Path("results/word_shuffle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)


def alpha_ols(eigs, frac=0.30):
    eigs = np.sort(np.abs(eigs))[::-1]
    eigs = eigs[eigs > 0]
    k = max(int(len(eigs) * frac), 5)
    tail = eigs[:k]
    log_r = np.log(np.arange(1, len(tail) + 1))
    log_e = np.log(tail)
    slope, _, r_value, _, _ = stats.linregress(log_e, log_r)
    return abs(slope), r_value ** 2


def spacing_ratio(eigs):
    eigs = np.sort(np.real(eigs))
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return None
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(ratios))


def shuffle_words(text):
    """Shuffle words in text, preserving vocabulary but destroying order."""
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def embed_and_analyze(model, texts, label=""):
    """Embed texts, compute Gram matrix, eigendecompose, return metrics."""
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(prefixed, batch_size=256, show_progress_bar=False,
                              normalize_embeddings=True)
    N, D = embeddings.shape
    X = embeddings - embeddings.mean(axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X = X / norms

    G = X @ X.T
    eigs = np.linalg.eigvalsh(G)[::-1]

    a_ols, r2 = alpha_ols(eigs, 0.30)
    gamma = N / D
    lp = (1 + np.sqrt(gamma)) ** 2
    sig_eigs = eigs[eigs > lp]
    r_sig = spacing_ratio(sig_eigs) if len(sig_eigs) > 3 else None

    return {
        "N": N, "D": D,
        "alpha_ols": float(a_ols),
        "r2": float(r2),
        "r_signal": r_sig,
        "n_signal": len(sig_eigs),
        "lambda_1": float(eigs[0]),
    }, eigs


# ─── Load corpus ───
print("Loading corpus...")
with open(CORPUS_PATH) as f:
    data = json.load(f)
docs = data["documents"]
categories = sorted(set(d["primary_category"] for d in docs))
print(f"Loaded {len(docs)} documents, {len(categories)} categories")

# ─── Load model + tokenizer ───
print("Loading Nomic Embed v2 MoE...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Get tokenizer for token count analysis
tokenizer = model.tokenizer

# ─── Per-category analysis ───
print("\n" + "=" * 70)
print("PER-CATEGORY WORD-ORDER SHUFFLE")
print("=" * 70)
print(f"{'Category':<22} {'α_orig':>8} {'α_shuf':>8} {'Δα':>8} {'%Δ':>8} "
      f"{'tok_orig':>10} {'tok_shuf':>10} {'%Δ_tok':>8}")
print("-" * 90)

results = {"categories": {}, "global": {}}

all_orig_texts = []
all_shuf_texts = []

for cat in categories:
    cat_docs = [d for d in docs if d["primary_category"] == cat]
    orig_texts = [d["text"] for d in cat_docs]
    shuf_texts = [shuffle_words(t) for t in orig_texts]

    all_orig_texts.extend(orig_texts)
    all_shuf_texts.extend(shuf_texts)

    # Token counts
    orig_tokens = [len(tokenizer.encode(t)) for t in orig_texts]
    shuf_tokens = [len(tokenizer.encode(t)) for t in shuf_texts]
    mean_orig_tok = np.mean(orig_tokens)
    mean_shuf_tok = np.mean(shuf_tokens)
    pct_tok_change = (mean_shuf_tok - mean_orig_tok) / mean_orig_tok * 100

    # Embed and analyze
    orig_metrics, orig_eigs = embed_and_analyze(model, orig_texts, f"{cat}_orig")
    shuf_metrics, shuf_eigs = embed_and_analyze(model, shuf_texts, f"{cat}_shuf")

    np.save(OUTPUT_DIR / f"eigs_orig_{cat}.npy", orig_eigs)
    np.save(OUTPUT_DIR / f"eigs_shuf_{cat}.npy", shuf_eigs)

    delta_alpha = shuf_metrics["alpha_ols"] - orig_metrics["alpha_ols"]
    pct_alpha = delta_alpha / orig_metrics["alpha_ols"] * 100

    cat_result = {
        "original": orig_metrics,
        "shuffled": shuf_metrics,
        "delta_alpha": float(delta_alpha),
        "pct_alpha_change": float(pct_alpha),
        "token_counts": {
            "original_mean": float(mean_orig_tok),
            "shuffled_mean": float(mean_shuf_tok),
            "pct_change": float(pct_tok_change),
            "original_std": float(np.std(orig_tokens)),
            "shuffled_std": float(np.std(shuf_tokens)),
        },
    }
    results["categories"][cat] = cat_result

    print(f"{cat:<22} {orig_metrics['alpha_ols']:>8.3f} {shuf_metrics['alpha_ols']:>8.3f} "
          f"{delta_alpha:>+8.3f} {pct_alpha:>+7.1f}% "
          f"{mean_orig_tok:>10.1f} {mean_shuf_tok:>10.1f} {pct_tok_change:>+7.1f}%")

# ─── Global analysis ───
print("\n--- Global (all 10K) ---")
global_orig, global_orig_eigs = embed_and_analyze(model, all_orig_texts, "global_orig")
global_shuf, global_shuf_eigs = embed_and_analyze(model, all_shuf_texts, "global_shuf")
np.save(OUTPUT_DIR / "eigs_global_orig.npy", global_orig_eigs)
np.save(OUTPUT_DIR / "eigs_global_shuf.npy", global_shuf_eigs)

results["global"] = {
    "original": global_orig,
    "shuffled": global_shuf,
    "delta_alpha": float(global_shuf["alpha_ols"] - global_orig["alpha_ols"]),
}
print(f"  Original α = {global_orig['alpha_ols']:.4f}")
print(f"  Shuffled α = {global_shuf['alpha_ols']:.4f}")
print(f"  Δα = {global_shuf['alpha_ols'] - global_orig['alpha_ols']:+.4f}")

# ─── Token count correlation with α change ───
print("\n--- Token Count vs α Change Correlation ---")
tok_changes = []
alpha_changes = []
for cat in categories:
    r = results["categories"][cat]
    tok_changes.append(r["token_counts"]["pct_change"])
    alpha_changes.append(r["pct_alpha_change"])

corr, p_val = stats.pearsonr(tok_changes, alpha_changes)
print(f"  Pearson r = {corr:.3f}, p = {p_val:.4f}")
results["token_alpha_correlation"] = {
    "pearson_r": float(corr),
    "p_value": float(p_val),
}

# ─── Save ───
with open(OUTPUT_DIR / "word_shuffle_arxiv.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'word_shuffle_arxiv.json'}")
print("Done!")
