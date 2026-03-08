#!/usr/bin/env python3
"""
Leave-One-Out Cross-Validation for Category Classification Using α
====================================================================
For each document in each category:
  1. Remove it from the category
  2. Compute α for remaining N-1 documents
  3. Compute α for each other category (all N documents)
  4. Assign the document to the category whose α is closest

Reports classification accuracy per category and confusion matrix.

Uses pre-computed embeddings from 02_n_dependence.py.

Output: loocv_results.json
"""

import json
import time
import os
import numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("results/loocv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
EMBED_PATH = Path("results/n_dependence/full_embeddings.npy")
TAIL_FRAC = 0.30
SEED = 42


def alpha_ols(eigs, frac=0.30):
    eigs = np.sort(np.abs(eigs))[::-1]
    eigs = eigs[eigs > 0]
    k = max(int(len(eigs) * frac), 5)
    tail = eigs[:k]
    if len(tail) < 5:
        return None, None
    log_r = np.log(np.arange(1, len(tail) + 1))
    log_e = np.log(tail)
    slope, _, r_value, _, _ = stats.linregress(log_e, log_r)
    return abs(slope), r_value ** 2


def gram_alpha(X, frac=0.30):
    """Compute α from a set of embeddings."""
    X_c = X - X.mean(axis=0)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X_c = X_c / norms
    G = X_c @ X_c.T
    eigs = np.linalg.eigvalsh(G)[::-1]
    a, r2 = alpha_ols(eigs, frac)
    return a


# ─── Load ───
print("Loading...")
with open(CORPUS_PATH) as f:
    data = json.load(f)
docs = data["documents"]
categories_list = [d["primary_category"] for d in docs]
cat_names = sorted(set(categories_list))

if EMBED_PATH.exists():
    embeddings = np.load(EMBED_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")
else:
    print("ERROR: Embeddings not found. Run 02_n_dependence.py first.")
    exit(1)

# Build category index
cat_indices = {}
for cat in cat_names:
    cat_indices[cat] = [j for j, c in enumerate(categories_list) if c == cat]

# ─── Compute reference α for each category ───
print("\n--- Reference α per category ---")
ref_alpha = {}
for cat in cat_names:
    idx = cat_indices[cat]
    a = gram_alpha(embeddings[idx])
    ref_alpha[cat] = a
    print(f"  {cat:<22} N={len(idx):>3}  α={a:.4f}")

# ─── LOOCV ───
print(f"\n{'='*70}")
print(f"LEAVE-ONE-OUT CROSS-VALIDATION ({len(docs)} documents)")
print(f"{'='*70}")

correct = 0
total = 0
per_cat_correct = {c: 0 for c in cat_names}
per_cat_total = {c: 0 for c in cat_names}
confusion = {c1: {c2: 0 for c2 in cat_names} for c1 in cat_names}

t0 = time.time()

for cat_idx_i, true_cat in enumerate(cat_names):
    cat_doc_indices = cat_indices[true_cat]
    N_cat = len(cat_doc_indices)

    print(f"\n  {true_cat} (N={N_cat})...", end="", flush=True)

    for leave_out_pos in range(N_cat):
        # Remove one document, compute α for remaining
        remaining = [cat_doc_indices[j] for j in range(N_cat) if j != leave_out_pos]
        if len(remaining) < 10:  # Need minimum N for meaningful α
            continue

        a_without = gram_alpha(embeddings[remaining])
        if a_without is None:
            continue

        # Find closest category by α
        best_cat = None
        best_dist = float("inf")
        for cand_cat in cat_names:
            if cand_cat == true_cat:
                cand_alpha = a_without  # Use leave-one-out α
            else:
                cand_alpha = ref_alpha[cand_cat]

            dist = abs(a_without - cand_alpha) if cand_cat == true_cat else abs(ref_alpha[cand_cat] - a_without)
            # Actually: classify by which ref_alpha the leave-one-out α is closest to
            # But for true_cat, use a_without as the reference

        # Simpler: compute α_without, see which ref_alpha it's closest to
        best_cat = min(cat_names,
                       key=lambda c: abs(ref_alpha[c] - a_without) if c != true_cat
                       else abs(a_without - ref_alpha[true_cat]))

        # Actually the right approach: for classification, α_without IS the
        # estimate for this category with one doc removed. Compare it to
        # all reference alphas and assign to closest.
        best_cat = min(cat_names, key=lambda c: abs(ref_alpha[c] - a_without))

        confusion[true_cat][best_cat] += 1
        per_cat_total[true_cat] += 1
        total += 1
        if best_cat == true_cat:
            correct += 1
            per_cat_correct[true_cat] += 1

    acc = per_cat_correct[true_cat] / max(per_cat_total[true_cat], 1) * 100
    print(f" {per_cat_correct[true_cat]}/{per_cat_total[true_cat]} ({acc:.0f}%)")

elapsed = time.time() - t0
overall_acc = correct / max(total, 1) * 100

print(f"\n{'='*70}")
print(f"OVERALL ACCURACY: {correct}/{total} ({overall_acc:.1f}%)")
print(f"Time: {elapsed:.1f}s")
print(f"{'='*70}")

# ─── Per-category accuracy ───
print(f"\n{'Category':<22} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
print("-" * 50)
for cat in cat_names:
    acc = per_cat_correct[cat] / max(per_cat_total[cat], 1) * 100
    print(f"{cat:<22} {per_cat_correct[cat]:>8} {per_cat_total[cat]:>6} {acc:>9.1f}%")

# ─── Save ───
results = {
    "overall_accuracy": overall_acc,
    "correct": correct,
    "total": total,
    "time_s": elapsed,
    "per_category": {
        cat: {
            "correct": per_cat_correct[cat],
            "total": per_cat_total[cat],
            "accuracy": per_cat_correct[cat] / max(per_cat_total[cat], 1) * 100,
        }
        for cat in cat_names
    },
    "confusion_matrix": confusion,
    "reference_alpha": {c: float(v) for c, v in ref_alpha.items()},
}

with open(OUTPUT_DIR / "loocv_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'loocv_results.json'}")
print("Done!")
