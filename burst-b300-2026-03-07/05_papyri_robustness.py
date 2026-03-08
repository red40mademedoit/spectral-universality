#!/usr/bin/env python3
"""
Documentary Papyri Subsampling Robustness
==========================================
Draw 10 independent random subsamples of N=2000 from the 55K DDB
documentary papyri, embed each, compute α and ⟨r⟩.

Shows that α is stable across different random draws.

Output: papyri_robustness.json
"""

import json
import time
import os
import numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("results/papyri_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAPYRI_DIR = os.path.expanduser("~/greek_data/ddb_extracted/ddb")
N_SUBSAMPLE = 2000
N_DRAWS = 10
TAIL_FRAC = 0.30
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


# ─── Load papyri ───
print("Loading documentary papyri...")
texts = []
papyri_dir = Path(PAPYRI_DIR)
if not papyri_dir.exists():
    print(f"ERROR: {PAPYRI_DIR} not found!")
    print("Looking for alternative locations...")
    for alt in ["~/corpus/extracted-papyri", "~/data-greek/ddb_extracted",
                "~/greek_data/ddb"]:
        alt_path = Path(os.path.expanduser(alt))
        if alt_path.exists():
            papyri_dir = alt_path
            print(f"  Found: {papyri_dir}")
            break

for fpath in sorted(papyri_dir.rglob("*.txt")):
    text = fpath.read_text(errors="ignore").strip()
    if len(text) > 20:  # Skip near-empty files
        texts.append(text)

print(f"Loaded {len(texts)} documents (min length > 20 chars)")
if len(texts) < N_SUBSAMPLE:
    print(f"WARNING: Only {len(texts)} texts available, need {N_SUBSAMPLE}")
    N_SUBSAMPLE = min(len(texts), N_SUBSAMPLE)

# ─── Load model ───
print("Loading Nomic Embed v2 MoE...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
print(f"Model loaded in {time.time()-t0:.1f}s")

# ─── 10 independent draws ───
print(f"\n{'='*60}")
print(f"SUBSAMPLING ROBUSTNESS: {N_DRAWS} draws of N={N_SUBSAMPLE}")
print(f"{'='*60}")

results = {"N_total": len(texts), "N_subsample": N_SUBSAMPLE, "N_draws": N_DRAWS, "draws": []}

alphas = []
r_signals = []

for draw in range(N_DRAWS):
    t0 = time.time()
    idx = rng.choice(len(texts), size=N_SUBSAMPLE, replace=False)
    sample_texts = [texts[i] for i in idx]

    # Embed
    prefixed = [f"search_document: {t}" for t in sample_texts]
    embeddings = model.encode(prefixed, batch_size=256, show_progress_bar=False,
                              normalize_embeddings=True)
    N, D = embeddings.shape

    # Center and normalize
    X = embeddings - embeddings.mean(axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X = X / norms

    # Gram matrix
    G = X @ X.T
    eigs = np.linalg.eigvalsh(G)[::-1]
    np.save(OUTPUT_DIR / f"eigs_draw_{draw}.npy", eigs)

    a, r2 = alpha_ols(eigs)
    gamma = N / D
    lp = (1 + np.sqrt(gamma)) ** 2
    sig_eigs = eigs[eigs > lp]
    r_sig = spacing_ratio(sig_eigs) if len(sig_eigs) > 3 else None

    elapsed = time.time() - t0
    alphas.append(a)
    if r_sig:
        r_signals.append(r_sig)

    entry = {
        "draw": draw,
        "N": N, "D": D,
        "alpha_ols": float(a),
        "r2": float(r2),
        "r_signal": r_sig,
        "n_signal": len(sig_eigs),
        "lambda_1": float(eigs[0]),
        "time_s": elapsed,
    }
    results["draws"].append(entry)
    r_s = f"{r_sig:.4f}" if r_sig else "N/A"
    print(f"  Draw {draw:2d}: α={a:.4f}  R²={r2:.4f}  ⟨r⟩={r_s}  "
          f"λ₁={eigs[0]:.1f}  ({elapsed:.1f}s)")

# ─── Summary ───
print(f"\n--- Summary ---")
print(f"  α mean = {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")
print(f"  α range = [{min(alphas):.4f}, {max(alphas):.4f}]")
print(f"  α CV = {np.std(alphas)/np.mean(alphas)*100:.1f}%")
if r_signals:
    print(f"  ⟨r⟩ mean = {np.mean(r_signals):.4f} ± {np.std(r_signals):.4f}")

results["summary"] = {
    "alpha_mean": float(np.mean(alphas)),
    "alpha_std": float(np.std(alphas)),
    "alpha_cv_pct": float(np.std(alphas)/np.mean(alphas)*100),
    "alpha_range": [float(min(alphas)), float(max(alphas))],
    "r_signal_mean": float(np.mean(r_signals)) if r_signals else None,
    "r_signal_std": float(np.std(r_signals)) if r_signals else None,
}

with open(OUTPUT_DIR / "papyri_robustness.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'papyri_robustness.json'}")
print("Done!")
