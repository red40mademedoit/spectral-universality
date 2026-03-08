#!/usr/bin/env python3
"""
Internal Spectroscopy — Proper Power
======================================
Run RMT diagnostics (⟨r⟩, α, soft rank) through Qwen2.5-72B-Instruct
on 50 texts per domain across all 80 layers.

Domains: 5 arXiv categories × 50 texts = 250 forward passes.
Per text: capture hidden states at all 80 layers.
Per layer-domain pair: build activation covariance → eigendecompose → diagnostics.

Target: NVIDIA B300 288GB SXM6 (Qwen 72B bf16 = ~144GB, leaving 144GB headroom)

Output: internal_spectroscopy/ with per-layer, per-domain JSON + eigenvalue .npy files
"""

import json
import time
import os
import gc
import numpy as np
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("results/internal_spectroscopy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

# 5 maximally separated categories from z-score analysis
DOMAINS = ["math.AG", "cs.CL", "hep-th", "physics.hist-ph", "q-bio.MN"]
TEXTS_PER_DOMAIN = 50
MAX_SEQ_LEN = 512
SEED = 42

rng = np.random.default_rng(SEED)


def spacing_ratio(eigs):
    eigs = np.sort(np.real(eigs))
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return None
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(ratios))


def alpha_ols(eigs, frac=0.30):
    eigs = np.sort(np.abs(eigs))[::-1]
    eigs = eigs[eigs > 0]
    k = max(int(len(eigs) * frac), 5)
    if k < 5:
        return None, None
    tail = eigs[:k]
    log_r = np.log(np.arange(1, len(tail) + 1))
    log_e = np.log(tail)
    slope, _, r_value, _, _ = stats.linregress(log_e, log_r)
    return abs(slope), r_value ** 2


def soft_rank(eigs, threshold=0.99):
    """Fraction of eigenvalues needed to capture `threshold` of total variance."""
    eigs = np.sort(np.abs(eigs))[::-1]
    total = np.sum(eigs)
    if total < 1e-15:
        return None
    cumsum = np.cumsum(eigs) / total
    k = np.searchsorted(cumsum, threshold) + 1
    return k / len(eigs)


# ─── Load corpus ───
print("Loading corpus...")
with open(CORPUS_PATH) as f:
    data = json.load(f)
docs = data["documents"]

# Select texts per domain
domain_texts = {}
for domain in DOMAINS:
    cat_docs = [d for d in docs if d["primary_category"] == domain]
    if len(cat_docs) > TEXTS_PER_DOMAIN:
        chosen = rng.choice(len(cat_docs), size=TEXTS_PER_DOMAIN, replace=False)
        domain_texts[domain] = [cat_docs[i]["text"] for i in chosen]
    else:
        domain_texts[domain] = [d["text"] for d in cat_docs]
    print(f"  {domain}: {len(domain_texts[domain])} texts")

# ─── Load model ───
print(f"\nLoading {MODEL_NAME}...")
t0 = time.time()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True,
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

# ─── Capture activations ───
print(f"\n{'='*70}")
print(f"CAPTURING ACTIVATIONS: {len(DOMAINS)} domains × {TEXTS_PER_DOMAIN} texts × {n_layers} layers")
print(f"{'='*70}")

# For each domain, collect hidden states per layer
# Shape per domain per layer: (n_texts, hidden_dim) — take [CLS]/last token representation
domain_activations = {d: {l: [] for l in range(n_layers + 1)} for d in DOMAINS}

total_texts = sum(len(v) for v in domain_texts.values())
processed = 0

for domain in DOMAINS:
    print(f"\n--- {domain} ({len(domain_texts[domain])} texts) ---")
    for i, text in enumerate(domain_texts[domain]):
        t0 = time.time()

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LEN, padding=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract last-token hidden state from each layer
        # outputs.hidden_states is tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            # Take mean over sequence positions (more stable than last-token for short texts)
            h = hs[0].mean(dim=0).float().cpu().numpy()  # (hidden_dim,)
            domain_activations[domain][layer_idx].append(h)

        processed += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(domain_texts[domain])}  ({elapsed:.1f}s/text)")

        # Clear CUDA cache periodically
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

print(f"\nActivations captured: {processed} texts × {n_layers + 1} layers")

# ─── Compute spectral diagnostics per layer per domain ───
print(f"\n{'='*70}")
print("SPECTRAL DIAGNOSTICS PER LAYER")
print(f"{'='*70}")

layer_results = []

for layer_idx in range(n_layers + 1):
    layer_label = f"layer_{layer_idx:03d}" if layer_idx > 0 else "embedding"
    layer_data = {"layer": layer_idx, "label": layer_label, "domains": {}}

    for domain in DOMAINS:
        acts = np.array(domain_activations[domain][layer_idx])  # (n_texts, hidden_dim)
        N, D = acts.shape

        # Center
        acts_c = acts - acts.mean(axis=0)
        norms = np.linalg.norm(acts_c, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        acts_c = acts_c / norms

        # Covariance matrix (D×D or N×N, whichever is smaller)
        if N < D:
            # Gram matrix (N×N)
            G = acts_c @ acts_c.T
            eigs = np.linalg.eigvalsh(G)[::-1]
        else:
            # Covariance matrix (D×D)
            C = acts_c.T @ acts_c / N
            eigs = np.linalg.eigvalsh(C)[::-1]

        r = spacing_ratio(eigs)
        a, r2 = alpha_ols(eigs)
        sr = soft_rank(eigs)

        np.save(OUTPUT_DIR / f"eigs_{layer_label}_{domain}.npy", eigs)

        layer_data["domains"][domain] = {
            "N": N, "D": D,
            "r_mean": r,
            "alpha_ols": float(a) if a else None,
            "r2": float(r2) if r2 else None,
            "soft_rank": sr,
            "lambda_1": float(eigs[0]),
            "n_positive": int(np.sum(eigs > 0)),
        }

    layer_results.append(layer_data)

    # Print summary every 10 layers
    if layer_idx % 10 == 0 or layer_idx == n_layers:
        print(f"\n  Layer {layer_idx:3d}:")
        for domain in DOMAINS:
            d = layer_data["domains"][domain]
            a_str = f"{d['alpha_ols']:.3f}" if d['alpha_ols'] else "N/A"
            r_str = f"{d['r_mean']:.4f}" if d['r_mean'] else "N/A"
            sr_str = f"{d['soft_rank']:.3f}" if d['soft_rank'] else "N/A"
            print(f"    {domain:<20} α={a_str:<8} ⟨r⟩={r_str:<8} SR={sr_str}")

# ─── Save ───
output = {
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "domains": DOMAINS,
    "texts_per_domain": TEXTS_PER_DOMAIN,
    "max_seq_len": MAX_SEQ_LEN,
    "layers": layer_results,
}

with open(OUTPUT_DIR / "internal_spectroscopy.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'internal_spectroscopy.json'}")

# ─── Summary trajectories ───
print(f"\n{'='*70}")
print("TRAJECTORY SUMMARY (every 10th layer)")
print(f"{'='*70}")
print(f"{'Layer':>6} ", end="")
for d in DOMAINS:
    print(f"{'α_'+d:>14}", end="")
print()

for lr in layer_results:
    if lr["layer"] % 10 == 0 or lr["layer"] == n_layers:
        print(f"{lr['layer']:>6} ", end="")
        for d in DOMAINS:
            a = lr["domains"][d].get("alpha_ols")
            print(f"{a:>14.3f}" if a else f"{'N/A':>14}", end="")
        print()

print("\nDone!")
