#!/usr/bin/env python3
"""
Internal Spectroscopy v2 — Proper Power, Full Scope
=====================================================
Qwen2.5-72B-Instruct on B300 288GB.

Domains (8-10):
  - math.AG (highest arXiv α)
  - q-bio.MN (lowest arXiv α)
  - cs.CL (NLP)
  - hep-th (theoretical physics)
  - physics.hist-ph (weakest separation from null)
  - Greek literary (Plutarch, highest N)
  - Greek documentary papyri
  - Random English (word-shuffled arXiv, destroys syntax)

Per (layer, domain): activation covariance → eigendecompose → α, ⟨r⟩, soft rank
Per layer: weight matrix (W_Q, W_K, W_V) SVD → α (domain-independent control)
Per (layer, domain): attention matrix eigenspectra (exploratory)

100 texts per domain × 80 layers of Qwen 72B.
"""

import json
import time
import os
import gc
import glob
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("results/internal_spectroscopy_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
GREEK_LIT_DIR = os.path.expanduser("~/greek_data/literary/plutarch")
GREEK_DOC_DIR = os.path.expanduser("~/greek_data/ddb_extracted/ddb")
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

TEXTS_PER_DOMAIN = 100
MAX_SEQ_LEN = 512
SEED = 42
SAMPLE_HEADS = 8  # Attention heads to sample per layer

rng = np.random.default_rng(SEED)


# ═══════════════════════════════════════════════════════════
# Spectral utilities
# ═══════════════════════════════════════════════════════════

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
    if k < 5 or len(eigs) < 5:
        return None, None
    tail = eigs[:k]
    log_r = np.log(np.arange(1, len(tail) + 1))
    log_e = np.log(tail)
    slope, _, r_value, _, _ = stats.linregress(log_e, log_r)
    return abs(slope), r_value ** 2


def soft_rank(eigs, threshold=0.99):
    eigs = np.sort(np.abs(eigs))[::-1]
    total = np.sum(eigs)
    if total < 1e-15:
        return None
    cumsum = np.cumsum(eigs) / total
    k = np.searchsorted(cumsum, threshold) + 1
    return k / len(eigs)


def spectral_diagnostics(acts):
    """Compute all RMT diagnostics from activation matrix (N_texts, hidden_dim)."""
    N, D = acts.shape
    X = acts - acts.mean(axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    X = X / norms

    if N < D:
        G = X @ X.T  # Gram matrix
        eigs = np.linalg.eigvalsh(G)[::-1]
    else:
        C = X.T @ X / N  # Covariance
        eigs = np.linalg.eigvalsh(C)[::-1]

    r = spacing_ratio(eigs)
    a, r2 = alpha_ols(eigs)
    sr = soft_rank(eigs)

    return {
        "N": N, "D": D,
        "r_mean": r,
        "alpha_ols": float(a) if a else None,
        "r2": float(r2) if r2 else None,
        "soft_rank": sr,
        "lambda_1": float(eigs[0]),
        "n_positive": int(np.sum(eigs > 0)),
    }, eigs


# ═══════════════════════════════════════════════════════════
# Load domain texts
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("LOADING DOMAIN TEXTS")
print("=" * 70)

domain_texts = {}

# ArXiv categories
with open(CORPUS_PATH) as f:
    arxiv = json.load(f)["documents"]

arxiv_domains = ["math.AG", "q-bio.MN", "cs.CL", "hep-th", "physics.hist-ph"]
for dom in arxiv_domains:
    cat_docs = [d["text"] for d in arxiv if d["primary_category"] == dom]
    chosen = rng.choice(len(cat_docs), size=min(TEXTS_PER_DOMAIN, len(cat_docs)), replace=False)
    domain_texts[dom] = [cat_docs[i] for i in chosen]
    print(f"  {dom}: {len(domain_texts[dom])} texts")

# Greek literary (Plutarch — highest N Koine author)
greek_lit_texts = []
lit_path = Path(GREEK_LIT_DIR)
if lit_path.exists():
    for f in sorted(lit_path.glob("*.txt")):
        text = f.read_text(errors="ignore").strip()
        if len(text) > 50:
            greek_lit_texts.append(text)
if len(greek_lit_texts) >= TEXTS_PER_DOMAIN:
    chosen = rng.choice(len(greek_lit_texts), size=TEXTS_PER_DOMAIN, replace=False)
    domain_texts["greek_literary"] = [greek_lit_texts[i] for i in chosen]
else:
    domain_texts["greek_literary"] = greek_lit_texts[:TEXTS_PER_DOMAIN]
print(f"  greek_literary (Plutarch): {len(domain_texts['greek_literary'])} texts")

# Greek documentary papyri
greek_doc_texts = []
doc_path = Path(GREEK_DOC_DIR)
if doc_path.exists():
    all_papyri = list(doc_path.rglob("*.txt"))
    rng.shuffle(all_papyri)
    for f in all_papyri:
        text = f.read_text(errors="ignore").strip()
        if len(text) > 50:
            greek_doc_texts.append(text)
            if len(greek_doc_texts) >= TEXTS_PER_DOMAIN:
                break
domain_texts["greek_documentary"] = greek_doc_texts[:TEXTS_PER_DOMAIN]
print(f"  greek_documentary: {len(domain_texts['greek_documentary'])} texts")

# Random English (word-shuffled arXiv — destroys syntax, preserves vocabulary)
random_texts = []
for d in arxiv[:200]:
    words = d["text"].split()
    rng.shuffle(words)
    random_texts.append(" ".join(words))
domain_texts["random_english"] = random_texts[:TEXTS_PER_DOMAIN]
print(f"  random_english: {len(domain_texts['random_english'])} texts")

DOMAINS = list(domain_texts.keys())
print(f"\nTotal domains: {len(DOMAINS)}, texts per domain: {TEXTS_PER_DOMAIN}")
total_fwd = sum(len(v) for v in domain_texts.values())
print(f"Total forward passes: {total_fwd}")


# ═══════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"LOADING {MODEL_NAME}")
print(f"{'='*70}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True,
    output_attentions=True,  # For attention weight spectra
)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
n_heads = model.config.num_attention_heads
head_dim = hidden_dim // n_heads
print(f"Layers: {n_layers}, Hidden: {hidden_dim}, Heads: {n_heads}, Head dim: {head_dim}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════
# Phase 1: Weight matrix α (domain-independent, run once)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PHASE 1: WEIGHT MATRIX α (domain-independent)")
print(f"{'='*70}")

weight_results = []

WEIGHT_SAMPLE_LAYERS = list(range(0, n_layers, 10))  # Every 10th layer only
print(f"  Sampling {len(WEIGHT_SAMPLE_LAYERS)} of {n_layers} layers for weight SVD")

for layer_idx in WEIGHT_SAMPLE_LAYERS:
    layer = model.model.layers[layer_idx]

    layer_weight_data = {"layer": layer_idx, "matrices": {}}

    # Only W_K and W_V (small: 1024×8192 for GQA), skip W_Q and W_O (8192×8192)
    for name, attr in [("W_K", "k_proj"), ("W_V", "v_proj")]:
        proj = getattr(layer.self_attn, attr, None)
        if proj is None:
            continue
        W = proj.weight.detach().float().cpu().numpy()

        try:
            sv = np.linalg.svd(W, compute_uv=False)
            eigs_w = sv ** 2
            a_w, r2_w = alpha_ols(eigs_w)
            r_w = spacing_ratio(eigs_w)
            sr_w = soft_rank(eigs_w)

            np.save(OUTPUT_DIR / f"weight_sv_layer{layer_idx:03d}_{name}.npy", sv)

            layer_weight_data["matrices"][name] = {
                "shape": list(W.shape),
                "alpha_ols": float(a_w) if a_w else None,
                "r2": float(r2_w) if r2_w else None,
                "r_mean": r_w,
                "soft_rank": sr_w,
                "sv_max": float(sv[0]),
                "sv_min": float(sv[-1]),
            }
        except Exception as e:
            layer_weight_data["matrices"][name] = {"error": str(e)}

    weight_results.append(layer_weight_data)

    wk = layer_weight_data["matrices"].get("W_K", {})
    a_k = wk.get("alpha_ols", "N/A")
    a_k_s = f"{a_k:.3f}" if isinstance(a_k, float) else a_k
    print(f"  Layer {layer_idx:3d}: W_K α={a_k_s}")

# Save weight results
with open(OUTPUT_DIR / "weight_matrix_results.json", "w") as f:
    json.dump(weight_results, f, indent=2, default=str)
print("Weight matrix analysis saved.")


# ═══════════════════════════════════════════════════════════
# Phase 2: Activation α per layer per domain (the headline)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"PHASE 2: ACTIVATION α ({len(DOMAINS)} domains × {TEXTS_PER_DOMAIN} texts × {n_layers} layers)")
print(f"{'='*70}")

# Collect activations: domain_activations[domain][layer] = list of vectors
domain_activations = {d: defaultdict(list) for d in DOMAINS}
# Also collect attention patterns for a subsample
domain_attentions = {d: defaultdict(list) for d in DOMAINS}
ATTENTION_SUBSAMPLE = 20  # Only store attention for first 20 texts per domain

processed = 0
t_start = time.time()

for domain in DOMAINS:
    texts = domain_texts[domain]
    n_texts = len(texts)
    print(f"\n--- {domain} ({n_texts} texts) ---")
    t_dom = time.time()

    for i, text in enumerate(texts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LEN, padding=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Hidden states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            h = hs[0].mean(dim=0).float().cpu().numpy()  # Mean-pool over positions
            domain_activations[domain][layer_idx].append(h)

        # Attention: tuple of n_layers tensors, each (1, n_heads, seq_len, seq_len)
        if i < ATTENTION_SUBSAMPLE and outputs.attentions is not None:
            for layer_idx, attn in enumerate(outputs.attentions):
                # Average across heads → (seq_len, seq_len)
                avg_attn = attn[0].mean(dim=0).float().cpu().numpy()
                domain_attentions[domain][layer_idx].append(avg_attn)

        processed += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_dom
            rate = (i + 1) / elapsed
            eta = (n_texts - i - 1) / rate
            print(f"    {i+1}/{n_texts}  ({rate:.1f} texts/s, ETA {eta:.0f}s)")

        # Periodic cleanup
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    dom_elapsed = time.time() - t_dom
    print(f"  {domain}: {n_texts} texts in {dom_elapsed:.1f}s ({n_texts/dom_elapsed:.1f} texts/s)")

total_elapsed = time.time() - t_start
print(f"\nAll activations captured: {processed} texts in {total_elapsed:.1f}s")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════
# Phase 3: Compute spectral diagnostics per layer per domain
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("PHASE 3: SPECTRAL DIAGNOSTICS (activation covariance)")
print(f"{'='*70}")

activation_results = []

for layer_idx in range(n_layers + 1):
    layer_label = f"layer_{layer_idx:03d}" if layer_idx > 0 else "embedding"
    layer_data = {"layer": layer_idx, "label": layer_label, "domains": {}}

    for domain in DOMAINS:
        acts = np.array(domain_activations[domain][layer_idx])
        diag, eigs = spectral_diagnostics(acts)
        np.save(OUTPUT_DIR / f"eigs_act_{layer_label}_{domain}.npy", eigs)
        layer_data["domains"][domain] = diag

    activation_results.append(layer_data)

    # Print every 10 layers
    if layer_idx % 10 == 0 or layer_idx == n_layers:
        print(f"\n  Layer {layer_idx:3d}:")
        for domain in DOMAINS:
            d = layer_data["domains"][domain]
            a_s = f"{d['alpha_ols']:.3f}" if d['alpha_ols'] else "N/A"
            r_s = f"{d['r_mean']:.4f}" if d['r_mean'] else "N/A"
            sr_s = f"{d['soft_rank']:.3f}" if d['soft_rank'] else "N/A"
            print(f"    {domain:<22} α={a_s:<8} ⟨r⟩={r_s:<8} SR={sr_s}")

with open(OUTPUT_DIR / "activation_results.json", "w") as f:
    json.dump(activation_results, f, indent=2, default=str)
print("\nActivation spectral diagnostics saved.")


# ═══════════════════════════════════════════════════════════
# Phase 4: Attention weight spectra (exploratory)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"PHASE 4: ATTENTION WEIGHT SPECTRA (N={ATTENTION_SUBSAMPLE} per domain)")
print(f"{'='*70}")

attention_results = []

for layer_idx in range(n_layers):
    layer_data = {"layer": layer_idx, "domains": {}}

    for domain in DOMAINS:
        attn_matrices = domain_attentions[domain].get(layer_idx, [])
        if len(attn_matrices) < 5:
            layer_data["domains"][domain] = {"n": len(attn_matrices), "skipped": True}
            continue

        # Stack attention matrices: (n_texts, seq_len, seq_len)
        # For spectral analysis, flatten each (seq_len, seq_len) → vector
        # Then build covariance over texts
        # OR: eigendecompose each attention matrix directly

        # Approach: per-text eigenspectra, then aggregate
        alphas_per_text = []
        rs_per_text = []
        for attn_mat in attn_matrices:
            eigs_a = np.linalg.eigvalsh(attn_mat)[::-1]
            a, r2 = alpha_ols(eigs_a)
            r = spacing_ratio(eigs_a)
            if a is not None:
                alphas_per_text.append(a)
            if r is not None:
                rs_per_text.append(r)

        layer_data["domains"][domain] = {
            "n": len(attn_matrices),
            "alpha_mean": float(np.mean(alphas_per_text)) if alphas_per_text else None,
            "alpha_std": float(np.std(alphas_per_text)) if alphas_per_text else None,
            "r_mean": float(np.mean(rs_per_text)) if rs_per_text else None,
            "r_std": float(np.std(rs_per_text)) if rs_per_text else None,
        }

    attention_results.append(layer_data)

    if layer_idx % 10 == 0:
        print(f"  Layer {layer_idx:3d}:")
        for domain in DOMAINS[:3]:  # Just first 3 for brevity
            d = layer_data["domains"].get(domain, {})
            if d.get("skipped"):
                continue
            a_s = f"{d.get('alpha_mean', 0):.3f}" if d.get('alpha_mean') else "N/A"
            r_s = f"{d.get('r_mean', 0):.4f}" if d.get('r_mean') else "N/A"
            print(f"    {domain:<22} attn_α={a_s:<8} attn_⟨r⟩={r_s}")

with open(OUTPUT_DIR / "attention_results.json", "w") as f:
    json.dump(attention_results, f, indent=2, default=str)
print("Attention spectral analysis saved.")


# ═══════════════════════════════════════════════════════════
# Summary: α trajectories (the money table)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("α TRAJECTORY SUMMARY (every 5th layer)")
print(f"{'='*70}")

header = f"{'Layer':>6}"
for d in DOMAINS:
    short = d[:12]
    header += f" {short:>13}"
print(header)
print("-" * (6 + 14 * len(DOMAINS)))

for lr in activation_results:
    if lr["layer"] % 5 == 0 or lr["layer"] == n_layers:
        row = f"{lr['layer']:>6}"
        for d in DOMAINS:
            a = lr["domains"][d].get("alpha_ols")
            row += f" {a:>13.3f}" if a else f" {'N/A':>13}"
        print(row)

# Check: do ⟨r⟩ values remain flat while α separates?
print(f"\n⟨r⟩ TRAJECTORY (every 10th layer — expect flat Poisson)")
print(f"{'Layer':>6}", end="")
for d in DOMAINS:
    print(f" {d[:12]:>13}", end="")
print()
for lr in activation_results:
    if lr["layer"] % 10 == 0 or lr["layer"] == n_layers:
        row = f"{lr['layer']:>6}"
        for d in DOMAINS:
            r = lr["domains"][d].get("r_mean")
            row += f" {r:>13.4f}" if r else f" {'N/A':>13}"
        print(row)


# ═══════════════════════════════════════════════════════════
# Save master output
# ═══════════════════════════════════════════════════════════

master = {
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "n_heads": n_heads,
    "domains": DOMAINS,
    "texts_per_domain": TEXTS_PER_DOMAIN,
    "attention_subsample": ATTENTION_SUBSAMPLE,
    "max_seq_len": MAX_SEQ_LEN,
    "total_forward_passes": processed,
    "total_time_s": total_elapsed,
    "activations": activation_results,
    "weights": weight_results,
    "attentions": attention_results,
}

with open(OUTPUT_DIR / "master_results.json", "w") as f:
    json.dump(master, f, indent=2, default=str)

# Also save a compact trajectory-only file for quick inspection
trajectories = {}
for domain in DOMAINS:
    trajectories[domain] = {
        "alpha": [lr["domains"][domain].get("alpha_ols") for lr in activation_results],
        "r_mean": [lr["domains"][domain].get("r_mean") for lr in activation_results],
        "soft_rank": [lr["domains"][domain].get("soft_rank") for lr in activation_results],
    }
with open(OUTPUT_DIR / "trajectories.json", "w") as f:
    json.dump(trajectories, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"ALL DONE — {processed} forward passes in {total_elapsed:.1f}s")
print(f"Results: {OUTPUT_DIR}")
print(f"Master JSON: {(OUTPUT_DIR / 'master_results.json').stat().st_size:,} bytes")
print(f"{'='*70}")
