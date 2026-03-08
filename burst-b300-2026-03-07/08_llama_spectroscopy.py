#!/usr/bin/env python3
"""
Cross-Architecture Internal Spectroscopy — Llama 3.1 70B
=========================================================
Same analysis as 06v2 but on Llama 3.1 70B-Instruct.
Reuses the same domain texts. Compares α trajectories across architectures.

If trajectories match Qwen → universal property of deep transformers.
If different → architecture-dependent, needs different framing.
"""

import json
import time
import os
import gc
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("results/llama_spectroscopy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse domain texts from the Qwen run
QWEN_RESULTS = Path("results/internal_spectroscopy_v2/master_results.json")
CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
GREEK_LIT_DIR = os.path.expanduser("~/greek_data/literary/plutarch")
GREEK_DOC_DIR = os.path.expanduser("~/greek_data/ddb_extracted/ddb")
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

TEXTS_PER_DOMAIN = 100
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


# ═══════════════════════════════════════════════════════════
# Load domain texts (same as Qwen run)
# ═══════════════════════════════════════════════════════════

print("Loading domain texts...")
domain_texts = {}

with open(CORPUS_PATH) as f:
    arxiv = json.load(f)["documents"]

for dom in ["math.AG", "q-bio.MN", "cs.CL", "hep-th", "physics.hist-ph"]:
    cat_docs = [d["text"] for d in arxiv if d["primary_category"] == dom]
    chosen = rng.choice(len(cat_docs), size=min(TEXTS_PER_DOMAIN, len(cat_docs)), replace=False)
    domain_texts[dom] = [cat_docs[i] for i in chosen]

# Greek literary
lit_path = Path(GREEK_LIT_DIR)
if lit_path.exists():
    greek_lit = [f.read_text(errors="ignore").strip() for f in sorted(lit_path.glob("*.txt"))
                 if len(f.read_text(errors="ignore").strip()) > 50]
    domain_texts["greek_literary"] = greek_lit[:TEXTS_PER_DOMAIN]

# Greek documentary
doc_path = Path(GREEK_DOC_DIR)
if doc_path.exists():
    all_papyri = list(doc_path.rglob("*.txt"))
    rng.shuffle(all_papyri)
    greek_doc = []
    for f in all_papyri:
        text = f.read_text(errors="ignore").strip()
        if len(text) > 50:
            greek_doc.append(text)
            if len(greek_doc) >= TEXTS_PER_DOMAIN:
                break
    domain_texts["greek_documentary"] = greek_doc

# Random English
random_texts = []
for d in arxiv[:200]:
    words = d["text"].split()
    rng.shuffle(words)
    random_texts.append(" ".join(words))
domain_texts["random_english"] = random_texts[:TEXTS_PER_DOMAIN]

DOMAINS = list(domain_texts.keys())
for d in DOMAINS:
    print(f"  {d}: {len(domain_texts[d])} texts")


# ═══════════════════════════════════════════════════════════
# Load Llama 3.1 70B
# ═══════════════════════════════════════════════════════════

print(f"\nLoading {MODEL_NAME}...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True,
)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Layers: {n_layers}, Hidden: {hidden_dim}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════
# Capture activations
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"CAPTURING ACTIVATIONS: {len(DOMAINS)} domains × {TEXTS_PER_DOMAIN} texts")
print(f"{'='*70}")

domain_activations = {d: defaultdict(list) for d in DOMAINS}
processed = 0
t_start = time.time()

for domain in DOMAINS:
    texts = domain_texts[domain]
    print(f"\n--- {domain} ({len(texts)} texts) ---")
    t_dom = time.time()

    for i, text in enumerate(texts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LEN, padding=False
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        for layer_idx, hs in enumerate(outputs.hidden_states):
            h = hs[0].mean(dim=0).float().cpu().numpy()
            domain_activations[domain][layer_idx].append(h)

        processed += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_dom
            print(f"    {i+1}/{len(texts)}  ({(i+1)/elapsed:.1f} texts/s)")

        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

total_time = time.time() - t_start
print(f"\nTotal: {processed} texts in {total_time:.1f}s")


# ═══════════════════════════════════════════════════════════
# Compute diagnostics
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SPECTRAL DIAGNOSTICS")
print(f"{'='*70}")

results = []
for layer_idx in range(n_layers + 1):
    layer_data = {"layer": layer_idx, "domains": {}}
    for domain in DOMAINS:
        acts = np.array(domain_activations[domain][layer_idx])
        N, D = acts.shape
        X = acts - acts.mean(axis=0)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        X = X / norms

        if N < D:
            G = X @ X.T
            eigs = np.linalg.eigvalsh(G)[::-1]
        else:
            C = X.T @ X / N
            eigs = np.linalg.eigvalsh(C)[::-1]

        r = spacing_ratio(eigs)
        a, r2 = alpha_ols(eigs)
        sr = soft_rank(eigs)
        np.save(OUTPUT_DIR / f"eigs_layer{layer_idx:03d}_{domain}.npy", eigs)

        layer_data["domains"][domain] = {
            "N": N, "D": D, "r_mean": r,
            "alpha_ols": float(a) if a else None,
            "r2": float(r2) if r2 else None,
            "soft_rank": sr,
        }
    results.append(layer_data)

    if layer_idx % 10 == 0 or layer_idx == n_layers:
        print(f"\n  Layer {layer_idx:3d}:")
        for domain in DOMAINS:
            d = layer_data["domains"][domain]
            a_s = f"{d['alpha_ols']:.3f}" if d['alpha_ols'] else "N/A"
            r_s = f"{d['r_mean']:.4f}" if d['r_mean'] else "N/A"
            print(f"    {domain:<22} α={a_s:<8} ⟨r⟩={r_s}")

# Save
trajectories = {}
for domain in DOMAINS:
    trajectories[domain] = {
        "alpha": [lr["domains"][domain].get("alpha_ols") for lr in results],
        "r_mean": [lr["domains"][domain].get("r_mean") for lr in results],
        "soft_rank": [lr["domains"][domain].get("soft_rank") for lr in results],
    }

output = {
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "domains": DOMAINS,
    "texts_per_domain": TEXTS_PER_DOMAIN,
    "total_time_s": total_time,
    "layers": results,
    "trajectories": trajectories,
}

with open(OUTPUT_DIR / "llama_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

with open(OUTPUT_DIR / "trajectories.json", "w") as f:
    json.dump(trajectories, f, indent=2, default=str)

print(f"\nDone! Results: {OUTPUT_DIR}")
