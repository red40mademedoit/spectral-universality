#!/usr/bin/env python3
"""
SPhilBERTa Internal Spectroscopy — Domain-Specialist vs Generalist
====================================================================
Run per-layer activation spectroscopy through SPhilBERTa (125M param RoBERTa
trained on Ancient Greek and Latin) on Greek literary + documentary texts.

Compare α trajectories against Qwen 72B (generalist) on same inputs.
Question: does register separation emerge at different layer depths in a
domain-specialist model vs a generalist that barely knows Greek?

SPhilBERTa: bowphs/SPhilBERTa — 12 layers, 768 hidden dim
- Trained on Ancient Greek and Latin texts
- Shows cleanest Greek register separation at embedding level (literary 1.72 vs papyri 6.26)
- BUT: training is overwhelmingly Attic/Homeric, Koine is outside training sweet spot
  (Tzanoulinou et al. — makes clean register separation MORE impressive, not less)

Also runs SynCLM vs MicroBERT word-shuffle comparison if time permits.

Domains:
  - Greek literary (Plutarch, Koine, 100+ texts)
  - Greek documentary papyri (DDB, Koine, 100 texts)
  - Greek literary (Galen, Koine, for author comparison)
  - Greek literary shuffled (word-order destroyed)

Output: sphilberta_spectroscopy/ with per-layer per-domain JSON + eigenvalues
"""

import json
import time
import os
import gc
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("results/sphilberta_spectroscopy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GREEK_LIT_DIR = os.path.expanduser("~/greek_data/literary")
GREEK_DOC_DIR = os.path.expanduser("~/greek_data/ddb_extracted/ddb")

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


def load_texts_from_dir(dirpath, max_n=100, min_len=50):
    """Load text files from a directory, shuffled."""
    texts = []
    path = Path(dirpath)
    if not path.exists():
        print(f"  WARNING: {dirpath} not found")
        return texts
    files = list(path.rglob("*.txt"))
    rng.shuffle(files)
    for f in files:
        text = f.read_text(errors="ignore").strip()
        if len(text) > min_len:
            texts.append(text)
            if len(texts) >= max_n:
                break
    return texts


# ═══════════════════════════════════════════════════════════
# Load domain texts
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("LOADING GREEK DOMAIN TEXTS")
print("=" * 70)

domain_texts = {}

# Plutarch (highest-N Koine literary author)
plutarch = load_texts_from_dir(f"{GREEK_LIT_DIR}/plutarch", TEXTS_PER_DOMAIN)
domain_texts["plutarch"] = plutarch
print(f"  plutarch: {len(plutarch)} texts")

# Galen (second-highest N, medical writer)
galen = load_texts_from_dir(f"{GREEK_LIT_DIR}/galen", TEXTS_PER_DOMAIN)
domain_texts["galen"] = galen
print(f"  galen: {len(galen)} texts")

# Libanius (rhetor, 103 texts available)
libanius = load_texts_from_dir(f"{GREEK_LIT_DIR}/libanius", TEXTS_PER_DOMAIN)
domain_texts["libanius"] = libanius
print(f"  libanius: {len(libanius)} texts")

# Documentary papyri
papyri = load_texts_from_dir(GREEK_DOC_DIR, TEXTS_PER_DOMAIN)
domain_texts["documentary"] = papyri
print(f"  documentary: {len(papyri)} texts")

# Shuffled Plutarch (word-order destroyed — same vocab, no syntax)
shuffled = []
for text in plutarch:
    words = text.split()
    rng.shuffle(words)
    shuffled.append(" ".join(words))
domain_texts["plutarch_shuffled"] = shuffled[:TEXTS_PER_DOMAIN]
print(f"  plutarch_shuffled: {len(domain_texts['plutarch_shuffled'])} texts")

DOMAINS = list(domain_texts.keys())
print(f"\nTotal domains: {len(DOMAINS)}")


# ═══════════════════════════════════════════════════════════
# Run spectroscopy through SPhilBERTa
# ═══════════════════════════════════════════════════════════

def run_spectroscopy(model_name, tokenizer_name, domains, texts_dict, output_prefix):
    """Run full spectroscopy through a model. Returns results dict."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"LOADING {model_name}")
    print(f"{'='*70}")

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
    ).to("cuda").eval()

    n_layers = mdl.config.num_hidden_layers
    hidden_dim = mdl.config.hidden_size
    print(f"Layers: {n_layers}, Hidden: {hidden_dim}")
    print(f"Model dtype: {next(mdl.parameters()).dtype}")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Capture activations
    domain_activations = {d: defaultdict(list) for d in domains}
    processed = 0
    t_start = time.time()

    for domain in domains:
        texts = texts_dict[domain]
        print(f"\n--- {domain} ({len(texts)} texts) ---")
        t_dom = time.time()

        for i, text in enumerate(texts):
            inputs = tok(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LEN, padding=False
            ).to("cuda")

            with torch.no_grad():
                outputs = mdl(**inputs)

            for layer_idx, hs in enumerate(outputs.hidden_states):
                h = hs[0].mean(dim=0).float().cpu().numpy()
                domain_activations[domain][layer_idx].append(h)

            processed += 1
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t_dom
                print(f"    {i+1}/{len(texts)}  ({(i+1)/elapsed:.1f} texts/s)")

    total_time = time.time() - t_start
    print(f"\nTotal: {processed} texts in {total_time:.1f}s")

    # Compute diagnostics
    print(f"\n--- Spectral diagnostics ---")
    layer_results = []
    out_dir = OUTPUT_DIR / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(n_layers + 1):
        layer_data = {"layer": layer_idx, "domains": {}}

        for domain in domains:
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
            np.save(out_dir / f"eigs_layer{layer_idx:03d}_{domain}.npy", eigs)

            layer_data["domains"][domain] = {
                "N": N, "D": D,
                "r_mean": r,
                "alpha_ols": float(a) if a else None,
                "r2": float(r2) if r2 else None,
                "soft_rank": sr,
                "lambda_1": float(eigs[0]),
            }

        layer_results.append(layer_data)

        if layer_idx % 3 == 0 or layer_idx == n_layers:
            print(f"  Layer {layer_idx:3d}:", end="")
            for domain in domains:
                d = layer_data["domains"][domain]
                a_s = f"{d['alpha_ols']:.3f}" if d['alpha_ols'] else "N/A"
                print(f"  {domain[:8]}={a_s}", end="")
            print()

    # Trajectories
    trajectories = {}
    for domain in domains:
        trajectories[domain] = {
            "alpha": [lr["domains"][domain].get("alpha_ols") for lr in layer_results],
            "r_mean": [lr["domains"][domain].get("r_mean") for lr in layer_results],
            "soft_rank": [lr["domains"][domain].get("soft_rank") for lr in layer_results],
        }

    output = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "domains": domains,
        "texts_per_domain": {d: len(texts_dict[d]) for d in domains},
        "total_time_s": total_time,
        "layers": layer_results,
        "trajectories": trajectories,
    }

    with open(out_dir / f"{output_prefix}_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_dir / f"{output_prefix}_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2, default=str)

    # Cleanup
    del mdl, tok
    torch.cuda.empty_cache()
    gc.collect()

    return output


# ═══════════════════════════════════════════════════════════
# Run 1: SPhilBERTa (domain specialist)
# ═══════════════════════════════════════════════════════════

sphilberta_results = run_spectroscopy(
    model_name="bowphs/SPhilBERTa",
    tokenizer_name="bowphs/SPhilBERTa",
    domains=DOMAINS,
    texts_dict=domain_texts,
    output_prefix="sphilberta"
)


# ═══════════════════════════════════════════════════════════
# Run 2: SynCLM (syntactic bias) — if available
# ═══════════════════════════════════════════════════════════

try:
    synclm_results = run_spectroscopy(
        model_name="Pclanglais/SynCLM",
        tokenizer_name="Pclanglais/SynCLM",
        domains=DOMAINS,
        texts_dict=domain_texts,
        output_prefix="synclm"
    )
    print("\nSynCLM run complete!")
except Exception as e:
    print(f"\nSynCLM failed: {e}")
    print("Trying MicroBERT instead...")
    try:
        microbert_results = run_spectroscopy(
            model_name="Pclanglais/MicroBERT",
            tokenizer_name="Pclanglais/MicroBERT",
            domains=DOMAINS,
            texts_dict=domain_texts,
            output_prefix="microbert"
        )
    except Exception as e2:
        print(f"MicroBERT also failed: {e2}")
        print("Skipping syntactic bias comparison.")


# ═══════════════════════════════════════════════════════════
# Summary comparison
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("CROSS-MODEL TRAJECTORY COMPARISON")
print(f"{'='*70}")

# SPhilBERTa summary
print("\nSPhilBERTa α trajectory (every 2nd layer):")
for lr in sphilberta_results["layers"]:
    if lr["layer"] % 2 == 0 or lr["layer"] == sphilberta_results["n_layers"]:
        row = f"  Layer {lr['layer']:3d}:"
        for d in DOMAINS:
            a = lr["domains"][d].get("alpha_ols")
            row += f"  {d[:8]}={a:.3f}" if a else f"  {d[:8]}=N/A"
        print(row)

print(f"\n{'='*70}")
print("DONE — Results in results/sphilberta_spectroscopy/")
print(f"{'='*70}")
