#!/usr/bin/env python3
"""
Cross-Architecture Internal Spectroscopy — Embedding Models
============================================================
Same per-layer activation analysis as 06v2 (Qwen 72B) and 08 (Llama 70B)
but on smaller embedding models:

  1. Nomic Embed v2 MoE (137M, 12 layers, 768d)  — the model we USE for outer α
  2. PPLX Embed v1 0.6B (~0.6B, 1024d)           — new Perplexity embedding model
  3. PPLX Embed v1 4B   (~4B, 2560d)              — larger Perplexity variant

Key question: does α domain-separation appear inside these models too?
If yes → separation is a universal transformer property, not Qwen-specific.
Nomic is meta: we're looking inside the tool we use to measure α.
"""

import json
import time
import os
import gc
import sys
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

SEED = 42
rng = np.random.default_rng(SEED)

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
GREEK_LIT_DIR = os.path.expanduser("~/greek_data/literary/plutarch")
GREEK_DOC_DIR = os.path.expanduser("~/greek_data/ddb_extracted/ddb")

TEXTS_PER_DOMAIN = 100
MAX_SEQ_LEN = 512

MODELS = [
    {
        "name": "nomic-v2-moe",
        "hf_id": "nomic-ai/nomic-embed-text-v2-moe",
        "dtype": "float32",
        "trust_remote_code": True,
        "prefix": "search_document: ",
    },
    {
        "name": "pplx-0.6b",
        "hf_id": "perplexity-ai/pplx-embed-v1-0.6B",
        "dtype": "float32",
        "trust_remote_code": True,
        "prefix": "",
    },
    {
        "name": "pplx-4b",
        "hf_id": "perplexity-ai/pplx-embed-v1-4B",
        "dtype": "float32",
        "trust_remote_code": True,
        "prefix": "",
    },
]

# Allow running a single model via CLI arg
if len(sys.argv) > 1:
    target = sys.argv[1]
    MODELS = [m for m in MODELS if m["name"] == target]
    if not MODELS:
        print(f"Unknown model: {target}")
        print(f"Options: nomic-v2-moe, pplx-0.6b, pplx-4b")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Utility functions
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


def alpha_mle(eigs):
    try:
        import powerlaw
        eigs_pos = eigs[eigs > 0]
        if len(eigs_pos) < 10:
            return None
        fit = powerlaw.Fit(eigs_pos, verbose=False)
        return float(fit.power_law.alpha)
    except Exception:
        return None


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
    domain_texts[dom] = [cat_docs[int(i)] for i in chosen]

# Greek literary
lit_path = Path(GREEK_LIT_DIR)
if lit_path.exists():
    greek_lit = []
    for f in sorted(lit_path.glob("*.txt")):
        text = f.read_text(errors="ignore").strip()
        if len(text) > 50:
            greek_lit.append(text)
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

# Random English (word-shuffled arXiv)
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
# Run each model
# ═══════════════════════════════════════════════════════════

import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

for model_cfg in MODELS:
    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    prefix = model_cfg["prefix"]

    output_dir = Path(f"results/multi_model_spectroscopy/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    if (output_dir / f"{name}_results.json").exists():
        print(f"\n{'='*70}")
        print(f"SKIPPING {name} — already complete")
        print(f"{'='*70}")
        continue

    print(f"\n{'='*70}")
    print(f"LOADING {name} ({hf_id})")
    print(f"{'='*70}")
    t_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=model_cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dt = torch.float32 if model_cfg["dtype"] == "float32" else torch.bfloat16

    model = AutoModel.from_pretrained(
        hf_id,
        trust_remote_code=model_cfg["trust_remote_code"],
        dtype=dt,
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    gpu_mem = torch.cuda.memory_allocated() / 1e9

    print(f"  Loaded in {time.time() - t_load:.1f}s")
    print(f"  Params: {n_params:.1f}M, Layers: {n_layers}, Hidden: {hidden_dim}")
    print(f"  GPU memory: {gpu_mem:.2f} GB")

    # ── Try a test forward pass to verify hidden_states ──
    # Some custom models (e.g. NomicBert) don't accept output_hidden_states as kwarg
    # Try multiple approaches
    test_in = tokenizer("test", return_tensors="pt", truncation=True,
                        max_length=32).to(device)
    use_hooks = False

    # Approach 1: set config flag, call without kwarg
    model.config.output_hidden_states = True
    try:
        with torch.no_grad():
            test_out = model(**test_in)
        if hasattr(test_out, "hidden_states") and test_out.hidden_states is not None:
            actual_layers = len(test_out.hidden_states) - 1
            print(f"  Config-based hidden_states works! Got {actual_layers} layers + embedding")
        else:
            raise ValueError("No hidden_states in output")
        del test_out
    except Exception as e1:
        print(f"  Config approach failed: {e1}")
        # Approach 2: pass as kwarg
        try:
            with torch.no_grad():
                test_out = model(**test_in, output_hidden_states=True)
            if hasattr(test_out, "hidden_states") and test_out.hidden_states is not None:
                actual_layers = len(test_out.hidden_states) - 1
                print(f"  Kwarg-based hidden_states works! Got {actual_layers} layers + embedding")
            else:
                raise ValueError("No hidden_states in output")
            del test_out
        except Exception as e2:
            print(f"  Kwarg approach also failed: {e2}")
            print(f"  Falling back to forward hooks")
            use_hooks = True

    # ── Capture activations ──
    print(f"\n  CAPTURING ACTIVATIONS: {len(DOMAINS)} domains × {TEXTS_PER_DOMAIN} texts")
    domain_activations = {d: defaultdict(list) for d in DOMAINS}
    processed = 0
    t_start = time.time()

    for domain in DOMAINS:
        texts = domain_texts[domain]
        print(f"\n  --- {domain} ({len(texts)} texts) ---")
        t_dom = time.time()

        for i, text in enumerate(texts):
            input_text = prefix + text[:3000]
            inputs = tokenizer(
                input_text, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LEN, padding=False
            ).to(device)

            if not use_hooks:
                with torch.no_grad():
                    outputs = model(**inputs)
                hidden_states = outputs.hidden_states
            else:
                # Fallback: hook into encoder/decoder layers
                captured = []
                hooks = []

                def make_hook(storage):
                    def hook_fn(module, inp, out):
                        if isinstance(out, tuple):
                            storage.append(out[0])
                        else:
                            storage.append(out)
                    return hook_fn

                # Try to find transformer layers
                encoder = getattr(model, 'encoder', None)
                if encoder is None:
                    encoder = getattr(model, 'transformer', None)
                if encoder is None:
                    # Walk to find layer list
                    for child_name, child in model.named_children():
                        if hasattr(child, 'layer') or hasattr(child, 'layers'):
                            encoder = child
                            break

                if encoder is not None:
                    layer_list = getattr(encoder, 'layer', getattr(encoder, 'layers', []))
                    for layer in layer_list:
                        hooks.append(layer.register_forward_hook(make_hook(captured)))

                with torch.no_grad():
                    outputs = model(**inputs)

                for h in hooks:
                    h.remove()

                # Prepend embedding if we can find it
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = captured  # layers only, no embedding
                else:
                    hidden_states = captured

            for layer_idx, hs in enumerate(hidden_states):
                if isinstance(hs, torch.Tensor):
                    h = hs[0].mean(dim=0).float().cpu().numpy()
                else:
                    h = hs[0].mean(dim=0).float().cpu().numpy()
                domain_activations[domain][layer_idx].append(h)

            processed += 1
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t_dom
                rate = (i + 1) / elapsed
                print(f"      {i+1}/{len(texts)}  ({rate:.1f} texts/s)")

    total_time = time.time() - t_start
    print(f"\n  Total: {processed} texts in {total_time:.1f}s ({processed/total_time:.1f} texts/s)")

    # ── Compute spectral diagnostics ──
    print(f"\n  SPECTRAL DIAGNOSTICS")
    n_actual_layers = max(max(domain_activations[d].keys()) for d in DOMAINS) + 1
    print(f"  Actual layer count from activations: {n_actual_layers}")

    results_layers = []
    for layer_idx in range(n_actual_layers):
        layer_data = {"layer": layer_idx, "domains": {}}
        for domain in DOMAINS:
            if layer_idx not in domain_activations[domain]:
                continue
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
            a_mle = alpha_mle(eigs)
            sr = soft_rank(eigs)

            np.save(output_dir / f"eigs_layer{layer_idx:03d}_{domain}.npy", eigs)

            layer_data["domains"][domain] = {
                "N": int(N), "D": int(D),
                "r_mean": r,
                "alpha_ols": float(a) if a else None,
                "alpha_mle": float(a_mle) if a_mle else None,
                "r2": float(r2) if r2 else None,
                "soft_rank": sr,
            }
        results_layers.append(layer_data)

        # Print every few layers
        if layer_idx % 3 == 0 or layer_idx == n_actual_layers - 1:
            print(f"\n    Layer {layer_idx:3d}:")
            for domain in DOMAINS:
                if domain in layer_data["domains"]:
                    d = layer_data["domains"][domain]
                    a_s = f"{d['alpha_ols']:.3f}" if d['alpha_ols'] else "N/A"
                    r_s = f"{d['r_mean']:.4f}" if d['r_mean'] else "N/A"
                    sr_s = f"{d['soft_rank']:.3f}" if d['soft_rank'] else "N/A"
                    print(f"      {domain:<22} α={a_s:<8} ⟨r⟩={r_s:<8} SR={sr_s}")

    # ── Build trajectories ──
    trajectories = {}
    for domain in DOMAINS:
        traj = {"alpha_ols": [], "alpha_mle": [], "r_mean": [], "soft_rank": [], "layers": []}
        for lr in results_layers:
            if domain in lr["domains"]:
                d = lr["domains"][domain]
                traj["layers"].append(lr["layer"])
                traj["alpha_ols"].append(d.get("alpha_ols"))
                traj["alpha_mle"].append(d.get("alpha_mle"))
                traj["r_mean"].append(d.get("r_mean"))
                traj["soft_rank"].append(d.get("soft_rank"))
        trajectories[domain] = traj

    # ── Save ──
    output = {
        "model": hf_id,
        "model_name": name,
        "n_layers": n_layers,
        "n_actual_layers": n_actual_layers,
        "hidden_dim": hidden_dim,
        "params_M": round(n_params, 1),
        "domains": DOMAINS,
        "texts_per_domain": TEXTS_PER_DOMAIN,
        "total_time_s": round(total_time, 1),
        "layers": results_layers,
        "trajectories": trajectories,
    }

    with open(output_dir / f"{name}_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    with open(output_dir / f"{name}_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2, default=str)

    print(f"\n  ✓ {name} COMPLETE — results in {output_dir}/")

    # ── Cleanup ──
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU freed: {torch.cuda.memory_allocated() / 1e9:.2f} GB remaining")

print(f"\n{'='*70}")
print("ALL EMBEDDING MODELS COMPLETE")
print(f"{'='*70}")
