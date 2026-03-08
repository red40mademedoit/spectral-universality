#!/usr/bin/env python3
"""
Cross-Architecture Internal Spectroscopy — DeepSeek-R1 (via r1-1776)
=====================================================================
671B MoE model (256 experts, top-8 routing, 61 transformer layers).
Loaded in 4-bit NF4 via bitsandbytes (~335GB → GPU + CPU offload).

CAPTURES THREE THINGS PER (LAYER, DOMAIN):
  1. Activation eigenspectrum (α, ⟨r⟩, soft rank) — same as Qwen/Llama
  2. Expert utilization histogram (which of 256 experts, how often)
  3. Gating entropy (concentrated vs distributed routing)

KEY QUESTION: Do tokens from different spectral regimes get routed to
different experts? If documentary Greek consistently hits a narrow expert
subset while literary Greek spreads wider, and α tracks that pattern,
we've connected spectral structure to the routing mechanism.

SECONDARY: Does the U-shape trajectory persist in MoE? If yes, it's
universal across dense + MoE architectures. If shallower/shifted,
MoE routing may partially substitute for dense spectral compression.

NOTE: DeepSeek-V3 alternates dense attention + MoE FFN within each layer.
We can compare α evolution through dense vs MoE sub-layers.
"""

import json
import time
import os
import gc
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

# Monkey-patch: r1-1776's custom code uses a removed transformers import
import transformers.utils.import_utils as _iu
if not hasattr(_iu, 'is_torch_fx_available'):
    _iu.is_torch_fx_available = lambda: False

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path("results/deepseek_spectroscopy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = os.path.expanduser("~/arxiv_spectral_corpus_10k.json")
GREEK_LIT_DIR = os.path.expanduser("~/greek_data/literary/plutarch")
GREEK_DOC_DIR = os.path.expanduser("~/greek_data/ddb_extracted/ddb")

MODEL_NAME = "perplexity-ai/r1-1776"
TEXTS_PER_DOMAIN = 50  # Reduced — MoE + 4bit is slower
MAX_SEQ_LEN = 512


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


def gating_entropy(expert_counts, n_experts=256):
    """Compute Shannon entropy of expert utilization distribution."""
    total = expert_counts.sum()
    if total == 0:
        return 0.0
    p = expert_counts / total
    p = p[p > 0]  # avoid log(0)
    return float(-np.sum(p * np.log(p)))


def max_entropy(n_experts=256):
    """Maximum possible entropy (uniform routing)."""
    return float(np.log(n_experts))


# ═══════════════════════════════════════════════════════════
# Load domain texts
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
    for fp in all_papyri:
        text = fp.read_text(errors="ignore").strip()
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
# Load DeepSeek-R1 in 4-bit
# ═══════════════════════════════════════════════════════════

print(f"\nLoading {MODEL_NAME} in 4-bit NF4...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

t_load = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True,
)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
load_time = time.time() - t_load

print(f"Loaded in {load_time:.0f}s")
print(f"Layers: {n_layers}, Hidden: {hidden_dim}")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Report device map distribution
if hasattr(model, 'hf_device_map'):
    devices = {}
    for k, v in model.hf_device_map.items():
        dev = str(v)
        devices[dev] = devices.get(dev, 0) + 1
    for dev, count in sorted(devices.items()):
        print(f"  Device {dev}: {count} modules")


# ═══════════════════════════════════════════════════════════
# Identify MoE vs Dense layers + set up routing hooks
# ═══════════════════════════════════════════════════════════

print(f"\nAnalyzing MoE architecture...")

# Find which layers have MoE (gate attribute in their MLP)
moe_layers = {}
dense_layers = []

# Navigate model structure — DeepSeek uses model.model.layers
inner_model = model.model if hasattr(model, 'model') else model
layers_module = inner_model.layers if hasattr(inner_model, 'layers') else None

if layers_module is None:
    print("WARNING: Cannot find model layers — trying alternative paths")
    for name, module in model.named_modules():
        if 'layers' in name and hasattr(module, '__len__'):
            layers_module = module
            break

if layers_module is not None:
    for layer_idx in range(len(layers_module)):
        layer = layers_module[layer_idx]
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            dense_layers.append(layer_idx)
            continue

        # Check for MoE gate
        gate = getattr(mlp, 'gate', None)
        if gate is not None:
            n_experts = None
            # Try to find number of experts
            experts = getattr(mlp, 'experts', None)
            if experts is not None:
                n_experts = len(experts) if hasattr(experts, '__len__') else None
            if n_experts is None:
                # Try config
                n_experts = getattr(model.config, 'n_routed_experts', 256)
            moe_layers[layer_idx] = {
                'gate': gate,
                'n_experts': n_experts,
                'n_shared': getattr(model.config, 'n_shared_experts', 0),
                'topk': getattr(model.config, 'num_experts_per_tok', 8),
            }
        else:
            dense_layers.append(layer_idx)

    print(f"  MoE layers: {len(moe_layers)} (experts per layer: {moe_layers[list(moe_layers.keys())[0]]['n_experts'] if moe_layers else 'N/A'})")
    print(f"  Dense layers: {len(dense_layers)}")
    if moe_layers:
        first_moe = min(moe_layers.keys())
        last_moe = max(moe_layers.keys())
        print(f"  MoE range: layers {first_moe}-{last_moe}")
        info = moe_layers[first_moe]
        print(f"  Top-k: {info['topk']}, Shared experts: {info['n_shared']}")
else:
    print("WARNING: Could not identify layer structure")


# ═══════════════════════════════════════════════════════════
# Set up routing capture hooks
# ═══════════════════════════════════════════════════════════

# Storage for routing decisions per forward pass
routing_capture = {}
routing_hooks = []

def make_gate_hook(layer_idx, storage):
    """Hook that captures gate output (routing logits) for MoE layers."""
    def hook_fn(module, input_tensor, output):
        # Gate output is typically routing logits/scores
        # Store the raw output for later processing
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        # Get top-k expert indices per token
        if logits is not None and logits.dim() >= 2:
            topk = moe_layers[layer_idx]['topk']
            # logits shape: (batch, seq_len, n_experts) or (total_tokens, n_experts)
            topk_vals, topk_indices = torch.topk(logits.float(), k=min(topk, logits.shape[-1]), dim=-1)
            storage[layer_idx] = {
                'expert_indices': topk_indices.cpu().numpy(),
                'expert_weights': torch.softmax(topk_vals, dim=-1).cpu().numpy(),
                'n_tokens': int(logits.shape[-2]) if logits.dim() >= 2 else 1,
            }
    return hook_fn

# Register hooks on all MoE gates
for layer_idx, info in moe_layers.items():
    hook = info['gate'].register_forward_hook(
        make_gate_hook(layer_idx, routing_capture)
    )
    routing_hooks.append(hook)

print(f"  Registered {len(routing_hooks)} routing hooks")


# ═══════════════════════════════════════════════════════════
# Capture activations + routing
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"CAPTURING ACTIVATIONS + ROUTING: {len(DOMAINS)} domains × {TEXTS_PER_DOMAIN} texts")
print(f"{'='*70}")

domain_activations = {d: defaultdict(list) for d in DOMAINS}
# Per-domain expert utilization: domain -> layer -> histogram of expert usage
domain_routing = {d: defaultdict(lambda: np.zeros(256, dtype=np.int64)) for d in DOMAINS}
# Per-domain gating details: domain -> layer -> list of per-text routing info
domain_routing_detail = {d: defaultdict(list) for d in DOMAINS}

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

        # Clear routing capture for this forward pass
        routing_capture.clear()

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract hidden states
        hidden_states = outputs.hidden_states
        if hidden_states is not None:
            for layer_idx, hs in enumerate(hidden_states):
                h = hs[0].mean(dim=0).float().cpu().numpy()
                domain_activations[domain][layer_idx].append(h)

        # Extract routing decisions
        for layer_idx, routing_info in routing_capture.items():
            expert_indices = routing_info['expert_indices']  # (batch, seq, topk) or (tokens, topk)
            # Flatten to get all expert assignments
            flat_indices = expert_indices.flatten()
            n_experts = moe_layers[layer_idx]['n_experts']
            # Update utilization histogram
            for idx in flat_indices:
                if 0 <= idx < 256:
                    domain_routing[domain][layer_idx][idx] += 1

            # Per-text routing stats
            unique_experts = len(np.unique(flat_indices))
            text_histogram = np.zeros(n_experts, dtype=np.int64)
            for idx in flat_indices:
                if 0 <= idx < n_experts:
                    text_histogram[idx] += 1
            text_entropy = gating_entropy(text_histogram, n_experts)

            domain_routing_detail[domain][layer_idx].append({
                'unique_experts': unique_experts,
                'gating_entropy': text_entropy,
                'n_tokens': routing_info['n_tokens'],
            })

        processed += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_dom
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate if rate > 0 else 0
            routing_layers = len(routing_capture)
            print(f"    {i+1}/{len(texts)}  ({rate:.2f} texts/s, ETA: {eta:.0f}s, routing layers captured: {routing_layers})")

        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()

total_time = time.time() - t_start
print(f"\nTotal: {processed} texts in {total_time:.0f}s ({processed/total_time:.2f} texts/s)")

# Remove hooks
for h in routing_hooks:
    h.remove()


# ═══════════════════════════════════════════════════════════
# Compute spectral diagnostics
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SPECTRAL DIAGNOSTICS")
print(f"{'='*70}")

results = []
for layer_idx in range(n_layers + 1):
    layer_data = {"layer": layer_idx, "is_moe": layer_idx in moe_layers, "domains": {}}
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
        sr = soft_rank(eigs)
        np.save(OUTPUT_DIR / f"eigs_layer{layer_idx:03d}_{domain}.npy", eigs)

        entry = {
            "N": int(N), "D": int(D), "r_mean": r,
            "alpha_ols": float(a) if a else None,
            "r2": float(r2) if r2 else None,
            "soft_rank": sr,
        }

        # Add routing stats for MoE layers
        if layer_idx in moe_layers and layer_idx in domain_routing[domain]:
            hist = domain_routing[domain][layer_idx]
            entry["expert_utilization"] = hist.tolist()
            entry["gating_entropy"] = gating_entropy(hist, moe_layers[layer_idx]['n_experts'])
            entry["max_gating_entropy"] = max_entropy(moe_layers[layer_idx]['n_experts'])
            entry["entropy_ratio"] = entry["gating_entropy"] / entry["max_gating_entropy"] if entry["max_gating_entropy"] > 0 else None
            entry["active_experts"] = int(np.sum(hist > 0))
            entry["top10_expert_share"] = float(np.sort(hist)[-10:].sum() / hist.sum()) if hist.sum() > 0 else None

            # Per-text routing detail averages
            details = domain_routing_detail[domain][layer_idx]
            if details:
                entry["mean_unique_experts_per_text"] = float(np.mean([d['unique_experts'] for d in details]))
                entry["mean_gating_entropy_per_text"] = float(np.mean([d['gating_entropy'] for d in details]))

        layer_data["domains"][domain] = entry
    results.append(layer_data)

    if layer_idx % 10 == 0 or layer_idx == n_layers:
        print(f"\n  Layer {layer_idx:3d} {'[MoE]' if layer_idx in moe_layers else '[Dense]'}:")
        for domain in DOMAINS:
            if domain in layer_data["domains"]:
                d = layer_data["domains"][domain]
                a_s = f"{d['alpha_ols']:.3f}" if d['alpha_ols'] else "N/A"
                r_s = f"{d['r_mean']:.4f}" if d['r_mean'] else "N/A"
                routing_str = ""
                if "gating_entropy" in d:
                    ent_ratio = d.get("entropy_ratio", 0)
                    active = d.get("active_experts", 0)
                    ent_str = f"{ent_ratio:.3f}" if ent_ratio else "N/A"
                    routing_str = f" H/Hmax={ent_str} active={active}"
                print(f"    {domain:<22} α={a_s:<8} ⟨r⟩={r_s:<8}{routing_str}")


# ═══════════════════════════════════════════════════════════
# Routing analysis: domain × layer expert overlap
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("ROUTING ANALYSIS: Domain Expert Overlap")
print(f"{'='*70}")

routing_analysis = {}
for layer_idx in sorted(moe_layers.keys()):
    if layer_idx % 10 != 0 and layer_idx != max(moe_layers.keys()):
        continue  # Sample every 10th MoE layer for readability

    print(f"\n  Layer {layer_idx}:")
    domain_expert_sets = {}
    for domain in DOMAINS:
        hist = domain_routing[domain][layer_idx]
        if hist.sum() == 0:
            continue
        # Top experts for this domain (those with >1% share)
        threshold = hist.sum() * 0.01
        top_experts = set(np.where(hist > threshold)[0])
        domain_expert_sets[domain] = top_experts
        n_active = np.sum(hist > 0)
        ent = gating_entropy(hist, moe_layers[layer_idx]['n_experts'])
        h_max = max_entropy(moe_layers[layer_idx]['n_experts'])
        print(f"    {domain:<22} active={n_active:<4} H/Hmax={ent/h_max:.3f}  top1%: {len(top_experts)} experts")

    # Compute pairwise Jaccard overlap between domain expert sets
    if len(domain_expert_sets) >= 2:
        overlaps = {}
        domain_list = list(domain_expert_sets.keys())
        for i in range(len(domain_list)):
            for j in range(i+1, len(domain_list)):
                d1, d2 = domain_list[i], domain_list[j]
                s1, s2 = domain_expert_sets[d1], domain_expert_sets[d2]
                if len(s1 | s2) > 0:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    overlaps[f"{d1}_vs_{d2}"] = round(jaccard, 3)

        # Key comparisons
        key_pairs = [
            ("greek_literary", "greek_documentary"),
            ("math.AG", "greek_documentary"),
            ("random_english", "cs.CL"),
        ]
        print(f"    --- Key overlaps (Jaccard) ---")
        for d1, d2 in key_pairs:
            key = f"{d1}_vs_{d2}"
            if key in overlaps:
                j = overlaps[key]
                print(f"      {d1} vs {d2}: J={j:.3f}")
            else:
                # Try reverse
                key2 = f"{d2}_vs_{d1}"
                if key2 in overlaps:
                    print(f"      {d1} vs {d2}: J={overlaps[key2]:.3f}")

        routing_analysis[layer_idx] = overlaps

# ═══════════════════════════════════════════════════════════
# Save everything
# ═══════════════════════════════════════════════════════════

trajectories = {}
for domain in DOMAINS:
    traj = {"alpha_ols": [], "r_mean": [], "soft_rank": [], "layers": [],
            "gating_entropy": [], "entropy_ratio": [], "active_experts": []}
    for lr in results:
        if domain in lr["domains"]:
            d = lr["domains"][domain]
            traj["layers"].append(lr["layer"])
            traj["alpha_ols"].append(d.get("alpha_ols"))
            traj["r_mean"].append(d.get("r_mean"))
            traj["soft_rank"].append(d.get("soft_rank"))
            traj["gating_entropy"].append(d.get("gating_entropy"))
            traj["entropy_ratio"].append(d.get("entropy_ratio"))
            traj["active_experts"].append(d.get("active_experts"))
    trajectories[domain] = traj

output = {
    "model": MODEL_NAME,
    "architecture": "DeepSeek-R1 MoE (671B total, ~37B active)",
    "quantization": "NF4 4-bit (bitsandbytes)",
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "n_moe_layers": len(moe_layers),
    "n_dense_layers": len(dense_layers),
    "moe_layer_indices": sorted(moe_layers.keys()),
    "dense_layer_indices": sorted(dense_layers),
    "n_experts": moe_layers[list(moe_layers.keys())[0]]['n_experts'] if moe_layers else None,
    "topk": moe_layers[list(moe_layers.keys())[0]]['topk'] if moe_layers else None,
    "domains": DOMAINS,
    "texts_per_domain": TEXTS_PER_DOMAIN,
    "total_time_s": round(total_time, 1),
    "load_time_s": round(load_time, 1),
    "layers": results,
    "trajectories": trajectories,
    "routing_analysis": {str(k): v for k, v in routing_analysis.items()},
}

# Save full results (might be large due to expert utilization histograms)
with open(OUTPUT_DIR / "deepseek_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

# Save compact trajectories
with open(OUTPUT_DIR / "trajectories.json", "w") as f:
    json.dump(trajectories, f, indent=2, default=str)

# Save routing analysis separately
routing_summary = {}
for domain in DOMAINS:
    domain_summary = {}
    for layer_idx in sorted(moe_layers.keys()):
        hist = domain_routing[domain][layer_idx]
        if hist.sum() > 0:
            n_experts = moe_layers[layer_idx]['n_experts']
            domain_summary[str(layer_idx)] = {
                "gating_entropy": gating_entropy(hist, n_experts),
                "entropy_ratio": gating_entropy(hist, n_experts) / max_entropy(n_experts),
                "active_experts": int(np.sum(hist > 0)),
                "top10_share": float(np.sort(hist)[-10:].sum() / hist.sum()),
            }
    routing_summary[domain] = domain_summary

with open(OUTPUT_DIR / "routing_analysis.json", "w") as f:
    json.dump(routing_summary, f, indent=2)

# Save per-layer expert utilization histograms as numpy
for layer_idx in sorted(moe_layers.keys()):
    for domain in DOMAINS:
        hist = domain_routing[domain][layer_idx]
        if hist.sum() > 0:
            np.save(OUTPUT_DIR / f"routing_layer{layer_idx:03d}_{domain}.npy", hist)

print(f"\n{'='*70}")
print(f"COMPLETE!")
print(f"Wall time: {(time.time() - t_load)/60:.0f} minutes")
print(f"Results: {OUTPUT_DIR}")
print(f"{'='*70}")
