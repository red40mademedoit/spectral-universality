#!/bin/bash
# Run cross-model spectroscopy: embedding models first, then Llama
# Embedding models are small enough to run alongside Qwen (if still loaded)
set -e

echo "=== Cross-Model Internal Spectroscopy ==="
echo "Models: Nomic v2 MoE, PPLX 0.6B, PPLX 4B, Llama 70B"
echo "Start: $(date)"
echo ""

# Ensure powerlaw is installed
pip install -q powerlaw 2>/dev/null

# ── Phase 1: Small embedding models (run with whatever's on GPU) ──
echo "=========================================="
echo "PHASE 1: Embedding models (Nomic, PPLX)"
echo "=========================================="

python3 -u 10_embedding_spectroscopy.py 2>&1 | tee embedding_spectroscopy.log

echo ""
echo "Phase 1 done at $(date)"
echo ""

# ── Phase 2: Llama 70B (needs full GPU) ──
echo "=========================================="
echo "PHASE 2: Llama 3.1 70B"
echo "=========================================="

# Kill any remaining Python processes using GPU (Qwen attention, SPhilBERTa, etc.)
echo "Clearing GPU for Llama..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if [ "$pid" != "$$" ]; then
        echo "  Killing GPU process $pid"
        kill -9 $pid 2>/dev/null || true
    fi
done
sleep 5
echo "GPU memory after cleanup:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

python3 -u 08_llama_spectroscopy.py 2>&1 | tee llama_spectroscopy.log

echo ""
echo "=========================================="
echo "ALL CROSS-MODEL SPECTROSCOPY COMPLETE"
echo "Finish: $(date)"
echo "=========================================="
