#!/bin/bash
# 00_setup.sh — H200 environment setup (~2 min)
# Run first after SSH into burst instance
set -euo pipefail

echo "=== H200 Burst Validation Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

# ── Kill megablocks (ALWAYS FIRST) ──────────────────────────
pip uninstall -y megablocks deepspeed 2>/dev/null || true
echo "✓ megablocks/deepspeed removed"

# ── Core deps ───────────────────────────────────────────────
pip install -q \
    numpy scipy \
    sentence-transformers \
    torch \
    powerlaw \
    tqdm

# powerlaw = Clauset-Shalizi-Newman MLE implementation
# sentence-transformers pulls in transformers + tokenizers

echo "✓ Dependencies installed"

# ── Verify Nomic model downloads ────────────────────────────
python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading Nomic v2 MoE...')
model = SentenceTransformer('nomic-ai/nomic-embed-text-v2-moe', trust_remote_code=True)
print(f'✓ Nomic loaded, dim={model.get_sentence_embedding_dimension()}')
"

# ── Create output dirs ──────────────────────────────────────
mkdir -p ~/results/{eigenvalues,alpha_nulls,bootstrap,tail_sweep,matryoshka,gram_vs_cov,mp_validation,token_counts}
echo "✓ Output directories created"

echo ""
echo "=== Setup complete. Run 01_upload_data.sh next. ==="
