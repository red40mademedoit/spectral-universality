#!/bin/bash
# B300 setup — run first after SSH
set -e

echo "=== B300 Burst Setup ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Kill bloatware
pip uninstall -y megablocks deepspeed --break-system-packages 2>/dev/null || true

# Install dependencies
pip install --break-system-packages --quiet sentence-transformers powerlaw numpy scipy torch transformers accelerate einops

# Verify
python3 -c "
import torch, numpy, scipy, powerlaw
from sentence_transformers import SentenceTransformer
print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
print(f'numpy {numpy.__version__}, scipy {scipy.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('All deps OK')
"

echo "=== Setup complete ==="
