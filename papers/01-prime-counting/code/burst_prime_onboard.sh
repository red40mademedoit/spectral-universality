#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Prime Gap Manifold — B200 Onboard (Lean)
# ═══════════════════════════════════════════════════════════════
# Just primesieve + cupy + scipy. Nothing else.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══ PRIME GAP MANIFOLD — B200 ONBOARD ═══"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  GPU: $GPU ($VRAM, driver $DRIVER)"

# ── primesieve (apt) ─────────────────────────────────────────
echo "→ Installing primesieve..."
sudo apt-get update -qq
sudo apt-get install -y -qq primesieve libprimesieve-dev 2>/dev/null || {
    echo "  apt failed, compiling from source..."
    cd /tmp
    git clone --depth 1 https://github.com/kimwalisch/primesieve.git
    cd primesieve && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc) && sudo make install
    sudo ldconfig
    cd ~
}

# Verify primesieve
primesieve 1e6 --count
echo "  primesieve ✓"

# ── Python env ───────────────────────────────────────────────
echo "→ Setting up Python..."
python3 -m venv ~/env 2>/dev/null || true
source ~/env/bin/activate

# CRITICAL: kill megablocks (poisons sm_100/Blackwell CUDA)
pip uninstall -y megablocks deepspeed 2>/dev/null || true

pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# Core dependencies
echo "→ Installing numpy + scipy + cupy..."
pip install --no-cache-dir \
    numpy scipy matplotlib \
    2>&1 | tail -3

# cupy — cuda 12.x for B200
echo "→ Installing cupy-cuda12x..."
pip install --no-cache-dir cupy-cuda12x 2>&1 | tail -3

# primesieve python bindings
echo "→ Installing primesieve Python..."
pip install --no-cache-dir primesieve 2>&1 | tail -3

# ── Verify everything ───────────────────────────────────────
echo "→ Verification..."
python3 << 'PYEOF'
import sys
errors = []

try:
    import numpy as np
    print(f"  numpy {np.__version__} ✓")
except Exception as e:
    errors.append(f"numpy: {e}")

try:
    import cupy as cp
    n_gpu = cp.cuda.runtime.getDeviceCount()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"  cupy {cp.__version__}, {n_gpu} GPU(s) ✓")
    # Test cuFFT
    x = cp.random.randn(10_000_000, dtype=cp.float64)
    y = cp.fft.rfft(x)
    print(f"  cuFFT: 10M-point FFT → {len(y)} bins ✓")
    del x, y
    cp.get_default_memory_pool().free_all_blocks()
except Exception as e:
    errors.append(f"cupy/cuFFT: {e}")

try:
    from scipy.special import expi
    print(f"  scipy expi ✓")
except Exception as e:
    errors.append(f"scipy: {e}")

try:
    import primesieve
    p = primesieve.primes(1_000_000)
    print(f"  primesieve Python: {len(p)} primes up to 10⁶ ✓")
except Exception as e:
    errors.append(f"primesieve Python: {e}")
    print(f"  primesieve Python failed ({e}) — will use CLI")

if errors:
    print(f"\n  ERRORS: {errors}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED ✓")
PYEOF

# ── Workspace ────────────────────────────────────────────────
mkdir -p ~/prime-gap-manifold/results

echo ""
echo "═══ ONBOARD COMPLETE ═══"
echo "  source ~/env/bin/activate"
echo "  cd ~/prime-gap-manifold"
echo "  python3 per_zero_gpu.py --limit 1e9"
echo "  python3 per_zero_gpu.py --limit 1e10"
echo "  python3 per_zero_gpu.py --limit 1e11  # ~33GB RAM, ~10min"
