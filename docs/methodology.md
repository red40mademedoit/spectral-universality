# Unified Methodology

## The Pipeline

All four sub-projects follow the same core pipeline:

```
System → Matrix → Eigenvalues → Signal/Noise Separation → Spacing Statistics → Classification
```

### Step 1: Matrix Construction

| Domain | Input | Matrix | Type | Size |
|--------|-------|--------|------|------|
| Primes | π(x), ψ(x) | FFT spectrum | Real | M/2+1 |
| Embeddings | Documents | Gram matrix X·X^T | Real symmetric | N×N |
| Proteins | 3D structure | Hessian (elastic network) | Real symmetric | 3N×3N |
| Photosynthesis | Chromophore positions | Coupling Hamiltonian | Hermitian | n_chrom × n_chrom |

### Step 2: Eigenvalue Extraction

Standard diagonalization. For large matrices (proteins, embeddings), use LAPACK via numpy/scipy.

### Step 3: Signal/Noise Separation

**Marchenko-Pastur filtering** (embeddings): Compute upper edge λ_+ = σ²(1+√γ)². Eigenvalues above λ_+ are signal.

**Physical cutoff** (proteins): Remove lowest 6 modes (rigid body translation/rotation).

**Not needed** (primes): FFT amplitudes are the signal directly.

**Not applicable** (photosynthesis at N=7): All eigenvalues are signal — no bulk to separate.

### Step 4: Spacing Statistics

**Spacing ratios** ⟨r⟩ (preferred): Scale-invariant, no unfolding required.
```python
r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
```

**Brody parameter** q (alternative): Requires unfolding to unit mean spacing. Fit via MLE.

### Step 5: Classification

| ⟨r⟩ range | Brody q | Class | Interpretation |
|-----------|---------|-------|----------------|
| ≈ 0.386 | 0 | Poisson | Independent, integrable |
| ≈ 0.531 | ~0.7 | GOE | Time-reversal symmetric correlations |
| ≈ 0.603 | ~1.0 | GUE | Unitary, maximally correlated |

## Statistical Requirements

| Statistic | Minimum eigenvalues | Reliable eigenvalues | Precise eigenvalues |
|-----------|--------------------|--------------------|-------------------|
| ⟨r⟩ | 10 | 50 | 200+ |
| Brody q | 20 | 100 | 500+ |
| Full P(s) | 100 | 500 | 2000+ |

**Warning:** FMO (N=7, 6 spacings) is below the minimum for ALL statistics.

## Common Pitfalls

1. **Unfolding artifacts**: Polynomial unfolding can create spurious level repulsion in small samples.
2. **Finite-size effects**: Small matrices have systematic ⟨r⟩ bias. Use finite-N corrections or Monte Carlo calibration.
3. **MP edge at γ ≈ 1**: Marchenko-Pastur filtering is unreliable when N/D ≈ 1.
4. **Bootstrap ≠ accuracy**: Bootstrap CIs measure estimator precision, not model correctness.
5. **Brody q > 1**: Outside model domain. Use alternative repulsion models.
