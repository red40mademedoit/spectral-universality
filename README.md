# Spectral Universality Project

**One method. Four domains. One question: do eigenvalue spacing statistics reveal universal organizational principles?**

## The Method

Given any system that produces a similarity or interaction matrix:

1. Compute the eigenvalue spectrum
2. Separate signal from noise (Marchenko-Pastur edge)
3. Compute spacing ratios ⟨r⟩ or Brody parameter q
4. Classify: Poisson (independent) → GOE (correlated) → GUE (maximally entangled)

## The Domains

| # | Domain | Matrix | Key Finding | Status |
|---|--------|--------|-------------|--------|
| 01 | **Prime Counting Functions** | FFT of π(x)-Li(x) | 50/50 zeros detected at SNR 5-91; amplitude decay r=0.998 matches 1/\|ρ\| | Data collected |
| 02 | **Embedding Spaces** | Document similarity (Gram) | ⟨r⟩ classifies organizational structure, not topic | Draft |
| 03 | **Protein Vibrations** | Hessian normal modes (AF3) | AlphaFold encodes dynamical info without training | Draft |
| 04 | **Photosynthetic Hamiltonians** | Chromophore coupling | Transport efficiency correlates with Brody q (ρ=0.75) | Early |

## The Cross-Domain Story

```
Localized / Independent                    Delocalized / Entangled
        Poisson (⟨r⟩ ≈ 0.386)  →  GOE (⟨r⟩ ≈ 0.536)  →  GUE (⟨r⟩ ≈ 0.603)

Primes:     Gap FFT (no detection)                          Counting fn (50/50)
Embeddings: Quantum physics (0.47)    Stat mech (0.53)      String theory (0.63)
Proteins:   Melanin pathway (q≈0.1)                         Structural (q≈0.67)
Photosyn:   Mutant FMO (q≈0.8)       Wild-type FMO (q≈1.1) LHCII trimer (q≈1.5*)

* q > 1 exceeds Brody model domain — needs investigation
```

## Known Issues / Open Problems

- **FMO N=7 problem:** 7×7 Hamiltonian → 6 spacing ratios. Brody fit from 6 points has ~±0.5 true uncertainty. Bootstrap CIs underestimate this.
- **Brody q > 1:** Outside model domain (0 ≤ q ≤ 1). LHCII q=1.51 requires alternative repulsion model.
- **γ ≈ 1 finite-size effects:** Embedding Gram matrices at N/D ≈ 1 sit on the Marchenko-Pastur critical edge. Empirical threshold shifts may be artifacts.
- **GUE attractor hypothesis:** Suggestive but lacks mechanism and sufficient sample size. Needs controls (random/non-optimized systems).
- **Cross-domain "universality":** RMT classes appearing in multiple systems is expected (that's what universality means). The open question is whether transitions between classes carry equivalent meaning across domains.

## Repository Structure

```
spectral-universality/
├── README.md                          # This file
├── papers/
│   ├── 01-prime-counting/             # Riemann zero detection via counting functions
│   ├── 02-embedding-spaces/           # arXiv spectral taxonomy
│   ├── 03-protein-vibrations/         # AF3 normal mode RMT
│   └── 04-photosynthesis/             # FMO/LHCII Hamiltonian statistics
├── shared/                            # Common RMT utilities
│   ├── rmt_utils.py                   # Spacing ratios, Brody fit, MP filtering
│   ├── plotting.py                    # Consistent figure style
│   └── constants.py                   # Known zeros, RMT thresholds
└── docs/
    ├── methodology.md                 # Unified methodology
    └── cross-domain.md                # Cross-domain analysis
```

## Authors

- Shadow (Joseph Hayden) — NTS / ShadowCorp LLC
- Dreadbot 3.2.666 — Reasoning engine

## License

TBD
