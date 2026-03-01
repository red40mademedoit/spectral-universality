# 04 — RMT Statistics of Photosynthetic Hamiltonians

## Summary

Photosynthetic light-harvesting complexes (FMO, LHCII) show RMT universality class transitions that correlate with quantum transport efficiency. Wild-type systems trend toward GOE/GUE; mutants regress toward Poisson.

## Key Findings

- Wild-type FMO: q = 1.08 ± 0.10 (GOE-like)
- Mutant FMO: q = 0.80 ± 0.08 (toward Poisson)
- LHCII trimers: q = 1.51 (beyond Brody domain)
- Spearman ρ = 0.75 between q and transport efficiency (p < 0.01)

## CRITICAL ISSUES

### N=7 Problem (Severity: HIGH)
FMO has 7 chromophores → 7×7 Hamiltonian → 6 eigenvalue spacings. **You cannot reliably fit a Brody parameter from 6 data points.** True uncertainty is ~±0.5, not the ±0.10 from bootstrap. Bootstrap gives precision on the estimator, not accuracy of the model.

### q > 1 (Severity: HIGH)
Brody distribution is defined for 0 ≤ q ≤ 1. LHCII q = 1.51 is outside the model domain. This is not "enhanced level repulsion" — it's a fit extrapolating past its valid range. Need alternative repulsion models (e.g., power-law P(s) ∝ s^β for arbitrary β).

### Sample Size (Severity: MEDIUM)
ρ = 0.75 from ~7-10 complexes. Barely significant. Need 20+ systems for robust correlation.

### GUE Attractor Hypothesis (Severity: MEDIUM)
Suggestive pattern, no mechanism. Need:
- Controls (random Hamiltonians, non-photosynthetic chromophore systems)
- More examples across different organisms
- Physical argument for why GUE statistics → better transport

## Remediation Path

1. Use PSI (96 chlorophylls → 96×96), PSII, phycobilisomes for larger matrices
2. Replace Brody with power-law spacing model for q > 1 regime
3. Ensemble averaging: generate disorder realizations of each Hamiltonian
4. Control study: random Hermitian 7×7 matrices → what q distribution do you get?

## Status

Early. Needs significant methodological work before claims are defensible.
