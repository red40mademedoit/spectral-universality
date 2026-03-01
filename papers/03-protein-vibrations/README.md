# 03 — RMT Classification of Protein Vibrations from AlphaFold 3 Structures

## Summary

Analyze normal mode spectra of AlphaFold 3 predicted structures via Brody parameter q. AF3 encodes dynamical information (vibrational statistics) without explicit training on dynamics.

## Key Findings

- Melanin pathway proteins: q ≈ 0.1 (near-Poisson) → localized dynamics
- Helical/structural proteins: q up to 0.67 → correlated, collective vibrations
- Brody q differentiates functional classes
- AF3 internal representations capture more than static coordinates

## Known Issues

- Need larger protein set for statistical power
- Hessian construction method affects results (elastic network vs all-atom)
- Brody fit reliability depends on matrix size (3N×3N for N residues — generally large enough)

## Status

Draft. 22→38 proteins analyzed (Paper v3.1). Zenodo package prepared.

## Source Material

- Previous analysis: session 2026-02-21
- Zenodo archive: packed but not yet submitted
