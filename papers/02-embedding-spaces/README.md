# 02 — Spectral Taxonomy of Document Embedding Spaces

## Summary

Classify document corpora by the eigenvalue spacing statistics of their embedding similarity matrices. ⟨r⟩ reveals organizational architecture (independent → correlated → entangled), not topic.

## Key Findings

- C00 (HEP-th pure): ⟨r⟩ = 0.627 ± 0.032 → GUE → dense cross-references
- C08 (Quantum physics): ⟨r⟩ = 0.473 ± 0.027 → Poisson → self-contained results
- Same topic (QFT) appears in both GUE and intermediate regimes
- C07 (spectral theory): GUE statistics — papers about eigenvalue distributions have eigenvalue distributions following RMT

## Known Issues

- γ = N/D ≈ 0.973 sits on MP critical edge — finite-size effects may dominate
- Empirical Poisson threshold (0.50) shifted from theoretical (0.386) — needs investigation
- Need γ ≪ 1 and γ ≫ 1 controls

## Status

Draft. Code and data need migration from spectral observatory pipeline.

## Source Material

- Spectral Observatory: `/mnt/storage/projects/rag-pipeline/scripts/spectral_observatory.py`
- arXiv embeddings: previously on Milvus (KVM8, now nuked) — need re-embedding
