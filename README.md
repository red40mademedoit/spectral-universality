# Spectral Dimension as Content Fingerprint

**Random matrix theory applied to embedding eigenvalue spectra. The spacing ratio ⟨r⟩ measures model geometry. The power-law exponent α measures content.**

## Background

This project began with a falsified result. A [prior analysis](https://zenodo.org/) (DOI withdrawn) claimed that ⟨r⟩ in embedding Gram matrices classifies content domains. Subsequent testing revealed that ⟨r⟩ is **invariant to input content**: TF-IDF, SVD, NMF, character n-grams, hashing vectorizer, random projection, shuffled text, and encrypted text all produce the same ⟨r⟩ regime. That result was withdrawn.

The present work established that α (the power-law exponent of the eigenvalue tail), not ⟨r⟩, carries the signal. α diverges across domains, registers, authors, and languages — while ⟨r⟩ stays flat.

## Key Results

| Experiment | Finding |
|---|---|
| **KS p-values** (CSN bootstrap, B=500) | 24/25 arXiv categories pass power-law plausibility (p ≥ 0.10) |
| **LOOCV classification** | 52.5% accuracy on 25 categories (13× chance), 8 categories at 100% |
| **Word shuffle** (25 categories) | α inflates 3–6%, token count Δ = 0.0% — BPE hypothesis dead |
| **Papyri robustness** (10 × N=2000) | α = 0.6276 ± 0.0012, CV = 0.2% |
| **Temporal stability** (1000 years) | Documentary Koine α = 0.847 ± 0.007 across Ptolemaic/Roman/Late Antique |
| **Register ladder** (Koine Greek) | Documentary 0.87 → Literary papyri 1.00 → Literary authors 1.49 |
| **Internal spectroscopy** (6 models) | Activation α is domain-dependent; weight α is domain-independent |
| **Cross-model** (125M–72B params) | Documentary lowest α across all architectures tested |
| **SPhilBERTa crossover** | Domain specialist inverts spectral ordering; generalist preserves it |
| **Syntactic compression** | α_total = α_lexical − Δα_syntactic (novel decomposition) |

## Repository Structure

```
spectral-universality/
├── README.md
├── LICENSE
├── CITATION.cff
├── papers/
│   └── spectral-paper-v3.tex          # Current draft (March 2026)
├── shared/                             # Common RMT utilities
│   ├── rmt_utils.py                   # Spacing ratios, MP filtering, Brody fit
│   ├── plotting.py                    # Figure generation
│   └── constants.py                   # RMT thresholds
├── burst-validation-2026-03-07/        # H200 burst: arXiv + Greek validation
│   ├── 00_setup.sh                    # Environment setup
│   ├── 01_upload_data.sh              # Data transfer
│   ├── 01b_upload_greek.sh            # Greek data transfer
│   ├── 02_backfill_nulls.py           # Tests A–I (label shuffle, bootstrap, tail sweep, etc.)
│   ├── 02b_greek_validation.py        # Tests G1–G5 (dialect-quarantined Greek)
│   ├── 03_download_results.sh         # Result retrieval
│   └── extract_literary_by_author.py  # Literary corpus preparation
├── burst-b300-2026-03-07/              # B300 burst: expanded experiments
│   ├── 00_upload.sh                   # Data transfer to B300
│   ├── 01_setup.sh                    # Environment + dependency setup
│   ├── 02_n_dependence.py             # N-subsampling curve (N=50 to 10K)
│   ├── 03_word_shuffle_arxiv.py       # Per-category word shuffle (25 categories)
│   ├── 04_ks_pvalues.py              # CSN bootstrap KS p-values
│   ├── 05_papyri_robustness.py        # 10-draw subsampling robustness
│   ├── 06v2_internal_spectroscopy.py  # Qwen 72B activation + weight spectroscopy
│   ├── 07_loocv_classification.py     # Leave-one-out cross-validation
│   ├── 08_llama_spectroscopy.py       # Llama 3.3 70B activation spectroscopy
│   ├── 09_sphilberta_spectroscopy.py  # SPhilBERTa (domain specialist) spectroscopy
│   ├── 10_embedding_spectroscopy.py   # Nomic v2 MoE + PPLX 0.6B/4B spectroscopy
│   ├── 11_deepseek_spectroscopy.py    # DeepSeek R1 MoE routing (pending)
│   ├── run_all.sh                     # Sequential execution
│   └── run_cross_model.sh             # Cross-model batch
└── docs/
    ├── methodology.md                 # Spectral pipeline documentation
    └── cross-domain.md                # Cross-domain analysis notes
```

## Method

For an embedding matrix X ∈ ℝ^{N×D}:

1. **Center**: X̃ = X − X̄
2. **Row-normalize**: X̂ᵢ = X̃ᵢ / ‖X̃ᵢ‖
3. **Gram matrix**: G = X̂X̂ᵀ / D
4. **Eigendecompose**: λ₁ ≥ λ₂ ≥ ⋯ ≥ λ_N
5. **Signal separation**: λ₊ = (1 + √(N/D))²

**⟨r⟩** (spacing ratio): measures eigenvalue correlations. Invariant to content — measures model architecture only.

**α** (power-law exponent): measures the distribution of eigenvalue magnitudes. Content-dependent — diverges by domain, register, author, and language.

Three α estimators: OLS (log-log regression, top 30%), MLE (Clauset–Shalizi–Newman), Hill estimator.

## Models Tested (Internal Spectroscopy)

| Model | Architecture | Params | Documentary α (final layer) |
|---|---|---|---|
| Qwen 2.5-72B | Dense decoder, GQA | 72B | 0.647 |
| Llama 3.3-70B | Dense decoder, GQA | 70B | 0.769 |
| Nomic v2 MoE | Encoder + MoE | 475M | 1.104 |
| PPLX 0.6B | Encoder (Qwen3) | 596M | 1.854 |
| PPLX 4B | Encoder (Qwen3) | 4B | 1.914 |
| SPhilBERTa | Encoder (RoBERTa) | 125M | Inverted ordering |

## Corpora

- **arXiv**: 10,000 abstracts, 25 categories × 400 docs
- **DDB papyri**: 55,001 documentary texts (tax receipts, contracts, letters)
- **Koine literary**: 692 texts from 33 dialect-quarantined authors
- **Literary papyri (DCLP)**: 2,112 fragments

## Status

- ✅ All validation experiments complete (12/12)
- ✅ Paper v3.0 drafted
- ⏳ DeepSeek R1 MoE routing experiment (pending 8×H200 burst)
- ⏳ Figures for final submission

## Authors

- Joseph Hayden — Northland Tech Solutions / ShadowCorp LLC

## License

MIT
