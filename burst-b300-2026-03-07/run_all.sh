#!/bin/bash
# B300 Master Orchestrator
# Runs all priority experiments in optimal order.
# Dependencies: 02 must finish before 03/04/07 (they reuse embeddings)
#
# Estimated time: ~45 min total on B300
#   02: ~5 min  (embed 10K + N-dependence)
#   03: ~8 min  (word shuffle 25 categories × 2 embeds)
#   04: ~15 min (CSN bootstrap 25 categories × 500 bootstrap)
#   05: ~10 min (papyri 10 draws × embed 2K)
#   06: ~20 min (Qwen 72B internal spectroscopy 250 texts)
#   07: ~5 min  (LOOCV uses cached embeddings)
#
# Total cost at $7.91/hr: ~$6

set -e
cd "$(dirname "$0")"

LOG="run_all.log"
echo "=== B300 Burst Started $(date) ===" | tee $LOG

# Step 0: Setup
echo "" | tee -a $LOG
echo "=== STEP 0: SETUP ===" | tee -a $LOG
bash 01_setup.sh 2>&1 | tee -a $LOG

# Step 1: N-dependence (must run first — produces embeddings for later scripts)
echo "" | tee -a $LOG
echo "=== STEP 1: N-DEPENDENCE ===" | tee -a $LOG
python3 02_n_dependence.py 2>&1 | tee -a $LOG

# Step 2: Word shuffle per category
echo "" | tee -a $LOG
echo "=== STEP 2: WORD SHUFFLE ===" | tee -a $LOG
python3 03_word_shuffle_arxiv.py 2>&1 | tee -a $LOG

# Step 3: CSN KS p-values
echo "" | tee -a $LOG
echo "=== STEP 3: KS P-VALUES ===" | tee -a $LOG
python3 04_ks_pvalues.py 2>&1 | tee -a $LOG

# Step 4: Papyri robustness (independent, can run any time)
echo "" | tee -a $LOG
echo "=== STEP 4: PAPYRI ROBUSTNESS ===" | tee -a $LOG
python3 05_papyri_robustness.py 2>&1 | tee -a $LOG

# Step 5: Internal spectroscopy (longest, runs last)
echo "" | tee -a $LOG
echo "=== STEP 5: INTERNAL SPECTROSCOPY ===" | tee -a $LOG
python3 06_internal_spectroscopy.py 2>&1 | tee -a $LOG

# Step 6: LOOCV classification
echo "" | tee -a $LOG
echo "=== STEP 6: LOOCV CLASSIFICATION ===" | tee -a $LOG
python3 07_loocv_classification.py 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL DONE $(date) ===" | tee -a $LOG
echo "Results in results/*/"

# Tar everything for download
echo "Packaging results..."
tar czf results_b300_$(date +%Y%m%d_%H%M%S).tar.gz results/ run_all.log
echo "Package ready for download."
