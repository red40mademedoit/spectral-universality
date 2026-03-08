#!/bin/bash
# Upload everything to B300 instance.
# Usage: ./00_upload.sh <IP>
# Run with VPN OFF for max bandwidth.

set -e

IP="${1:?Usage: $0 <IP>}"
USER="shadeform"
DEST="~/"
BURST_DIR="/mnt/storage/projects/spectral-universality/burst-b300-2026-03-07"

echo "=== Uploading to B300 at $IP ==="

# 1. Upload arXiv corpus (22MB)
echo "1/4: arXiv corpus..."
scp ~/datasets/arxiv_spectral_corpus_10k.json ${USER}@${IP}:${DEST}/

# 2. Upload Greek data (already tarred from previous burst)
GREEK_TAR="/tmp/greek_data.tar.gz"
if [ ! -f "$GREEK_TAR" ]; then
    echo "2/4: Packing Greek data..."
    cd ~/Projects/stoikheia/corpus
    tar czf $GREEK_TAR extracted-papyri/
    # Add literary texts
    cd ~/data-greek-literary
    tar rzf $GREEK_TAR koine/
else
    echo "2/4: Using cached Greek tar..."
fi
echo "  Uploading Greek data..."
scp $GREEK_TAR ${USER}@${IP}:${DEST}/

# 3. Upload scripts
echo "3/4: Scripts..."
scp ${BURST_DIR}/*.py ${BURST_DIR}/*.sh ${USER}@${IP}:${DEST}/

# 4. Unpack Greek data on remote
echo "4/4: Unpacking Greek data on remote..."
ssh ${USER}@${IP} "
    mkdir -p ~/greek_data/ddb_extracted ~/greek_data/literary
    cd ~/greek_data
    tar xzf ~/greek_data.tar.gz
    # Move files to expected locations
    if [ -d extracted-papyri ]; then
        mv extracted-papyri/* ddb_extracted/ 2>/dev/null || true
    fi
    if [ -d koine ]; then
        mv koine/* literary/ 2>/dev/null || true
    fi
    echo 'DDB files:' \$(ls ddb_extracted/ | wc -l)
    echo 'Literary dirs:' \$(ls literary/ | wc -l)
    rm -f ~/greek_data.tar.gz
"

echo ""
echo "=== Upload complete ==="
echo "SSH in and run: nohup bash run_all.sh > ~/run.log 2>&1 &"
