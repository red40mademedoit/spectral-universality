#!/bin/bash
# 01b_upload_greek.sh — Upload Greek corpus to burst instance
# Usage: ./01b_upload_greek.sh <IP>
# Run from Pop-OS. VPN OFF!
set -euo pipefail

IP="${1:?Usage: $0 <H200_IP>}"
SSH="ssh -o StrictHostKeyChecking=no root@${IP}"
RSYNC="rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no'"

PAPYRI_DIR="$HOME/Projects/stoikheia/corpus/extracted-papyri"

echo "=== Uploading Greek data to ${IP} ==="

# Create remote dirs
${SSH} "mkdir -p ~/data/greek/literary"

# Upload papyri texts — DDB is 237MB / 55K files, upload selectively
# DDB: subsample to ~5000 per period to keep upload manageable
echo "--- Uploading DDB papyri (full — 237MB, 55K files) ---"
eval ${RSYNC} "${PAPYRI_DIR}/ddb/" root@${IP}:~/data/greek/ddb/

echo "--- Uploading DCLP papyri (14MB, 2K files) ---"
eval ${RSYNC} "${PAPYRI_DIR}/dclp/" root@${IP}:~/data/greek/dclp/

# Upload metadata files
echo "--- Uploading metadata ---"
eval ${RSYNC} \
    "${PAPYRI_DIR}/_metadata_index.json" \
    "${PAPYRI_DIR}/_hgv_metadata.jsonl" \
    "${PAPYRI_DIR}/_extraction_stats.json" \
    root@${IP}:~/data/greek/

# Upload literary texts
# These need to be organized as ~/data/greek/literary/{author}/*.txt
# Check if they exist in the stoikheia corpus structure
LITERARY_SRC="$HOME/Projects/stoikheia/corpus"
if [ -d "${LITERARY_SRC}/canonical-greekLit" ]; then
    echo "--- Literary texts need pre-processing ---"
    echo "    canonical-greekLit is XML, not yet split by author."
    echo "    The burst script expects ~/data/greek/literary/{author}/*.txt"
    echo ""
    echo "    If you have pre-chunked literary texts, put them in:"
    echo "    ~/data/greek/literary/{plutarch,galen,lucian,...}/*.txt"
    echo "    and re-run this script."
    echo ""
    echo "    For now, the script will run papyri tests (G2, G3, G4)"
    echo "    but skip author-level tests (G1, G5) unless literary"
    echo "    texts are uploaded separately."
fi

# Upload the analysis script
echo "--- Uploading analysis script ---"
eval ${RSYNC} "$(dirname "$0")/02b_greek_validation.py" root@${IP}:~/

echo ""
echo "=== Upload complete ==="
echo "Files on remote:"
${SSH} "echo 'DDB:' && ls ~/data/greek/ddb/ | wc -l && echo 'files' && \
        echo 'DCLP:' && ls ~/data/greek/dclp/ | wc -l && echo 'files' && \
        echo 'Literary:' && ls ~/data/greek/literary/ 2>/dev/null | wc -l && echo 'dirs' && \
        echo 'Metadata:' && ls ~/data/greek/*.json ~/data/greek/*.jsonl 2>/dev/null"
echo ""
echo "Now SSH in and run: python3 ~/02b_greek_validation.py"
