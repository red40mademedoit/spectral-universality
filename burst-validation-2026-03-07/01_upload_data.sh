#!/bin/bash
# 01_upload_data.sh — Upload corpus + controls to burst instance
# Usage: ./01_upload_data.sh <IP>
# Run from Pop-OS. VPN OFF!
set -euo pipefail

IP="${1:?Usage: $0 <H200_IP>}"
SSH="ssh -o StrictHostKeyChecking=no root@${IP}"
RSYNC="rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no'"

echo "=== Uploading to ${IP} ==="
echo "VPN status — make sure it's OFF for speed:"
curl -s ifconfig.me || true
echo ""

# Create remote dirs
${SSH} "mkdir -p ~/data/controls"

# Upload corpus (22MB)
echo "--- Uploading arxiv corpus (22MB) ---"
eval ${RSYNC} ~/datasets/arxiv_spectral_corpus_10k.json root@${IP}:~/data/

# Upload shuffled texts (33MB total)
echo "--- Uploading control texts (33MB) ---"
eval ${RSYNC} ~/datasets/controls/*.embed.jsonl root@${IP}:~/data/controls/

# Upload the analysis script
echo "--- Uploading analysis scripts ---"
eval ${RSYNC} "$(dirname "$0")/02_backfill_nulls.py" root@${IP}:~/

echo ""
echo "=== Upload complete ==="
echo "Files on remote:"
${SSH} "ls -lh ~/data/ ~/data/controls/"
echo ""
echo "Now SSH in and run: python3 ~/02_backfill_nulls.py"
