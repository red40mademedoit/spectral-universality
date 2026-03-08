#!/bin/bash
# 03_download_results.sh — Download all results from burst instance
# Usage: ./03_download_results.sh <IP>
# Run from Pop-OS.
set -euo pipefail

IP="${1:?Usage: $0 <H200_IP>}"
LOCAL_DIR=~/burst-results-2026-03-07-validation
SSH="ssh -o StrictHostKeyChecking=no root@${IP}"
RSYNC="rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no'"

mkdir -p "${LOCAL_DIR}"

echo "=== Downloading results from ${IP} ==="

# Download everything
eval ${RSYNC} root@${IP}:~/results/ "${LOCAL_DIR}/"

echo ""
echo "=== Downloaded ==="
du -sh "${LOCAL_DIR}"
echo ""
echo "Contents:"
find "${LOCAL_DIR}" -type f | sort | while read f; do
    size=$(stat --printf="%s" "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    echo "  $(echo $f | sed "s|${LOCAL_DIR}/||")  $(numfmt --to=iec $size 2>/dev/null || echo "${size}B")"
done

echo ""
echo "=== Kill the instance! ==="
echo "Shadeform dashboard: https://platform.shadeform.ai/dashboard"
