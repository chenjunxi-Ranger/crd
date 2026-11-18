#!/usr/bin/env bash
set -euo pipefail
BASE=/root/fgb_data
mkdir -p "$BASE/COCO" "$BASE/VisualGenome" "$BASE/zips"
cd "$BASE/zips"
log="$BASE/download.log"
exec >>"$log" 2>&1

echo "=== START $(date -Is) ==="

echo "[1/4] COCO train2014 $(date -Is)"
wget -c --tries=5 --timeout=60 -O train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip
unzip -q -o train2014.zip -d "$BASE/COCO"
rm -f train2014.zip

echo "[2/4] COCO val2014 $(date -Is)"
wget -c --tries=5 --timeout=60 -O val2014.zip \
  http://images.cocodataset.org/zips/val2014.zip
unzip -q -o val2014.zip -d "$BASE/COCO"
rm -f val2014.zip

echo "[3/4] VG images $(date -Is)"
wget -c --tries=5 --timeout=60 -O vg_images.zip \
  https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip -q -o vg_images.zip -d "$BASE/VisualGenome"
rm -f vg_images.zip

echo "[4/4] VG images2 $(date -Is)"
wget -c --tries=5 --timeout=60 -O vg_images2.zip \
  https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip -q -o vg_images2.zip -d "$BASE/VisualGenome"
rm -f vg_images2.zip

echo "=== DONE $(date -Is) ==="
du -sh "$BASE/COCO" "$BASE/VisualGenome"
