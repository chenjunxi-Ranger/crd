#!/usr/bin/env bash
set -euo pipefail
BASE=/root/fgb_data
cd "$BASE/zips"
exec >>"$BASE/download.log" 2>&1
echo "=== ARIA START $(date -Is) ==="

if [ -f test_coco.zip ] && [ ! -f train2014.zip ]; then
  mv test_coco.zip train2014.zip
fi
if [ ! -f train2014.zip ]; then
  aria2c -x16 -s16 -k1M -c -o train2014.zip \
    http://images.cocodataset.org/zips/train2014.zip
fi
echo "[extract] train2014 $(date -Is)"
unzip -q -o train2014.zip -d "$BASE/COCO"
rm -f train2014.zip

echo "[2/4] val2014 $(date -Is)"
aria2c -x16 -s16 -k1M -c -o val2014.zip \
  http://images.cocodataset.org/zips/val2014.zip
unzip -q -o val2014.zip -d "$BASE/COCO"
rm -f val2014.zip

echo "[3/4] VG images $(date -Is)"
aria2c -x16 -s16 -k1M -c -o vg_images.zip \
  https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip -q -o vg_images.zip -d "$BASE/VisualGenome"
rm -f vg_images.zip

echo "[4/4] VG images2 $(date -Is)"
aria2c -x16 -s16 -k1M -c -o vg_images2.zip \
  https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip -q -o vg_images2.zip -d "$BASE/VisualGenome"
rm -f vg_images2.zip

echo "=== ARIA DONE $(date -Is) ==="
du -sh "$BASE/COCO" "$BASE/VisualGenome"
ls "$BASE/VisualGenome" | head
