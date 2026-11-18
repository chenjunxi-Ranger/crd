#!/usr/bin/env bash
set -euo pipefail
BASE=/root/fgb_data/hf
mkdir -p "$BASE"
cd "$BASE"
exec >>/root/fgb_data/coco_hf.log 2>&1
echo "=== COCO HF START $(date -Is) ==="
for f in \
  train-00000-of-00007-972b4bfe28bd4e4b.parquet \
  train-00001-of-00007-b2d8dc89d7110f6b.parquet \
  train-00002-of-00007-cea370f4c405743f.parquet \
  train-00003-of-00007-06a8b48c32ddd2e4.parquet \
  train-00004-of-00007-e946754bb226cf87.parquet \
  train-00005-of-00007-b5645027ed2d18ba.parquet \
  train-00006-of-00007-928b11e847b901a9.parquet \
  validation-00000-of-00004-a5d64955dc6bcdbf.parquet \
  validation-00001-of-00004-6a8ae074a6390bc7.parquet \
  validation-00002-of-00004-67875abc9f143cfa.parquet \
  validation-00003-of-00004-28b9cf5b5238f451.parquet ; do
  echo "--- $f $(date -Is)"
  aria2c -x16 -s16 -k1M -c -o "$f" \
    "https://hf-mirror.com/datasets/NaiveDev/coco-2014-instance/resolve/main/data/$f"
done
echo "=== COCO HF DOWNLOADED $(date -Is) ==="
du -sh "$BASE"
