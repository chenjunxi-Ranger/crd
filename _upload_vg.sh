#!/usr/bin/env bash
set -e
SRC=/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/VisualGenome
KEY=/home/chenjunxi/.ssh/id_ed25519_fgb
DST=root@117.50.44.252:/root/fgb_data/zips/
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" \
  "$SRC/images.zip" "$DST" &
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" \
  "$SRC/images2.zip" "$DST" &
wait
echo VG_UPLOAD_DONE
