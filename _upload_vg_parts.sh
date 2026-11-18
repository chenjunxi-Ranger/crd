#!/usr/bin/env bash
set -e
SPLIT=/mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7/vg_split
VG=/mnt/d/PyCharm/PythonProject/异常标签检测/数据集/VisualGenome
KEY=/home/chenjunxi/.ssh/id_ed25519_fgb
DST=root@117.50.44.252:/root/fgb_data/zips/parts/
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" "$SPLIT/part_0" "$DST" &
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" "$SPLIT/part_1" "$DST" &
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" "$SPLIT/part_2" "$DST" &
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" "$SPLIT/part_3" "$DST" &
rsync -avP --partial -e "ssh -i $KEY -p 23 -o StrictHostKeyChecking=no" "$VG/images2.zip" "root@117.50.44.252:/root/fgb_data/zips/" &
wait
echo VG_PARTS_DONE
