#!/usr/bin/env bash
set -e
SPLIT=/mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7/vg_split
SPLIT2=/mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7/vg_split2
H=root@117.50.44.252
SSH="ssh -i /home/chenjunxi/.ssh/id_ed25519_fgb -p 23 -o StrictHostKeyChecking=no -o ControlPath=/tmp/fgbssh-%r@%h:%p"
for f in "$SPLIT"/part_0 "$SPLIT"/part_1 "$SPLIT"/part_2 "$SPLIT"/part_3; do
  rsync -avP --partial -e "$SSH" "$f" "$H:/root/fgb_data/zips/parts/" &
done
for f in "$SPLIT2"/i2_0 "$SPLIT2"/i2_1 "$SPLIT2"/i2_2; do
  rsync -avP --partial -e "$SSH" "$f" "$H:/root/fgb_data/zips/parts_i2/" &
done
wait
echo VG_7_DONE
