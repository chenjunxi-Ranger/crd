#!/usr/bin/env bash
set -e
KEY=/home/chenjunxi/.ssh/id_ed25519_fgb
H=root@117.50.44.252
R="-e ssh -i $KEY -p 23 -o StrictHostKeyChecking=no -avP --partial"
D=/mnt/d/PyCharm/PythonProject/异常标签检测
ssh -i "$KEY" -p 23 -o StrictHostKeyChecking=no "$H" "mkdir -p /root/fgb_data/COCO/annotations /root/fgb_data/karpathy /root/fgb_handoff/p1"
rsync $R "$D/数据集/MS-COCO/annotations/captions_train2014.json" "$H:/root/fgb_data/COCO/annotations/"
rsync $R "$D/数据集/MS-COCO/annotations/captions_val2014.json" "$H:/root/fgb_data/COCO/annotations/"
rsync $R "$D/数据集/karpathy_split/dataset_coco.json" "$H:/root/fgb_data/karpathy/"
rsync $R "$D/数据集/karpathy_split/dataset_flickr30k.json" "$H:/root/fgb_data/karpathy/"
rsync $R "$D/mymodel7/handoff_fgb/fgbench_code.tar.gz" "$H:/root/fgb_handoff/"
rsync $R "$D/mymodel7/results/fgbench/p1/p1_protocol.json" "$H:/root/fgb_handoff/p1/"
rsync $R "$D/mymodel7/results/fgbench/p1/P1_DATA_REQUIREMENTS.md" "$H:/root/fgb_handoff/p1/"
echo P1_META_DONE
