#!/usr/bin/env bash
set -e
cd /mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7
export PYTHONPATH=/mnt/d/PyCharm/PythonProject/异常标签检测/mymodel7
python3 fgbench/experiments/p1_retrieval_full.py \
  --seeds 3 --noise 0.4 --topk 100 --budget 0.5 --eval_limit 1000
