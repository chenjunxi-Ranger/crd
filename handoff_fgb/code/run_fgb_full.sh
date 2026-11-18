#!/usr/bin/env bash
# Full FGBench TRACED run. Place this file in the _traced root and run:
#   tmux new-session -d -s traced_full "bash run_fgb_full.sh"
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p full_logs
export PYTHONUNBUFFERED=1
OUT="${TRACED_OUT:-traced_full}"

for ds in fgb_quantity fgb_relation fgb_action fgb_entity fgb_attr; do
  echo "=== START $ds $(date -Is) ==="
  for step in create_trajectory add_text_sim train_classifiers; do
    log="full_logs/${ds}_${step}.log"
    echo "--- $ds $step $(date -Is) ---"
    python3 "$step.py" \
      --output_dir "$OUT" \
      --dataset "$ds" \
      --noise_type cat \
      --noise_level 0.5 \
      --exploration_strat 0 \
      --top_k_val 8 \
      --alignment_metric BLIP \
      --batch_size_GCD "${TRACED_BATCH:-8}" \
      --num_steps_GCD 3 \
      --match_head itm \
      --seed 0 > "$log" 2>&1
    rc=$?
    echo "--- $ds $step rc=$rc $(date -Is) ---"
    if [ "$rc" -ne 0 ]; then
      echo "FAIL $ds $step; tail:"
      tail -40 "$log"
      exit 1
    fi
  done
done
echo "ALL_DONE $(date -Is)"
