from pathlib import Path

import pandas as pd

out = Path("handoff_fgb/data")
rows = []
for split in ("train", "val", "test"):
    d = pd.read_parquet(f"fgbench/data/fgbench_v1_strict/{split}.parquet")
    rows.append(d[["dataset", "target_image_id", "target_file"]])
d = pd.concat(rows).drop_duplicates()
coco = d[d.dataset == "COCO"]
tr = sorted(set(coco[coco.target_file.str.startswith("train2014")].target_file))
va = sorted(set(coco[coco.target_file.str.startswith("val2014")].target_file))
(out / "coco_train_files.txt").write_text("\n".join(tr), encoding="utf-8")
(out / "coco_val_files.txt").write_text("\n".join(va), encoding="utf-8")
print("coco train", len(tr), "val", len(va))
