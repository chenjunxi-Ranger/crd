import sys
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path("/root/fgb_data")
NEEDED = {
    "train": {Path(x).name for x in (BASE / "coco_train_files.txt").read_text().splitlines()},
    "val": {Path(x).name for x in (BASE / "coco_val_files.txt").read_text().splitlines()},
}
OUT = {"train": BASE / "COCO" / "train2014", "val": BASE / "COCO" / "val2014"}
for p in OUT.values():
    p.mkdir(parents=True, exist_ok=True)

files = [Path(x) for x in sys.argv[1:]]
count = {"train": 0, "val": 0}
for p in files:
    kind = "train" if "train-" in p.name else "val"
    needed = NEEDED[kind]
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=256, columns=["image"]):
        for r in batch.to_pylist():
            path = r["image"]["path"]
            if path in needed:
                (OUT[kind] / Path(path).name).write_bytes(r["image"]["bytes"])
                count[kind] += 1
    print(p.name, "cumulative", count, flush=True)
print("DONE", count)
