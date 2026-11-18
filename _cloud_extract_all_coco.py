import sys
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path("/root/fgb_data/COCO")
for p in (BASE / "train2014", BASE / "val2014"):
    p.mkdir(parents=True, exist_ok=True)

for arg in sys.argv[1:]:
    p = Path(arg)
    pf = pq.ParquetFile(p)
    n = 0
    for batch in pf.iter_batches(batch_size=256, columns=["image"]):
        for r in batch.to_pylist():
            name = Path(r["image"]["path"]).name
            if name.startswith("COCO_train2014"):
                out = BASE / "train2014" / name
            elif name.startswith("COCO_val2014"):
                out = BASE / "val2014" / name
            else:
                continue
            out.write_bytes(r["image"]["bytes"])
            n += 1
    print(p.name, "wrote", n, flush=True)
