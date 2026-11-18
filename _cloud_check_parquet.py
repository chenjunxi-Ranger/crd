import sys

import pyarrow.parquet as pq

for p in sys.argv[1:]:
    try:
        f = pq.ParquetFile(p)
        for i in range(f.metadata.num_row_groups):
            f.read_row_group(i, columns=["image"]).to_pylist()
        print("OK", p)
    except Exception as e:
        print("FAIL", p, type(e).__name__, e)
