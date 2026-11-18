import sys

import pyarrow.parquet as pq

p = sys.argv[1]
f = pq.ParquetFile(p)
print(f.schema_arrow)
print("rows", f.metadata.num_rows, "row_groups", f.metadata.num_row_groups)
t = f.read_row_group(0).slice(0, 1).to_pylist()[0]
for k, v in t.items():
    if isinstance(v, dict):
        print(k, {kk: (f"<{len(vv)} bytes>" if isinstance(vv, bytes) else vv)
                  for kk, vv in v.items()})
    elif isinstance(v, list):
        print(k, f"<list len {len(v)}>", v[:2])
    else:
        print(k, v)
