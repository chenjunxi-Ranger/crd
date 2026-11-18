import pyarrow.parquet as pq

p = "/root/fgb_data/hf/coco_inst_test0.parquet"
f = pq.ParquetFile(p)
print(f.schema_arrow)
print("rows", f.metadata.num_rows, "row_groups", f.metadata.num_row_groups)
t = f.read_row_group(0).slice(0, 1).to_pylist()[0]
for k, v in t.items():
    if isinstance(v, dict):
        print(k, {kk: (f"<{len(vv)} bytes>" if isinstance(vv, bytes) else vv)
                  for kk, vv in v.items()})
    else:
        print(k, v)
