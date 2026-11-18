import pyarrow.parquet as pq

p = "/root/fgb_data/hf/coco_test.parquet"
f = pq.ParquetFile(p)
print(f.schema_arrow)
print("rows", f.metadata.num_rows, "row_groups", f.metadata.num_row_groups)
t = f.read_row_group(0)
print(t.slice(0, 2).to_pylist())
