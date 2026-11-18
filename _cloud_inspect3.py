import pyarrow.parquet as pq

p = "/root/fgb_data/hf/coco_inst_train0.parquet"
f = pq.ParquetFile(p)
print("rows", f.metadata.num_rows)
paths = []
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i, columns=["image"]).to_pylist()
    paths.extend([r["image"]["path"] for r in t[:50]])
print(paths[:10])
print("val rows in sample:", sum("val2014" in (x or "") for x in paths))
print("train rows in sample:", sum("train2014" in (x or "") for x in paths))
