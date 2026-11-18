import pandas as pd

d = pd.read_parquet("fgbench/data/intermediate/vg/image_index.parquet")
need = set()
for split in ("train", "val", "test"):
    p = pd.read_parquet(f"fgbench/data/fgbench_v1_strict/{split}.parquet")
    v = p[p.dataset == "VG"].target_image_id.str.split(":").str[1].astype(int)
    need.update(v)
d = d[d.image_id.isin(need)]
d["url"] = "https://cs.stanford.edu/people/rak248/" + d.rel_path
d[["image_id", "rel_path", "url"]].to_csv("handoff_fgb/data/vg_urls.csv", index=False)
print("unique VG needed", len(d))
