from pathlib import Path

root = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\MS-COCO")
server = set(Path("server_coco_full.txt").read_text(encoding="utf-8").splitlines())
missing = []
for sub in ("train2014", "val2014"):
    for p in (root / sub).glob("*.jpg"):
        rel = f"{sub}/{p.name}"
        if rel not in server:
            missing.append(rel)
Path("coco_patch.txt").write_bytes("\n".join(missing).encode("utf-8"))
print("missing", len(missing), "train", sum(x.startswith("train") for x in missing),
      "val", sum(x.startswith("val") for x in missing))
