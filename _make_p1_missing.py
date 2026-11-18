from pathlib import Path

root = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\MS-COCO")
server = set(Path("server_coco.txt").read_text(encoding="utf-8").splitlines())
missing = []
for sub in ("train2014", "val2014"):
    for p in (root / sub).glob("*.jpg"):
        rel = f"{sub}/{p.name}"
        if rel not in server:
            missing.append(rel)
Path("p1_coco_missing.txt").write_text("\n".join(missing), encoding="utf-8")
print("missing", len(missing))
print("train", sum(x.startswith("train2014") for x in missing),
      "val", sum(x.startswith("val2014") for x in missing))
