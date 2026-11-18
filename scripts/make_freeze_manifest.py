"""F9: freeze manifest - hashes of code, data, weights and the canonical results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights")

CODE = [
    "fgbench/models/crd_backbone.py", "fgbench/models/crd_blip.py",
    "fgbench/models/crd_action.py", "fgbench/models/gmm_filter.py",
    "fgbench/evaluate/strict_main_table.py", "fgbench/evaluate/final_stats_strict.py",
    "fgbench/evaluate/ablation_report.py", "fgbench/evaluate/lemon_fgbench.py",
    "fgbench/evaluate/traced_fgbench.py", "fgbench/evaluate/method_comparison_blip.py",
    "fgbench/evaluate/sugarcrepe_eval.py", "fgbench/evaluate/seetrue_eval.py",
    "fgbench/evaluate/whatsup_eval.py",
]
DATA = ["fgbench/data/fgbench_v1_strict/train.parquet",
        "fgbench/data/fgbench_v1_strict/val.parquet",
        "fgbench/data/fgbench_v1_strict/test.parquet",
        "fgbench/data/fgbench_v1_strict/strict_stats.json"]
RESULTS = ["results/fgbench/strict_main_table.csv",
           "results/fgbench/final_stats_strict.csv",
           "results/fgbench/ablation_standardised.csv",
           "results/fgbench/method_comparison_blipitc.csv",
           "results/fgbench/method_comparison_itm.csv",
           "results/fgbench/method_comparison.csv"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def main() -> int:
    out = {"code": {}, "data": {}, "weights": {}, "results": {}, "scores": {}}
    for rel in CODE:
        p = ROOT / rel
        out["code"][rel] = sha(p) if p.is_file() else None
    for rel in DATA:
        p = ROOT / rel
        out["data"][rel] = sha(p) if p.is_file() else None
    for rel in RESULTS:
        p = ROOT / rel
        out["results"][rel] = sha(p) if p.is_file() else None
    for d, tag in (("clip-vit-base-patch16", "clip"), ("clip-vit-large-patch14", "clipL"),
                   ("siglip2-base-patch16-224", "siglip2"),
                   ("blip-itm-base-coco", "blip"), ("flava-full", "flava")):
        p = W / d
        if not p.is_dir():
            continue
        w = {}
        for f in sorted(p.glob("*")):
            if f.suffix in (".safetensors", ".bin", ".json", ".txt"):
                w[f.name] = {"mb": round(f.stat().st_size / 1e6, 1)}
        out["weights"][tag] = w

    OUT = ROOT / "results" / "fgbench"
    for pat, key in (("crdbb_scores_clip_*_both_row_s*.npz", "crd_scores"),
                     ("crd_action_scores_s*.npz", "action_scores"),
                     ("score_*_clip_*.npz", "baseline_scores"),
                     ("*_head.pt", "heads")):
        files = sorted(OUT.glob(pat))
        out["scores"][key] = {"n": len(files),
                              "files": [f.name for f in files][:60]}

    p = ROOT / "results" / "fgbench" / "FREEZE_MANIFEST.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", p)
    print("code", sum(v is not None for v in out["code"].values()), "/", len(CODE))
    print("data", sum(v is not None for v in out["data"].values()), "/", len(DATA))
    print("results", sum(v is not None for v in out["results"].values()), "/", len(RESULTS))
    for k, v in out["scores"].items():
        print(f"  {k}: {v['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
