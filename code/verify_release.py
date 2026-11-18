"""Consistency checks for the anonymous CRD/FGBench release."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"


def main() -> int:
    checks = []

    audit = json.loads((DATA / "audit" / "audit_reciprocal.json").read_text())
    checks.append(("audit passed", bool(audit["passed"])))
    checks.append(("full cycles = 88,024", audit["cycles"] == 88_024))

    all_df = pd.read_parquet(DATA / "manifests" / "all.parquet")
    test_df = pd.read_parquet(DATA / "manifests" / "test.parquet")
    checks.append(("full pairs = 352,096", len(all_df) == 352_096))
    checks.append(("test cycles = 17,565", test_df.cycle_id.nunique() == 17_565))

    main = pd.read_csv(RES / "paper_ready" / "main_table_strict.csv")
    crd = main[(main.method == "crd") & (main.facet == "Attribute")].iloc[0]
    lemon = main[(main.method == "lemon") & (main.facet == "Attribute")].iloc[0]
    crd_auc = float(crd.auroc_seedmean) if pd.notna(crd.auroc_seedmean) else float(crd.auroc)
    checks.append(("CRD Attribute AUROC", abs(crd_auc - 0.7187) < 1e-4))
    checks.append(("LEMoN official Attribute AUROC", abs(float(lemon.auroc) - 0.6755) < 1e-4))

    joint = pd.read_csv(RES / "paper_ready" / "joint_vs_additive.csv")
    checks.append(("joint rows = 24", len(joint) == 24))
    checks.append(("additive params", set(joint[joint["head"] == "additive"].params) == {65795}))
    checks.append(("joint params", set(joint[joint["head"] == "joint"].params) == {65665}))

    p1 = pd.concat([
        pd.read_csv(RES / "paper_ready" / "p1_negative" / n)
        for n in ("p1_classification_seeds0_2.csv", "p1_classification_seeds3_4.csv")
    ])
    checks.append(("P1 classification rows = 270", len(p1) == 270))
    for n in ("p1_hard.csv", "p1_retrieval_full.csv"):
        checks.append((f"exists {n}", (RES / "paper_ready" / "p1_negative" / n).is_file()))

    checks.append(("official LEMoN stats exists",
                   (RES / "paper_ready" / "official_lemon_stats.csv").is_file()))
    sup = pd.read_csv(RES / "paper_ready" / "official_baselines_supplement.csv")
    for facet in ("Entity", "Quantity", "Attribute", "Action", "Relation"):
        n = int(sup[(sup.method == "LEMoN") & (sup.facet == facet)].official_n_test.iloc[0])
        d = pd.read_csv(ROOT / "scores" / "official_lemon" / f"lemon_official_{facet}.csv")
        checks.append((f"official LEMoN test rows {facet}", int((d.split == "test").sum()) == n))

    for name, ok in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
