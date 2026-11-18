"""Measure Figure 2 geometry: box sizes, arrow-neighbour consistency, panel symmetry.

Run:  python scripts/audit_fig2_geometry.py
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fgbench" / "figures" / "fig2_crd_method.py"

spec = importlib.util.spec_from_file_location("fig2", SRC)
mod = importlib.util.module_from_spec(spec)
sys.modules["fig2"] = mod
spec.loader.exec_module(mod)

BOXES, ARROWS = [], []
_orig_box = mod.box


def rec_box(fig, x, y, w, h, **kw):
    BOXES.append({"x": round(x, 4), "y": round(y, 4),
                  "w": round(w, 4), "h": round(h, 4)})
    return _orig_box(fig, x, y, w, h, **kw)


_orig_arrow = mod.arrow


def rec_arrow(fig, x1, y1, x2, y2, **kw):
    ARROWS.append((round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)))
    return _orig_arrow(fig, x1, y1, x2, y2, **kw)


mod.box = rec_box
mod.arrow = rec_arrow
mod.main()

BW, BH = mod.BW, mod.BH
DX = mod.PANEL_DX
fail = []

print(f"boxes: {len(BOXES)}   arrows: {len(ARROWS)}")
print("\n[1] box sizes")
for (w, h), n in sorted(Counter((b["w"], b["h"]) for b in BOXES).items(),
                        key=lambda kv: -kv[1]):
    tag = "" if (w, h) == (BW, BH) else "   <- non-standard"
    print(f"    {w:.3f} x {h:.3f}  x{n}{tag}")

print("\n[2] horizontal arrows: equal height?")
bad = 0
for (x1, y1, x2, y2) in ARROWS:
    if abs(y1 - y2) < 1e-9 and abs(x1 - x2) > 1e-9:
        src = [b for b in BOXES if abs(b["y"] + b["h"] / 2 - y1) < 1e-6
               and abs(b["x"] + b["w"] - x1) < 0.021]
        dst = [b for b in BOXES if abs(b["y"] + b["h"] / 2 - y1) < 1e-6
               and abs(b["x"] - x2) < 0.021]
        if src and dst and abs(src[0]["h"] - dst[0]["h"]) > 1e-6:
            bad += 1
            print(f"    MISMATCH h {src[0]['h']} vs {dst[0]['h']} at y={y1}")
print(f"    violations: {bad}")
fail.append(bad)

print("\n[3] vertical arrows: equal width?")
bad = 0
for (x1, y1, x2, y2) in ARROWS:
    if abs(x1 - x2) < 1e-9 and abs(y1 - y2) > 1e-9:
        src = [b for b in BOXES if abs(b["x"] + b["w"] / 2 - x1) < 1e-6
               and abs(b["y"] - y2) < 0.021]
        dst = [b for b in BOXES if abs(b["x"] + b["w"] / 2 - x1) < 1e-6
               and abs(b["y"] + b["h"] - y1) < 0.021]
        if src and dst and abs(src[0]["w"] - dst[0]["w"]) > 1e-6:
            bad += 1
            print(f"    MISMATCH w {src[0]['w']} vs {dst[0]['w']} at x={x1}")
print(f"    violations: {bad}")
fail.append(bad)

print("\n[4] panel symmetry (shared modules must differ by exactly one offset)")
A = [b for b in BOXES if b["x"] < 0.5]
B = [b for b in BOXES if b["x"] >= 0.5]
offsets, unmatched_a, used_b = set(), [], set()
for b in A:
    cand = [i for i, c in enumerate(B) if i not in used_b
            and abs(c["y"] - b["y"]) < 1e-6 and abs(c["w"] - b["w"]) < 1e-6
            and abs(c["h"] - b["h"]) < 1e-6]
    if not cand:
        unmatched_a.append(b)
        continue
    i = cand[0]
    used_b.add(i)
    offsets.add(round(B[i]["x"] - b["x"], 4))
unmatched_b = [c for i, c in enumerate(B) if i not in used_b]
print(f"    distinct offsets: {sorted(offsets)}")
print(f"    panel-specific in (a): {len(unmatched_a)}  in (b): {len(unmatched_b)}")
for c in unmatched_b:
    print(f"      (b) only: x={c['x']:.3f} y={c['y']:.3f} "
          f"{c['w']:.3f}x{c['h']:.3f}")
ok = len(offsets) == 1 and offsets == {round(DX, 4)}
print(f"    single offset == PANEL_DX({DX}) ? {ok}")
fail.append(0 if ok else 1)

print("\n[5] additive block spans exactly two rows")
add = [b for b in BOXES if abs(b["h"] - (BH + mod.ROW_PITCH)) < 1e-6]
print(f"    found {len(add)} additive blocks, h={BH + mod.ROW_PITCH:.3f} "
      f"(BH + row pitch) -> {'OK' if len(add) == 2 else 'CHECK'}")

print("\nRESULT:", "PASS" if sum(fail) == 0 else f"FAIL ({sum(fail)} issues)")
