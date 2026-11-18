"""SugarCrepe 2-way choice evaluation (+ optional CRD calibration).

Protocol: for each (I, T+, T-) choose argmax s(I,T). CRD replaces s by
  s' = (s - mu)/sd - lam * (a(u) + b(v) + b0)
with (mu, sd, head, lam) taken from the FGBench-trained head — no SugarCrepe
label is ever used for fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "fgbench"
SC = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SugarCrepe\sugar-crepe-main")
IMG = SC / "data" / "coco" / "images" / "val2017"
WDIR = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights")
SUBSETS = ["add_obj", "add_att", "replace_obj", "replace_att",
           "replace_rel", "swap_obj", "swap_att"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Off(nn.Module):
    def __init__(self, d, width=64):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        return (self.a(u) + self.b(v) + self.b0).squeeze(-1)


def encode_clip(caps_or_paths, is_text):
    from src.encoders import JointEncoder
    from PIL import Image
    e = JointEncoder(str(WDIR / "clip-vit-base-patch16"))
    m, proc, dev = e.model, e.processor, e.device
    out = []
    with torch.no_grad():
        for i in range(0, len(caps_or_paths), 64):
            b = caps_or_paths[i:i + 64]
            if is_text:
                tk = proc(text=b, return_tensors="pt", padding=True,
                          truncation=True, max_length=77).to(dev)
                o = m.text_model(**tk)
                h = m.text_projection(o.pooler_output.float())
            else:
                px = proc(images=[Image.open(p).convert("RGB") for p in b],
                          return_tensors="pt")["pixel_values"].to(dev)
                o = m.vision_model(pixel_values=px)
                h = m.visual_projection(o.pooler_output.float())
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    return np.concatenate(out), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="blipitm",
                    choices=["clip", "siglip2", "blipitc", "blipitm"])
    ap.add_argument("--head", default="", help="path to a saved CRD head (.pt)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = []
    for s in SUBSETS:
        raw = json.load(open(SC / "data" / f"{s}.json", encoding="utf-8"))
        vals = list(raw.values()) if isinstance(raw, dict) else raw
        for r in vals:
            rows.append({"subset": s, "file": r["filename"],
                         "pos": r["caption"], "neg": r["negative_caption"]})
    print(f"pairs {len(rows)}", flush=True)

    files = sorted({r["file"] for r in rows})
    paths = [str(IMG / f) for f in files]

    from PIL import Image
    if args.model == "clip":
        U, _ = encode_clip(paths, False)
        Vp, _ = encode_clip([r["pos"] for r in rows], True)
        Vn, _ = encode_clip([r["neg"] for r in rows], True)
        idx = {f: i for i, f in enumerate(files)}
        ui = np.array([idx[r["file"]] for r in rows])
        s_pos = (U[ui] * Vp).sum(1)
        s_neg = (U[ui] * Vn).sum(1)
        Up, Vp_all, Vn_all = U[ui], Vp, Vn
    elif args.model == "siglip2":
        from transformers import AutoModel, AutoProcessor
        p = WDIR / "siglip2-base-patch16-224"
        m = AutoModel.from_pretrained(str(p)).to(DEV).eval()
        proc = AutoProcessor.from_pretrained(str(p))
        sc = m.logit_scale.exp().item()
        bias = m.logit_bias.item() if getattr(m, "logit_bias", None) is not None else 0.0

        def enc_im(ps):
            o = []
            with torch.no_grad():
                for i in range(0, len(ps), 64):
                    px = proc(images=[Image.open(x).convert("RGB") for x in ps[i:i + 64]],
                              return_tensors="pt")["pixel_values"].to(DEV)
                    o.append(torch.nn.functional.normalize(
                        m.get_image_features(pixel_values=px).float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        def enc_tx(ts):
            o = []
            with torch.no_grad():
                for i in range(0, len(ts), 128):
                    tk = proc(text=list(ts[i:i + 128]), return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=64).to(DEV)
                    o.append(torch.nn.functional.normalize(
                        m.get_text_features(input_ids=tk["input_ids"]).float(),
                        dim=-1).cpu().numpy())
            return np.concatenate(o)

        U = enc_im(paths); idx = {f: i for i, f in enumerate(files)}
        ui = np.array([idx[r["file"]] for r in rows])
        Vp = enc_tx([r["pos"] for r in rows]); Vn = enc_tx([r["neg"] for r in rows])
        Up = U[ui]
        s_pos = sc * (Up * Vp).sum(1) + bias
        s_neg = sc * (Up * Vn).sum(1) + bias
    else:
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        p = WDIR / "blip-itm-base-coco"
        m = BlipForImageTextRetrieval.from_pretrained(str(p)).to(DEV).eval()
        proc = BlipProcessor.from_pretrained(str(p))

        def enc_im(ps):
            o = []
            with torch.no_grad():
                for i in range(0, len(ps), 48):
                    px = proc(images=[Image.open(x).convert("RGB") for x in ps[i:i + 48]],
                              return_tensors="pt")["pixel_values"].to(DEV)
                    h = m.vision_model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :]
                    if hasattr(m, "vision_proj"):
                        h = m.vision_proj(h)
                    o.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        def enc_tx(ts):
            o = []
            with torch.no_grad():
                for i in range(0, len(ts), 64):
                    tk = proc(text=list(ts[i:i + 64]), return_tensors="pt",
                              padding=True, truncation=True, max_length=64).to(DEV)
                    h = m.text_encoder(input_ids=tk["input_ids"],
                                       attention_mask=tk["attention_mask"],
                                       return_dict=True).last_hidden_state[:, 0, :]
                    if hasattr(m, "text_proj"):
                        h = m.text_proj(h)
                    o.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
            return np.concatenate(o)

        U = enc_im(paths); idx = {f: i for i, f in enumerate(files)}
        ui = np.array([idx[r["file"]] for r in rows])
        Up = U[ui]
        Vp = enc_tx([r["pos"] for r in rows]); Vn = enc_tx([r["neg"] for r in rows])

        if args.model == "blipitc":
            s_pos = (Up * Vp).sum(1); s_neg = (Up * Vn).sum(1)
        else:
            def itm_scores(ps, ts):
                o = []
                with torch.no_grad():
                    for i in range(0, len(ts), 32):
                        px = proc(images=[Image.open(x).convert("RGB")
                                          for x in ps[i:i + 32]],
                                  return_tensors="pt")["pixel_values"].to(DEV)
                        tk = proc(text=list(ts[i:i + 32]), return_tensors="pt",
                                  padding=True, truncation=True, max_length=64).to(DEV)
                        lg = m(pixel_values=px, input_ids=tk["input_ids"],
                               attention_mask=tk["attention_mask"],
                               use_itm_head=True).itm_score
                        o.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
                return np.concatenate(o)
            icache = OUT / "sugarcrepe_emb_blipitm.npz"
            if icache.is_file():
                z = np.load(icache, allow_pickle=True)
                Up, Vp, Vn = z["Up"], z["Vp"], z["Vn"]
                s_pos, s_neg = z["s_pos"], z["s_neg"]
                print("  loaded ITM score cache", flush=True)
            else:
                t0 = time.time()
                ims = [str(IMG / r["file"]) for r in rows]
                s_pos = itm_scores(ims, [r["pos"] for r in rows])
                print(f"  itm pos {time.time()-t0:.0f}s", flush=True)
                s_neg = itm_scores(ims, [r["neg"] for r in rows])
                print(f"  itm done {time.time()-t0:.0f}s", flush=True)
                np.savez_compressed(icache, Up=Up, Vp=Vp, Vn=Vn,
                                    s_pos=s_pos, s_neg=s_neg)

    # ---- optional CRD ----
    if args.head:
        ck = torch.load(args.head, map_location=DEV)
        head = Off(ck["dim"]).to(DEV)
        head.load_state_dict(ck["state"]); head.eval()
        lam, mu, sd = ck.get("lam", 1.0), ck["mu"], ck["sd"]
        with torch.no_grad():
            def mh(Ux, Vx):
                return head(torch.tensor(Ux, device=DEV, dtype=torch.float32),
                            torch.tensor(Vx, device=DEV, dtype=torch.float32)).cpu().numpy()
            s_pos = (s_pos - mu) / sd - lam * mh(Up, Vp)
            s_neg = (s_neg - mu) / sd - lam * mh(Up, Vn)

    ok = s_pos > s_neg
    res = {}
    for s in SUBSETS:
        m = np.array([r["subset"] == s for r in rows])
        res[s] = round(float(ok[m].mean()), 4)
    res["overall"] = round(float(ok.mean()), 4)
    res["model"] = args.model + ("+crd" if args.head else "")
    res["n"] = len(rows)

    # ---- split by whether the image was in BLIP's COCO fine-tuning pool ----
    KP = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\karpathy_split\dataset_coco.json")
    if KP.is_file():
        sp = {}
        for im in json.load(open(KP, encoding="utf-8"))["images"]:
            sp[int(im["cocoid"])] = "train" if im["split"] in ("train", "restval") \
                else im["split"]
        seen = np.array([sp.get(int(Path(r["file"]).stem)) == "train" for r in rows])
        res["n_seen"] = int(seen.sum())
        res["n_unseen"] = int((~seen).sum())
        res["acc_seen"] = round(float(ok[seen].mean()), 4) if seen.any() else None
        res["acc_unseen"] = round(float(ok[~seen].mean()), 4) if (~seen).any() else None
    print(json.dumps(res, ensure_ascii=False), flush=True)
    tag = args.tag or args.model + ("_crd" if args.head else "")
    (OUT / f"sugarcrepe_{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
