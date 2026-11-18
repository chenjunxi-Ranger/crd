"""What'sUp / COCO-spatial: multi-way caption choice.

Formats:
  coco_qa_one_obj.json   [coco_id, opt0(正确), opt1]                 n=2,247
  coco_qa_two_obj.json   [coco_id, opt0(正确), opt1]                 n=440
  controlled_images.json [{image_path, caption_options[0]=正确}]      n=412

Accuracy = fraction where argmax_j s(I, opt_j) == 0 (first option is correct).
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
WU = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\WhatsUp\whatsup_vlms-main")
DATA = WU / "data"
COCOIMG = Path(r"D:\PyCharm\PythonProject\异常标签检测\数据集\SugarCrepe\sugar-crepe-main"
               r"\data\coco\images\val2017")
WDIR = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Off(nn.Module):
    def __init__(self, d, width=64):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        return (self.a(u) + self.b(v) + self.b0).squeeze(-1)


def load_subset(subset: str):
    if subset in ("one", "two"):
        f = "coco_qa_one_obj.json" if subset == "one" else "coco_qa_two_obj.json"
        rows = json.load(open(DATA / f, encoding="utf-8"))
        items = []
        for r in rows:
            img = COCOIMG / f"{str(r[0]).zfill(12)}.jpg"
            opts = list(r[1:])
            if img.is_file():
                items.append({"img": str(img), "opts": opts})
        return items
    rows = json.load(open(DATA / "controlled_images_dataset.json", encoding="utf-8"))
    items = []
    for r in rows:
        p = Path(r["image_path"].replace("data/", str(WU / "data") + "/"))
        if p.is_file():
            items.append({"img": str(p), "opts": list(r["caption_options"])})
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="one", choices=["one", "two", "controlled"])
    ap.add_argument("--model", default="clip",
                    choices=["clip", "siglip2", "blipitc", "blipitm"])
    ap.add_argument("--head", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = load_subset(args.subset)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("no items (images missing?)", flush=True)
        return 1
    J = min(len(it["opts"]) for it in items)
    print(f"{args.subset}: {len(items)} items x {J} options", flush=True)

    paths = [it["img"] for it in items]
    texts = [it["opts"][j] for it in items for j in range(J)]
    n = len(items)

    from PIL import Image

    def run_backbone():
        if args.model == "clip":
            from src.encoders import JointEncoder
            e = JointEncoder(str(WDIR / "clip-vit-base-patch16"))
            m, proc, dev = e.model, e.processor, e.device
            f = torch.nn.functional
            U, V = [], []
            with torch.no_grad():
                for i in range(0, n, 64):
                    px = proc(images=[Image.open(x).convert("RGB") for x in paths[i:i + 64]],
                              return_tensors="pt")["pixel_values"].to(dev)
                    o = m.vision_model(pixel_values=px)
                    U.append(f.normalize(m.visual_projection(o.pooler_output.float()),
                                         dim=-1).cpu().numpy())
                for i in range(0, len(texts), 128):
                    tk = proc(text=texts[i:i + 128], return_tensors="pt", padding=True,
                              truncation=True, max_length=77).to(dev)
                    o = m.text_model(**tk)
                    V.append(f.normalize(m.text_projection(o.pooler_output.float()),
                                         dim=-1).cpu().numpy())
            U = np.concatenate(U); V = np.concatenate(V)
            s = np.array([(U[i] * V[i * J:(i + 1) * J]).sum(1) for i in range(n)])
            return s, np.repeat(U, J, axis=0), V

        if args.model == "siglip2":
            from transformers import AutoModel, AutoProcessor
            p = WDIR / "siglip2-base-patch16-224"
            m = AutoModel.from_pretrained(str(p)).to(DEV).eval()
            proc = AutoProcessor.from_pretrained(str(p))
            sc = m.logit_scale.exp().item()
            bias = m.logit_bias.item() if getattr(m, "logit_bias", None) is not None else 0.0
            f = torch.nn.functional
            U, V = [], []
            with torch.no_grad():
                for i in range(0, n, 64):
                    px = proc(images=[Image.open(x).convert("RGB") for x in paths[i:i + 64]],
                              return_tensors="pt")["pixel_values"].to(DEV)
                    U.append(f.normalize(m.get_image_features(pixel_values=px).float(),
                                         dim=-1).cpu().numpy())
                for i in range(0, len(texts), 128):
                    tk = proc(text=texts[i:i + 128], return_tensors="pt",
                              padding="max_length", truncation=True,
                              max_length=64).to(DEV)
                    V.append(f.normalize(m.get_text_features(input_ids=tk["input_ids"]).float(),
                                         dim=-1).cpu().numpy())
            U = np.concatenate(U); V = np.concatenate(V)
            raw = np.array([(U[i] * V[i * J:(i + 1) * J]).sum(1) for i in range(n)])
            return sc * raw + bias, np.repeat(U, J, axis=0), V

        from transformers import BlipForImageTextRetrieval, BlipProcessor
        p = WDIR / "blip-itm-base-coco"
        m = BlipForImageTextRetrieval.from_pretrained(str(p)).to(DEV).eval()
        proc = BlipProcessor.from_pretrained(str(p))
        f = torch.nn.functional
        U, V = [], []
        with torch.no_grad():
            for i in range(0, n, 48):
                px = proc(images=[Image.open(x).convert("RGB") for x in paths[i:i + 48]],
                          return_tensors="pt")["pixel_values"].to(DEV)
                h = m.vision_model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :]
                if hasattr(m, "vision_proj"):
                    h = m.vision_proj(h)
                U.append(f.normalize(h.float(), dim=-1).cpu().numpy())
            for i in range(0, len(texts), 64):
                tk = proc(text=texts[i:i + 64], return_tensors="pt", padding=True,
                          truncation=True, max_length=64).to(DEV)
                h = m.text_encoder(input_ids=tk["input_ids"],
                                   attention_mask=tk["attention_mask"],
                                   return_dict=True).last_hidden_state[:, 0, :]
                if hasattr(m, "text_proj"):
                    h = m.text_proj(h)
                V.append(f.normalize(h.float(), dim=-1).cpu().numpy())
        U = np.concatenate(U); V = np.concatenate(V)
        if args.model == "blipitc":
            s = np.array([(U[i] * V[i * J:(i + 1) * J]).sum(1) for i in range(n)])
            return s, np.repeat(U, J, axis=0), V

        cache = OUT / f"whatsup_{args.subset}_itm.npz"
        if cache.is_file():
            s = np.load(cache, allow_pickle=True)["s"]
            print("  loaded ITM cache", flush=True)
        else:
            t0 = time.time()
            # score option-by-option (each image reused J times)
            s = np.zeros((n, J))
            for j in range(J):
                col = []
                with torch.no_grad():
                    for i in range(0, n, 32):
                        idx = list(range(i, min(i + 32, n)))
                        px = proc(images=[Image.open(paths[k]).convert("RGB") for k in idx],
                                  return_tensors="pt")["pixel_values"].to(DEV)
                        tk = proc(text=[items[k]["opts"][j] for k in idx], return_tensors="pt",
                                  padding=True, truncation=True, max_length=64).to(DEV)
                        lg = m(pixel_values=px, input_ids=tk["input_ids"],
                               attention_mask=tk["attention_mask"],
                               use_itm_head=True).itm_score
                        col.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
                        if j == 0 and (i // 32) % 20 == 0:
                            print(f"   opt{j} {i}/{n} {time.time()-t0:.0f}s", flush=True)
                s[:, j] = np.concatenate(col)
            np.savez_compressed(cache, s=s)
        Vr = np.repeat(V.reshape(n, J, -1), 1, axis=1).reshape(n * J, -1)
        return s, np.repeat(U, J, axis=0), Vr

    s, Up, Vr = run_backbone()

    if args.head:
        ck = torch.load(args.head, map_location=DEV)
        head = Off(ck["dim"]).to(DEV)
        head.load_state_dict(ck["state"]); head.eval()
        lam, mu, sd = ck.get("lam", 1.0), ck["mu"], ck["sd"]
        with torch.no_grad():
            mh = head(torch.tensor(Up, device=DEV, dtype=torch.float32),
                      torch.tensor(Vr, device=DEV, dtype=torch.float32)).cpu().numpy()
        s = (s - mu) / sd - lam * mh.reshape(n, J)

    acc = float((s.argmax(1) == 0).mean())
    per_opt = (s.argmax(1) == 0)
    res = {"subset": args.subset, "model": args.model + ("+crd" if args.head else ""),
           "n": n, "options": J, "acc": round(acc, 4)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    tag = f"{args.subset}_{args.model}" + ("_crd" if args.head else "")
    (OUT / f"whatsup_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
