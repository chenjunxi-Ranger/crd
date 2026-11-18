"""Zero-shot image-text retrieval R@1 on the cached 5k COCO val features."""
import numpy as np
import pandas as pd

rows = pd.read_parquet("results/fgbench/lemon_protocol/rows_frac0.40.parquet")
z = np.load("results/fgbench/lemon_protocol/features.npz", allow_pickle=True)
img = dict(zip(z["img_ids"].astype(str), z["img_vec"]))
txt = dict(zip(z["txt_keys"], z["txt_vec"]))

# unique images, unique captions and their owning image
imgs = rows.drop_duplicates("image_id")[["image_id"]].reset_index(drop=True)
caps = rows.drop_duplicates("caption")[["caption", "image_id"]].reset_index(drop=True)
caps = caps[caps.caption.isin(txt)].reset_index(drop=True)
imgs = imgs[imgs.image_id.astype(str).isin(img)].reset_index(drop=True)

U = np.stack([img[str(i)] for i in imgs.image_id]).astype("float32")
V = np.stack([txt[c] for c in caps.caption]).astype("float32")
U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
owner = caps.image_id.to_numpy()

# image -> text
hit = 0
for i in range(0, len(U), 512):
    s = U[i:i + 512] @ V.T
    top = s.argmax(1)
    hit += int((owner[top] == imgs.image_id.to_numpy()[i:i + 512]).sum())
i2t = hit / len(U)

# text -> image
hit = 0
for i in range(0, len(V), 1024):
    s = V[i:i + 1024] @ U.T
    top = s.argmax(1)
    hit += int((imgs.image_id.to_numpy()[top] == owner[i:i + 1024]).sum())
t2i = hit / len(V)

print(f"images={len(U)} captions={len(V)}")
print(f"R@1 image->text = {i2t:.4f}")
print(f"R@1 text->image = {t2i:.4f}")
