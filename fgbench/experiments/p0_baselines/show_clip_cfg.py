"""Print vision-tower config of the local CLIP checkpoints (patch size etc.)."""
import json
from pathlib import Path

W = Path(r"D:\PyCharm\PythonProject\异常标签检测\LEMoN+CLIPsim\weights")
for n in ("clip-vit-base-patch16", "clip-vit-base-patch32", "clip-vit-large-patch14"):
    c = json.loads((W / n / "config.json").read_text(encoding="utf-8"))
    v = c["vision_config"]
    img = v.get("image_size", 224)
    patches = (img // v["patch_size"]) ** 2
    print(f"{n:26s} layers={v['num_hidden_layers']:2d} width={v['hidden_size']:4d} "
          f"heads={v['num_attention_heads']:2d} patch={v['patch_size']:2d} "
          f"size={img} tokens={patches}")
