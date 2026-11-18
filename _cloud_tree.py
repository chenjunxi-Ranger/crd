import sys

import requests

repo = sys.argv[1]
r = requests.get(f"https://hf-mirror.com/api/datasets/{repo}/tree/main",
                 params={"recursive": "true"}, timeout=60)
for x in r.json():
    if x.get("type") == "file":
        print(x["size"], x["path"])
