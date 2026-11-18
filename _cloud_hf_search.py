import sys

import requests

q = sys.argv[1]
r = requests.get("https://hf-mirror.com/api/datasets",
                 params={"search": q, "limit": 50}, timeout=30)
for d in r.json():
    print(d["id"])
