"""Download a HF CLIP checkpoint via direct file URLs (hub API is blocked here)."""
from __future__ import annotations

import argparse
import socket
import sys
import urllib.request
from pathlib import Path

FILES = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "special_tokens_map.json", "pytorch_model.bin"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default=",".join(FILES))
    args = ap.parse_args()
    socket.setdefaulttimeout(600)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for f in args.files.split(","):
        dst = out / f
        if dst.is_file() and dst.stat().st_size > 0:
            print("skip", f); continue
        url = f"https://huggingface.co/{args.repo}/resolve/main/{f}"
        tmp = dst.with_suffix(dst.suffix + ".part")
        have = tmp.stat().st_size if tmp.is_file() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req) as r, open(tmp, "ab" if have else "wb") as w:
            while True:
                try:
                    chunk = r.read(1 << 20)
                except TimeoutError:
                    print("retry", f); break
                if not chunk:
                    break
                w.write(chunk)
        if dst.exists():
            continue
        tmp.replace(dst)
        print("ok", f, round(dst.stat().st_size / 1e6, 1), "MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
