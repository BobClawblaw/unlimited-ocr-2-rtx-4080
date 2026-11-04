"""Decode speed deep in a long prompt: the release's paper (ocr/pages/long-N.png, rendered from Unlimited-OCR-2.pdf)
`copies` times over as one request, the reply capped at `max_tokens`:
python ocr/longbench.py http://127.0.0.1:8896 [copies] [max_tokens]"""
import base64, hashlib, json, sys, time, urllib.request
from pathlib import Path
P = Path(__file__).resolve().parent / "pages"
url = sys.argv[1]
copies = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
pages = [f"long-{n}" for n in range(1, 16)] * copies
content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode((P / f"{p}.png").read_bytes()).decode()}} for p in pages]
body = {"model": "Unlimited-OCR-2", "messages": [{"role": "user", "content": content}], "stream": True, "max_tokens": cap}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); first = None; text = ""; usage = None
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        for c in d.get("choices", []):
            piece = c.get("delta", {}).get("content") or ""
            if piece and first is None: first = time.time()
            text += piece
end = time.time()
n = usage["completion_tokens"]
print(f"pages={len(pages)} prompt={usage['prompt_tokens']} tokens={n} ttft={first - t0:.2f}s decode={(n - 1) / (end - first):.1f} tok/s sha={hashlib.sha256(text.encode()).hexdigest()[:12]}")
