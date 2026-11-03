"""Pages to the running server, each reply against ocr/refs/<case>.txt (a quick check without the harness):
python -X utf8 ocr/check.py http://127.0.0.1:8896 CASE=page[,page...] ...; replies land in ocr/out/."""
import base64, difflib, json, sys, time, urllib.request
from pathlib import Path
OCR = Path(__file__).resolve().parent
url = sys.argv[1]
for case in sys.argv[2:]:
    name, pages = case.split("=")
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode((OCR / "pages" / f"{p}.png").read_bytes()).decode()}} for p in pages.split(",")]
    body = {"model": "Unlimited-OCR-2", "messages": [{"role": "user", "content": content}], "stream": True}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = None
    with urllib.request.urlopen(req) as r:
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
    ref = (OCR / "refs" / f"{name}.txt").read_text(encoding="utf-8")
    n = (usage or {}).get("completion_tokens", 0)
    sim = difflib.SequenceMatcher(None, text, ref).ratio()
    print(f"{name}: identical={text == ref} sim={sim:.4f} ttft={first - t0:.2f}s decode={n / max(end - first, 1e-9):.1f} tok/s tokens={n} prompt={(usage or {}).get('prompt_tokens')}")
    (OCR / "out").mkdir(exist_ok=True); (OCR / "out" / f"{name}.txt").write_text(text, encoding="utf-8")
