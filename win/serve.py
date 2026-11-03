"""The server: started in the background (pid file, log), stopped by its own pid, watched."""

import base64
import json
import subprocess
import time
import urllib.request

from .config import CACHE_BYTES_A_TOKEN, Settings


def weights_gib(s: Settings) -> float:
    """The checkpoint's share of the card once loaded: its safetensors shards."""
    return sum(f.stat().st_size for f in s.model.glob("*.safetensors")) / 2**30


def need_gib(s: Settings) -> float:
    """What a load holds on the card: the weights, the caches for `context` tokens, and about 1.5 GiB of prompt and
    tower scratch, cuBLASLt's workspace and the CUDA context."""
    return weights_gib(s) + s.context * CACHE_BYTES_A_TOKEN / 2**30 + 1.5


def health(s: Settings, timeout: float = 3) -> dict | None:
    try:
        with urllib.request.urlopen(s.url.removesuffix("/v1") + "/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def _alive(p: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {p}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    return f'"{p}"' in out


def pid(s: Settings) -> int | None:
    """The running server's pid; a pid file whose process is gone is removed."""
    try:
        p = int(s.pid_file.read_text())
        if _alive(p):
            return p
    except Exception:
        return None
    s.pid_file.unlink(missing_ok=True)
    return None


def free_gib() -> float:
    """The card's free memory, as nvidia-smi reports it."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0]) / 1024


def memory_check(s: Settings) -> None:
    need = need_gib(s)
    have = free_gib()
    if have < need:
        raise SystemExit(f"{have:.1f} GiB free on the card, the server needs {need:.1f}: stop other GPU work or lower context")


def wait_ready(s: Settings, proc: subprocess.Popen, limit: float = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < limit:
        if health(s):
            return
        if proc.poll() is not None:
            s.pid_file.unlink(missing_ok=True)
            raise SystemExit(f"the server exited while loading (status {proc.returncode}); see {s.log_file}")
        time.sleep(2)
    raise SystemExit(f"not ready after {limit:.0f} s; see {s.log_file}")


def up(s: Settings) -> None:
    if pid(s) or health(s):
        raise SystemExit(f"already serving at {s.url}")
    memory_check(s)
    s.cache.mkdir(parents=True, exist_ok=True)
    log = open(s.log_file, "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    p = subprocess.Popen(s.serve_args(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    s.pid_file.write_text(str(p.pid))
    wait_ready(s, p)
    print(f"serving {s.served_name} at {s.url} (pid {p.pid})")


def down(s: Settings) -> None:
    """Stops the server `up` started (its pid, nothing else)."""
    p = pid(s)
    if not p:
        print("not running")
        return
    subprocess.run(["taskkill", "/PID", str(p), "/T", "/F"], capture_output=True)
    for _ in range(60):
        if not _alive(p):
            break
        time.sleep(0.5)
    s.pid_file.unlink(missing_ok=True)
    print("stopped")


def status(s: Settings) -> str:
    h = health(s)
    return f"pid: {pid(s) or 'not running'}\nurl: {s.url}\nhealth: {json.dumps(h) if h else 'down'}"


def ocr(s: Settings, pages: list[str], instruction: str = "", max_tokens: int = 0) -> str:
    """Page images (PNG, JPEG or WebP files, in order) to the running server; the reply's text."""
    content = []
    for p in pages:
        data = open(p, "rb").read()
        kind = "png" if data[:4] == b"\x89PNG" else "jpeg" if data[:2] == b"\xff\xd8" else "webp"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{kind};base64," + base64.b64encode(data).decode()}})
    if instruction:
        content.append({"type": "text", "text": instruction})
    body = {"model": s.served_name, "messages": [{"role": "user", "content": content}]}
    if max_tokens:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(s.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["choices"][0]["message"]["content"]
