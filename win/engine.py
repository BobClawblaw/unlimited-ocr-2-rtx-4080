"""The engine: a worktree of the TensorFold checkout at the pinned commit, built under tf_cache (never in the
checkout) with nvcc for this card's SM, from the MSVC environment (vcvars64.bat). The family needs no libtfgrammar:
it decodes greedily under its page guard, with no tools or schemas."""

import subprocess
from pathlib import Path

from .config import Settings

# where tensorfold_sha lives until the family reaches TensorFold's main
TENSORFOLD_REPO = "https://github.com/BobClawblaw/TensorFold"
TENSORFOLD_BRANCH = "unlimited-ocr-cuda"


def _run(cmd: list[str] | str, cwd: Path | None = None, shell: bool = False) -> None:
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, shell=shell)


def build(s: Settings, rebuild: bool = False) -> Path:
    if s.binary:
        return s.run_binary
    if not s.tensorfold_sha:
        raise SystemExit("no tensorfold_sha pinned (local.toml)")
    src = s.path(s.tensorfold_src)
    have = subprocess.run(["git", "-C", str(src), "cat-file", "-e", f"{s.tensorfold_sha}^{{commit}}"], capture_output=True)
    if have.returncode != 0:  # a clone older than the pin
        _run(["git", "-C", str(src), "fetch", TENSORFOLD_REPO, TENSORFOLD_BRANCH])
    if not s.worktree.exists():
        s.worktree.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "-C", str(src), "worktree", "add", "--detach", str(s.worktree), s.tensorfold_sha])
    if rebuild or not s.built_binary.exists():
        zig = s.path(s.zig)
        _run(f'call "{s.vcvars}" >nul && "{zig}" build native install -Dsm={s.sm} -Dstrip=true -Dnvcc="{s.nvcc}"', cwd=s.worktree, shell=True)
    return s.built_binary
