"""python -m win build|up|down|status|logs|command|ocr|validate"""

import argparse
import subprocess
import sys

from . import config, engine, serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m win", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the pinned engine, under tf_cache")
    b.add_argument("--rebuild", action="store_true")
    sub.add_parser("up", help="start the server in the background")
    sub.add_parser("down", help="stop the server `up` started")
    sub.add_parser("status", help="pid, url and the server's health")
    logs = sub.add_parser("logs", help="the server log's last lines")
    logs.add_argument("-n", type=int, default=40)
    sub.add_parser("command", help="print the serve command")
    ch = sub.add_parser("ocr", help="page images (in order) to the running server; prints the transcription")
    ch.add_argument("pages", nargs="+")
    ch.add_argument("--instruction", default="", help="the text after the pages (default: Multi page parsing.)")
    ch.add_argument("--max-tokens", type=int, default=0)
    sub.add_parser("validate", help="check the settings, the checkpoint and the build without starting anything")
    a = p.parse_args(argv)
    s = config.load()
    if a.cmd == "build":
        engine.build(s, rebuild=a.rebuild)
        print(f"engine {s.run_binary}")
    elif a.cmd == "up":
        problems = config.validate(s)
        if problems:
            raise SystemExit("\n".join(problems))
        serve.up(s)
    elif a.cmd == "down":
        serve.down(s)
    elif a.cmd == "status":
        print(serve.status(s))
    elif a.cmd == "logs":
        print("\n".join(s.log_file.read_text(errors="replace").splitlines()[-a.n:]) if s.log_file.exists() else "no log")
    elif a.cmd == "command":
        print(subprocess.list2cmdline(s.serve_args()))
    elif a.cmd == "ocr":
        sys.stdout.reconfigure(encoding="utf-8")
        print(serve.ocr(s, a.pages, a.instruction, a.max_tokens))
    elif a.cmd == "validate":
        problems = config.validate(s)
        print("\n".join(problems) if problems else f"ok: {s.served_name} from {s.model} with {s.run_binary} at {s.url}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
