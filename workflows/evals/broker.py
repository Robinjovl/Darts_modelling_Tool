"""Parent-side simulation broker (design section 10).

The evaluated agent never runs simulations itself: it submits a study spec over a Unix socket;
the broker validates it, runs the workflow in the parent context into a parent-owned results
directory, and replies with the summary. Requests and replies are single JSON lines.
"""

from __future__ import annotations

import json
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

from workflows.spec import StudySpec, validate

MAX_REQUEST_BYTES = 4 * 1024 * 1024


class BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(MAX_REQUEST_BYTES)
        try:
            request = json.loads(raw)
            reply = self.server.dispatch(request)
        except Exception as exc:  # noqa: BLE001 - always answer the client
            reply = {"status": "error", "error": repr(exc)}
        self.wfile.write((json.dumps(reply, sort_keys=True) + "\n").encode("utf-8"))


class Broker(socketserver.ThreadingUnixStreamServer):
    """Serves ``{"spec": {...}, "study": "<name>", "command": "run"|"estimate"}`` requests."""

    daemon_threads = True

    def __init__(self, socket_path: str | Path, results_dir: str | Path, runner=None):
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or self.run_workflow
        self.log: list = []
        super().__init__(str(socket_path), BrokerHandler)

    def dispatch(self, request: dict) -> dict:
        command = request.get("command", "run")
        spec = request["spec"]
        validate(spec)
        name = Path(str(request.get("study", spec["name"]))).name
        if not name or name.startswith("."):
            raise ValueError("invalid study name")
        study_dir = self.results_dir / name
        spec_path = self.results_dir / f"{name}.spec.json"
        spec_path.write_text(
            json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8"
        )
        record = {
            "command": command,
            "study": name,
            "spec_hash": StudySpec.from_dict(spec).workflow_run_hash(),
        }
        self.log.append(record)
        result = self.runner(command, spec_path, study_dir)
        return {"status": "ok", "study_dir": str(study_dir), **result}

    @staticmethod
    def run_workflow(command: str, spec_path: Path, study_dir: Path) -> dict:
        argv = [sys.executable, "-m", "workflows", command, "--spec", str(spec_path)]
        if command != "estimate":
            argv += ["--study", str(study_dir)]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0:
            return {
                "result": None,
                "stderr": completed.stderr[-4000:],
                "returncode": completed.returncode,
            }
        return {"result": json.loads(completed.stdout.strip().splitlines()[-1])}

    def serve_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


def submit(
    socket_path: str | Path,
    spec: dict,
    study: str,
    command: str = "run",
    timeout_s: float = 3600.0,
) -> dict:
    """Client side: send one request and return the reply (used by the sandboxed CLI wrapper)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(str(socket_path))
        sock.sendall(
            (
                json.dumps({"spec": spec, "study": study, "command": command}) + "\n"
            ).encode("utf-8")
        )
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
    return json.loads(b"".join(chunks))


def main(argv: list | None = None) -> int:
    """``python -m workflows.evals.broker --spec study.json --study NAME [--command run]``
    submits a study to the broker named by ``WORKFLOWS_BROKER`` and prints the reply."""
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="python -m workflows.evals.broker")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--command", default="run", choices=("run", "estimate", "truth")
    )
    parser.add_argument("--socket", default=os.environ.get("WORKFLOWS_BROKER"))
    args = parser.parse_args(argv)
    if not args.socket:
        parser.error("no broker socket: pass --socket or set WORKFLOWS_BROKER")
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    reply = submit(args.socket, spec, args.study, command=args.command)
    print(json.dumps(reply, sort_keys=True))
    return 0 if reply.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
