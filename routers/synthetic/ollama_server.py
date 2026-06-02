"""Start Ollama serve in a background subprocess (non-blocking notebook kernel)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

_process: subprocess.Popen | None = None
DEFAULT_HOST = "http://127.0.0.1:11434"


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def _health_ok(timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(f"{ollama_base_url()}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ensure_ollama_server(*, wait_seconds: float = 60.0, log_path: Path | None = None) -> None:
    global _process
    if _health_ok():
        logger.info("Ollama already reachable at {}", ollama_base_url())
        return

    log_path = log_path or Path("cache/ollama_serve.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    _process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        start_new_session=sys.platform != "win32",
    )
    logger.info("Started ollama serve pid={} log={}", _process.pid, log_path)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _health_ok():
            logger.info("Ollama ready")
            return
        time.sleep(0.5)
    raise RuntimeError(f"Ollama not ready after {wait_seconds}s; check {log_path}")


def stop_ollama_server() -> None:
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
        logger.info("Terminated ollama serve pid={}", _process.pid)
    _process = None
