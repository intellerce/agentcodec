"""
Sandboxed Python execution for grading LLM-generated code.

Two backends, selected via ``$AGENTCODEC_CODE_SANDBOX``:

  * ``"subprocess"`` (default) — spawns a fresh ``python -I`` child via
    :mod:`subprocess`, capping memory / CPU time / wall-clock through
    ``resource.setrlimit`` in a ``preexec_fn``. No external dependencies.
    Linux/macOS only for the resource limits; Windows degrades to a
    wall-clock-only timeout. Fast (~50 ms per run) but the child shares
    the host filesystem and network.

  * ``"docker"`` (opt-in) — spawns a one-shot container per call with
    ``--network=none``, read-only filesystem (writable ``/tmp`` tmpfs),
    memory + CPU + PID caps, and ``--cap-drop=ALL``. Hard isolation;
    works on Windows. Requires Docker daemon; ~300–800 ms startup
    overhead per call. Override the image via
    ``$AGENTCODEC_CODE_DOCKER_IMAGE`` (default ``python:3.11-slim``).

Common env vars (apply to both backends):

  * ``AGENTCODEC_CODE_TIMEOUT_S``  — wall-clock kill (default ``10``).
  * ``AGENTCODEC_CODE_MEMORY_MB``  — RSS cap in MiB (default ``256``).

The selection is one env var so a project's `.env` can flip the default
without touching code. Adversarial LLM output should always use
``docker`` (or stronger, e.g. firecracker). The subprocess backend is
appropriate when the model output is trusted-source (your own pipeline,
your own LLM, your own benchmarks).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

# --- Knobs (env-overridable) ----------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DEFAULT_TIMEOUT_S = float(_env("AGENTCODEC_CODE_TIMEOUT_S", "10"))
DEFAULT_MEMORY_MB = int(_env("AGENTCODEC_CODE_MEMORY_MB", "256"))

# Docker-specific.
DEFAULT_DOCKER_IMAGE = _env("AGENTCODEC_CODE_DOCKER_IMAGE", "python:3.11-slim")
DEFAULT_DOCKER_CPUS = _env("AGENTCODEC_CODE_DOCKER_CPUS", "1")

# Selector. Anything other than "docker" (case-insensitive) means "subprocess".
def _selected_backend() -> str:
    raw = _env("AGENTCODEC_CODE_SANDBOX", "subprocess").strip().lower()
    return "docker" if raw == "docker" else "subprocess"


# --- Result type ----------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of one sandboxed program execution."""
    exit_code: int            # 0 on success; -1 on harness failure
    stdout: str
    stderr: str
    timed_out: bool           # True when the wall-clock killer fired
    elapsed_s: float          # host-measured wall clock
    backend: str = ""         # "subprocess" or "docker"
    error: str | None = None  # populated only on harness failure
                              # (e.g. docker missing) — not program errors

    @property
    def ok(self) -> bool:
        """Program ran and exited cleanly within the time budget."""
        return self.exit_code == 0 and not self.timed_out and self.error is None


# --- subprocess backend ---------------------------------------------------

def _build_preexec(memory_mb: int, timeout_s: float):
    """Build the ``preexec_fn`` that sets resource limits in the child.

    Runs after fork() and before exec(), so it doesn't affect the host.
    Skipped on Windows where ``resource`` is unavailable.
    """
    if os.name != "posix":
        return None
    try:
        import resource  # POSIX only
    except ImportError:
        return None

    bytes_cap = memory_mb * 1024 * 1024
    cpu_cap = int(timeout_s) + 1  # +1s grace so SIGXCPU fires after wall timeout

    def _apply() -> None:
        # RLIMIT_AS caps the address space — best signal for "process used
        # too much memory" on Linux. macOS implements it but the semantics
        # are looser; we set it on a best-effort basis.
        for limit_name, value in (
            ("RLIMIT_AS", bytes_cap),
            ("RLIMIT_DATA", bytes_cap),
            ("RLIMIT_CPU", cpu_cap),
        ):
            limit = getattr(resource, limit_name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                # Some sandboxes refuse to raise hard limits; skip silently.
                pass

    return _apply


def run_in_subprocess(
    program: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> ExecutionResult:
    """Run a Python program in a fresh isolated child process.

    Uses ``python -I`` so the child ignores ``PYTHONPATH`` and the user
    site-packages directory — it sees only the stdlib that came with the
    interpreter the host is using.
    """
    start = time.perf_counter()
    cmd = [sys.executable, "-I", "-c", program]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_build_preexec(memory_mb, timeout_s),
        )
        return ExecutionResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            elapsed_s=time.perf_counter() - start,
            backend="subprocess",
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return ExecutionResult(
            exit_code=-1,
            stdout=out, stderr=err,
            timed_out=True,
            elapsed_s=timeout_s,
            backend="subprocess",
        )
    except Exception as e:
        return ExecutionResult(
            exit_code=-1, stdout="", stderr="",
            timed_out=False,
            elapsed_s=time.perf_counter() - start,
            backend="subprocess",
            error=f"subprocess sandbox failed: {e!r}",
        )


# --- Docker backend -------------------------------------------------------

def is_docker_available() -> bool:
    """Return True iff the ``docker`` CLI is on ``$PATH``."""
    return shutil.which("docker") is not None


def run_in_docker(
    program: str,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    cpus: str = DEFAULT_DOCKER_CPUS,
) -> ExecutionResult:
    """Run a Python program in a one-shot isolated Docker container."""
    if not is_docker_available():
        return ExecutionResult(
            exit_code=-1, stdout="", stderr="",
            timed_out=False, elapsed_s=0.0,
            backend="docker",
            error="docker CLI not found on $PATH",
        )

    cmd = [
        "docker", "run", "--rm", "-i",
        "--network=none",
        f"--memory={memory_mb}m",
        f"--cpus={cpus}",
        "--read-only",
        "--tmpfs=/tmp:rw,size=64m",
        "--cap-drop=ALL",
        "--pids-limit=64",
        image,
        "python", "-",
    ]

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            input=program,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ExecutionResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            elapsed_s=time.perf_counter() - start,
            backend="docker",
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return ExecutionResult(
            exit_code=-1,
            stdout=out, stderr=err,
            timed_out=True,
            elapsed_s=timeout_s,
            backend="docker",
        )
    except FileNotFoundError:
        return ExecutionResult(
            exit_code=-1, stdout="", stderr="",
            timed_out=False, elapsed_s=0.0,
            backend="docker",
            error="docker CLI not found on $PATH",
        )
    except Exception as e:
        return ExecutionResult(
            exit_code=-1, stdout="", stderr="",
            timed_out=False,
            elapsed_s=time.perf_counter() - start,
            backend="docker",
            error=f"docker run failed: {e!r}",
        )


# --- Public dispatcher ----------------------------------------------------

def run_sandboxed(
    program: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> ExecutionResult:
    """Run ``program`` in the configured sandbox backend.

    Backend selection: ``$AGENTCODEC_CODE_SANDBOX`` — ``"docker"`` to
    use the container backend, anything else (including unset) selects
    the subprocess backend. Defaults are documented at the top of this
    file and in the project README.
    """
    if _selected_backend() == "docker":
        return run_in_docker(program, timeout_s=timeout_s, memory_mb=memory_mb)
    return run_in_subprocess(program, timeout_s=timeout_s, memory_mb=memory_mb)


__all__ = [
    "DEFAULT_DOCKER_CPUS",
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_MEMORY_MB",
    "DEFAULT_TIMEOUT_S",
    "ExecutionResult",
    "is_docker_available",
    "run_in_docker",
    "run_in_subprocess",
    "run_sandboxed",
]
