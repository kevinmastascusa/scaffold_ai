"""
Launcher for the existing Enhanced Flask UI that loads environment
variables from local .env files, then delegates to the project's
startup script.

This file exists to avoid editing existing source files while enabling
EXE packaging and secret handling via .env.
"""

import argparse
import os
import sys
from pathlib import Path


def load_env_files() -> None:
    """Load environment variables from .env files if python-dotenv is present.

    Load order (later does not override earlier to keep local overrides):
    - .env
    - .env.local
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv not installed; skip silently
        return

    cwd = Path(__file__).parent.absolute()

    # Base project .env
    base_env = cwd / ".env"
    if base_env.exists():
        load_dotenv(base_env, override=False)

    # Local overrides (never committed)
    local_env = cwd / ".env.local"
    if local_env.exists():
        load_dotenv(local_env, override=False)


def add_bundle_paths() -> None:
    """Ensure import paths work when frozen into an EXE."""
    base_dir = Path(__file__).parent.absolute()

    # Project root
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    # Bundled subpackages (PyInstaller --add-data copied these)
    frontend_dir = base_dir / "frontend"
    core_dir = base_dir / "scaffold_core"
    for p in (frontend_dir, core_dir):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def main() -> int:
    load_env_files()
    add_bundle_paths()

    # Defaults for strict/correct answers and auto-clearing memory
    os.environ.setdefault("SC_STRICT_ANSWERS", "0")
    os.environ.setdefault("SC_AUTO_CLEAR_MEMORY", "false")
    # Context/memory management defaults
    os.environ.setdefault("SC_AUTO_CLEAR_ON_TOPIC_SHIFT", "true")
    os.environ.setdefault("SC_TOPIC_SHIFT_THRESHOLD", "0.2")
    os.environ.setdefault("SC_MEMORY_TTL_MIN", "60")
    # Disable quality mode by default (explicitly off)
    os.environ.setdefault("SC_QUALITY_MODE", "0")
    # Disable truncation detection and proofreading by default
    os.environ.setdefault("SC_ENABLE_TRUNCATION_DETECTION", "0")
    os.environ.setdefault("SC_ENABLE_PROOFREAD", "0")
    # Disable reasoning exposure by default
    os.environ.setdefault("SC_EXPOSE_REASONING", "0")

    # Runtime provider controls
    # If SC_INCLUDE_CUDA=1, explicitly override any SC_FORCE_CPU setting
    include_cuda = str(os.getenv("SC_INCLUDE_CUDA", "")).lower() in (
        "1", "true", "yes"
    )
    force_cpu = str(os.getenv("SC_FORCE_CPU", "")).lower() in (
        "1", "true", "yes"
    )
    if include_cuda:
        # Prefer GPU when explicitly requested
        os.environ["SC_FORCE_CPU"] = "0"
        os.environ["ORT_DISABLE_GPU"] = "0"
        # Remove CPU-forcing device mask if present
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("ORT_DISABLE_GPU", "1")
    else:
        # Default: allow GPU if available
        os.environ.setdefault("ORT_DISABLE_GPU", "0")

    # Allow forwarding host/port args to the UI
    parser = argparse.ArgumentParser(
        description="Run Scaffold AI Enhanced UI"
    )
    parser.add_argument(
        "--host", type=str, default=os.getenv("SC_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("SC_PORT", "5002"))
    )
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="If set, find a free port starting from the requested one",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency and data file checks",
    )
    args = parser.parse_args()

    # Optionally auto-pick an open port
    if args.auto_port:
        import socket

        def _find_free_port(start: int) -> int:
            port = start
            while port < start + 100:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind((args.host, port))
                        return port
                    except OSError:
                        port += 1
            return start

        args.port = _find_free_port(args.port)

    # Export args so the startup script can see them
    # (it uses argparse itself; exporting helps when run under frozen apps)
    os.environ.setdefault("SC_HOST", str(args.host))
    os.environ.setdefault("SC_PORT", str(args.port))

    # Delegate to the existing startup script
    try:
        from frontend.start_enhanced_ui import main as start_main
        # Reconstruct argv for the delegate so its argparse works as expected
        delegate_argv = [sys.argv[0]]
        if args.host:
            delegate_argv += ["--host", str(args.host)]
        if args.port:
            delegate_argv += ["--port", str(args.port)]
        if args.skip_checks:
            delegate_argv += ["--skip-checks"]

        # Temporarily patch sys.argv for the delegate
        old_argv = sys.argv
        sys.argv = delegate_argv
        try:
            return start_main() or 0
        finally:
            sys.argv = old_argv
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"Failed to start Enhanced UI: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
