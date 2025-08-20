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
    os.environ.setdefault("SC_STRICT_ANSWERS", "true")
    os.environ.setdefault("SC_AUTO_CLEAR_MEMORY", "false")
    # Context/memory management defaults
    os.environ.setdefault("SC_AUTO_CLEAR_ON_TOPIC_SHIFT", "true")
    os.environ.setdefault("SC_TOPIC_SHIFT_THRESHOLD", "0.2")
    os.environ.setdefault("SC_MEMORY_TTL_MIN", "60")

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
        "--skip-checks",
        action="store_true",
        help="Skip dependency and data file checks",
    )
    args = parser.parse_args()

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
