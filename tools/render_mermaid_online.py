"""
Render Mermaid (.mmd) diagrams to SVG without Node, using the public Kroki API.

Usage:
  python tools/render_mermaid_online.py               # render all *.mmd in diagrams/
  python tools/render_mermaid_online.py <file1.mmd> <file2.mmd> ...

Output SVGs are written alongside inputs (same name, .svg extension).
"""

from __future__ import annotations

import sys
from pathlib import Path
import requests


KROKI_ENDPOINT = "https://kroki.io/mermaid/svg"


def render_mmd_to_svg(mmd_path: Path, out_svg: Path) -> None:
    source = mmd_path.read_text(encoding="utf-8")
    # Kroki accepts raw diagram text as the request body for /mermaid/svg
    resp = requests.post(KROKI_ENDPOINT, data=source.encode("utf-8"), headers={"Content-Type": "text/plain; charset=utf-8"}, timeout=30)
    resp.raise_for_status()
    out_svg.write_bytes(resp.content)
    print(f"[ok] {mmd_path} -> {out_svg}")


def main(argv: list[str]) -> int:
    project_root = Path(__file__).resolve().parents[1]
    diagrams_dir = project_root / "diagrams"

    if len(argv) > 1:
        inputs = [Path(a) for a in argv[1:]]
    else:
        inputs = sorted(diagrams_dir.glob("*.mmd"))

    if not inputs:
        print("No .mmd files found. Specify paths or place files under diagrams/.")
        return 1

    for mmd in inputs:
        if not mmd.exists():
            print(f"[skip] missing: {mmd}")
            continue
        out_svg = mmd.with_suffix(".svg")
        try:
            render_mmd_to_svg(mmd, out_svg)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {mmd}: {exc}")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


