#!/usr/bin/env python3
"""
Convert HTML to PDF with robust rendering:
- Replace Mermaid code blocks with embedded SVGs (via Kroki)
- Inline local <img> resources as data URIs
- Inject wide Unicode font fallbacks to avoid black boxes
- Export final HTML to PDF using Playwright (Chromium)

Usage:
  python tools/html_to_pdf.py <input.html> \
    --out <output.pdf> \
    --html-out <processed.html>

Notes:
- Requires: requests, playwright
  (and run: playwright install chromium)
- Mermaid detection supports
  <pre><code class="language-mermaid">…</code></pre>
  and simple <code>mermaid ...</code> blocks across lines.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import importlib
requests: Any = importlib.import_module("requests")  # type: ignore[assignment]


KROKI_ENDPOINT = "https://kroki.io/mermaid/svg"


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def resolve_base_dir_from_html(html: str, html_path: Path) -> Path:
    # Find <base href="..."> if present, else use file's directory
    m = re.search(r"<base[^>]*href=\"([^\"]+)\"", html, flags=re.IGNORECASE)
    if m:
        href = m.group(1)
        # Support file:/// URLs
        if href.startswith("file:///"):
            # Windows drive-style path in URL
            p = Path(href.replace("file:///", ""))
            return p
        # Absolute path
        if os.path.isabs(href):
            return Path(href)
        # Relative to HTML file
        return (html_path.parent / href).resolve()
    return html_path.parent


def encode_file_as_data_uri(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        # Default to binary
        mime = "application/octet-stream"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def replace_img_src_with_data_uris(html: str, base_dir: Path) -> str:
    # Replace <img src="..."> where src is local file
    def repl(match: re.Match) -> str:
        before, src, after = match.group(1), match.group(2), match.group(3)
        # Ignore data URIs and absolute http(s)
        if src.startswith("data:") or re.match(r"^[a-z]+://", src):
            return match.group(0)
        # Resolve against base_dir
        local_path = (base_dir / src).resolve()
        data_uri = encode_file_as_data_uri(local_path)
        if data_uri:
            return f"{before}{data_uri}{after}"
        return match.group(0)

    pattern = re.compile(
        r"(<img[^>]*?src=\")(.*?)(\"[^>]*>)",
        flags=re.IGNORECASE,
    )
    return pattern.sub(repl, html)


def extract_mermaid_blocks(html: str) -> list[Tuple[Tuple[int, int], str]]:
    """Find Mermaid code blocks and return list of ((start, end), code_text).

    Supports:
    - <pre><code class="language-mermaid"> ... </code></pre>
    - <code>mermaid\n...\n</code> (simplified fallback)
    """
    blocks: list[Tuple[Tuple[int, int], str]] = []

    # Pattern 1: <pre><code class="language-mermaid"> ... </code></pre>
    pattern1 = re.compile(
        r"<pre><code[^>]*class=\"[^\"]*language-mermaid[^\"]*\"[^>]*>"
        r"([\s\S]*?)</code></pre>",
        flags=re.IGNORECASE,
    )
    for m in pattern1.finditer(html):
        start, end = m.span()
        code_html = m.group(1)
        # Unescape basic entities
        code = (
            code_html.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
        )
        blocks.append(((start, end), code.strip()))

    # Pattern 2: <code>mermaid ...</code> (when fenced incorrectly in HTML)
    pattern2 = re.compile(
        r"<code>\s*mermaid\s*([\s\S]*?)</code>",
        flags=re.IGNORECASE,
    )
    for m in pattern2.finditer(html):
        start, end = m.span()
        code = m.group(1).strip()
        if code:
            blocks.append(((start, end), code))

    return blocks


def render_mermaid_to_svg(mermaid_text: str, timeout: float = 30.0) -> str:
    resp = requests.post(
        KROKI_ENDPOINT,
        data=mermaid_text.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.text


def replace_mermaid_with_img(html: str) -> str:
    blocks = extract_mermaid_blocks(html)
    if not blocks:
        return html

    # Replace from the end to keep indices valid
    result = html
    for (start, end), code in sorted(
        blocks, key=lambda x: x[0][0], reverse=True
    ):
        try:
            svg_text = render_mermaid_to_svg(code)
            data_uri = (
                "data:image/svg+xml;base64,"
                + base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
            )
            img_html = f'<img alt="Mermaid diagram" src="{data_uri}" />'
            result = result[:start] + img_html + result[end:]
        except requests.exceptions.RequestException as exc:  # noqa: BLE001
            # Leave original block on failure
            print(f"[warn] Mermaid render failed: {exc}")
    return result


def inject_font_fallbacks(html: str) -> str:
    head_close_idx = html.lower().find("</head>")
    if head_close_idx == -1:
        return html

    font_css = (
        "\n"
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
        "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\""
        " crossorigin>\n"
        "<link href=\"https://fonts.googleapis.com/css2?"
        "family=Noto+Sans:wght@400;600&"
        "family=Noto+Sans+Symbols+2&display=swap\""
        " rel=\"stylesheet\">\n"
        "<style>\n"
        "  body, p, li, td, th { font-family: 'Noto Sans', 'Segoe UI',\n"
        "    Roboto, Arial, sans-serif; }\n"
        "  code, pre { font-family: 'Cascadia Mono', 'Fira Code',\n"
        "    Consolas, 'Courier New', monospace; }\n"
        "  svg, .mermaid { font-family: 'Noto Sans', Arial, sans-serif\n"
        "    !important; }\n"
        "  * { -webkit-font-smoothing: antialiased;\n"
        "      -moz-osx-font-smoothing: grayscale; }\n"
        "</style>\n"
    )
    return html[:head_close_idx] + font_css + html[head_close_idx:]


def export_pdf_with_playwright(html_path: Path, pdf_path: Path) -> None:
    playwright_sync_api: Any = importlib.import_module("playwright.sync_api")
    sync_playwright = playwright_sync_api.sync_playwright  # type: ignore

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(
            f"file://{html_path.resolve()}",
            wait_until="load",
        )
        # Give time for external fonts to load
        page.wait_for_timeout(1000)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={
                "top": "20mm",
                "bottom": "20mm",
                "left": "15mm",
                "right": "15mm",
            },
        )
        browser.close()


def process_html(input_path: Path, processed_html_out: Optional[Path]) -> Path:
    html = read_text_file(input_path)
    base_dir = resolve_base_dir_from_html(html, input_path)

    html = replace_mermaid_with_img(html)
    html = replace_img_src_with_data_uris(html, base_dir)
    html = inject_font_fallbacks(html)

    # Write processed HTML
    if processed_html_out is None:
        processed_html_out = input_path.with_suffix(".inlined.html")
    write_text_file(processed_html_out, html)
    return processed_html_out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Inline diagrams and export HTML to PDF."
    )
    parser.add_argument(
        "input_html", type=Path, help="Path to input HTML file"
    )
    parser.add_argument(
        "--out", dest="output_pdf", type=Path, default=None,
        help="Output PDF path",
    )
    parser.add_argument(
        "--html-out", dest="output_html", type=Path, default=None,
        help="Output processed HTML path",
    )
    args = parser.parse_args(argv[1:])

    html_in: Path = args.input_html
    if not html_in.exists():
        print(f"Input HTML not found: {html_in}")
        return 2

    processed_html = process_html(html_in, args.output_html)
    pdf_out = args.output_pdf or html_in.with_suffix(".pdf")

    try:
        export_pdf_with_playwright(processed_html, pdf_out)
        print(f"[ok] PDF written: {pdf_out}")
    except Exception as exc:  # noqa: BLE001, PERF203
        print(f"[error] PDF export failed: {exc}")
        print("Hint: pip install playwright && playwright install chromium")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
