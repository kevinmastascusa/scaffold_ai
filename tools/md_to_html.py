import sys
from pathlib import Path
import markdown


def convert_markdown_to_html(md_path: Path, out_html: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base_href = project_root.as_uri() + "/"

    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "attr_list",
            "md_in_html",
        ],
        output_format="html5",
    )

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }
    pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    img { max-width: 100%; height: auto; }
    h1, h2, h3 { page-break-after: avoid; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px; }
    .page-break { page-break-before: always; }
    """

    html = f"""
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <base href=\"{base_href}\" />
    <style>{css}</style>
    <title>{md_path.name}</title>
  </head>
  <body>
    {html_body}
  </body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/md_to_html.py <input.md> <output.html>")
        sys.exit(2)
    convert_markdown_to_html(Path(sys.argv[1]), Path(sys.argv[2]))


