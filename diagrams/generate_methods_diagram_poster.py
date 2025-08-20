#!/usr/bin/env python3
"""
Generate a compact, poster-friendly Methods diagram (no Mermaid).

Outputs a small vertical flow of labeled boxes with arrows,
suited for research posters.

Dependencies: Pillow (already in requirements.txt)
Usage:
  python diagrams/generate_methods_diagram_poster.py \
    --output diagrams/methods_diagram_poster.png --theme mono --scale 1.0
  python diagrams/generate_methods_diagram_poster.py \
    --out diagrams/methods_diagram_poster.png --theme mono --scale 1.0
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from typing import cast


def load_font(preferred_size: int) -> ImageFont.ImageFont:
    """Load a compact sans-serif font; return a Pillow ImageFont instance."""
    # Try common fonts; fall back to default PIL bitmap font
    candidates = [
        # DejaVu shipped with many Pillow installs
        ("DejaVuSans.ttf", preferred_size),
        ("arial.ttf", preferred_size),
        ("segoeui.ttf", preferred_size),
        ("calibri.ttf", preferred_size),
        ("tahoma.ttf", preferred_size),
    ]
    for name, size in candidates:
        try:
            return cast(ImageFont.ImageFont, ImageFont.truetype(name, size))
        except Exception:
            continue
    # PIL's default font returns ImageFont.ImageFont
    return cast(ImageFont.ImageFont, ImageFont.load_default())


def sanitize_text(text: str) -> str:
    """Replace common Unicode glyphs with ASCII-safe equivalents.

    Improves font compatibility across systems.
    """
    replacements = {
        "\u2192": "->",  # right arrow
        "\u2011": "-",   # non-breaking hyphen
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2212": "-",   # minus sign
        "\u00B7": "-",   # middle dot
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201C": '"',    # left double quote
        "\u201D": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00D7": "x",   # multiplication sign
        "\u00A0": " ",   # no-break space
        "\u2009": " ",   # thin space
        "\u202F": " ",   # narrow no-break space
        "\u200B": "",    # zero-width space
        "\u200C": "",
        "\u200D": "",
        "\uFEFF": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    # Support multi-line labels with "\n"
    lines = text.split("\n")
    line_heights = []
    max_line_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, line_width)
        line_heights.append(line_height)

    total_text_height = sum(line_heights) + (len(lines) - 1) * 4
    x = left + (right - left - max_line_width) // 2
    y = top + (bottom - top - total_text_height) // 2

    for i, line in enumerate(lines):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_heights[i] + 4


def draw_arrow_down(
    draw: ImageDraw.ImageDraw,
    start_x: int,
    start_y: int,
    end_y: int,
    color: str,
    shaft_width: int = 3,
    head_size: int = 10,
) -> None:
    # Shaft
    draw.line(
        [(start_x, start_y), (start_x, end_y - head_size)],
        fill=color,
        width=shaft_width,
    )
    # Arrowhead (triangle)
    draw.polygon([
        (start_x, end_y),
        (start_x - head_size, end_y - head_size),
        (start_x + head_size, end_y - head_size),
    ], fill=color)


def draw_centered_title_subtitle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    fill: str,
    line_spacing: int = 4,
    padding: int = 12,
    max_subtitle_lines: int = 2,
    min_subtitle_px: int = 12,
) -> None:
    left, top, right, bottom = box
    # Measure title
    t_bbox = draw.textbbox((0, 0), title, font=title_font)
    t_h = t_bbox[3] - t_bbox[1]
    # Prepare subtitle wrapping within inner width (center within padding)
    inner_left = left + padding
    inner_right = right - padding
    inner_width = max(10, inner_right - inner_left)
    center_x = inner_left + inner_width // 2

    def text_width(txt: str, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), txt, font=font)
        return bbox[2] - bbox[0]

    def wrap_lines(
        txt: str,
        font: ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        if not txt:
            return []
        words = txt.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if text_width(candidate, font) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    # Adjust subtitle font size to fit within max_subtitle_lines
    s_font = subtitle_font
    s_lines: list[str] = wrap_lines(subtitle, s_font, inner_width)
    # Reduce font size progressively if the subtitle needs more lines
    current_size = 16
    # Note: ImageFont.ImageFont may not expose .size consistently.
    # Use a conservative starting size and shrink as needed.
    while (
        len(s_lines) > max_subtitle_lines
        and current_size > min_subtitle_px
    ):
        current_size -= 1
        s_font = load_font(current_size)
        s_lines = wrap_lines(subtitle, s_font, inner_width)

    # If still too many lines, truncate last line with ellipsis
    if len(s_lines) > max_subtitle_lines:
        s_lines = s_lines[:max_subtitle_lines]
        last = s_lines[-1]
        while last and text_width(last + "…", s_font) > inner_width:
            last = last[:-1]
        s_lines[-1] = (last + "…") if last else ""

    # Compute subtitle block size
    s_line_heights = [
        (
            draw.textbbox((0, 0), ln, font=s_font)[3]
            - draw.textbbox((0, 0), ln, font=s_font)[1]
        )
        for ln in (s_lines if s_lines else [""])
    ]
    s_h = (
        sum(s_line_heights) + (len(s_lines) - 1) * line_spacing
        if s_lines
        else 0
    )
    block_h = t_h + (line_spacing if s_lines else 0) + s_h
    # Center block vertically
    block_top = top + (bottom - top - block_h) // 2

    # Draw title at true center horizontally using anchor='mm'
    title_center_y = block_top + t_h // 2
    draw.text(
        (center_x, title_center_y),
        title,
        fill=fill,
        font=title_font,
        anchor="mm",
    )
    # Draw subtitle lines centered by anchor
    if s_lines:
        y = block_top + t_h + line_spacing
        for i, ln in enumerate(s_lines):
            line_center_y = y + s_line_heights[i] // 2
            draw.text(
                (center_x, line_center_y),
                ln,
                fill=fill,
                font=s_font,
                anchor="mm",
            )
            y += s_line_heights[i] + line_spacing


def generate_methods_diagram(
    output_path: Path,
    scale: float = 1.0,
    theme: str = "mono",
    ascii_only: bool = True,
) -> None:
    # Content (large title + smaller subtitle in each box)
    steps = [
        {
            "title": "DATA SOURCE",
            "subtitle": (
                "200 peer-reviewed papers on sustainability and climate "
                "pedagogy in engineering"
            ),
        },
        {
            "title": "PREPROCESS",
            "subtitle": (
                "Normalize text (UTF-8 NFC); remove control characters, "
                "headers/footers, boilerplate"
            ),
        },
        {
            "title": "CHUNK & CLEAN",
            "subtitle": (
                "Split into ~1,000-token chunks with 200 overlap; keep "
                "filename/page/source metadata"
            ),
        },
        {
            "title": (
                "EMBED -> FAISS INDEX" if ascii_only else "EMBED → FAISS INDEX"
            ),
            "subtitle": (
                "MiniLM embeddings (384-dim); FAISS IndexFlatL2 for "
                "similarity search"
            ),
        },
        {
            "title": "USER QUERY",
            "subtitle": (
                "Normalize; extract keywords; route to semantic+keyword search"
            ),
        },
        {
            "title": (
                "RETRIEVE TOP-K" if ascii_only else "RETRIEVE TOP‑K"
            ),
            "subtitle": (
                "FAISS kNN (K=50) collects initial candidate chunks"
            ),
        },
        {
            "title": ("RE-RANK" if ascii_only else "RE‑RANK"),
            "subtitle": (
                "Cross-encoder (ms-marco-MiniLM-L-6-v2) re-ranks by relevance"
            ),
        },
        {
            "title": "LLM GENERATE + CITE",
            "subtitle": (
                "Generate answer; cite document and page"
            ),
        },
    ]

    # Theme colors
    if theme == "mono":
        bg_color = "white"
        box_fill = "#FFFFFF"
        box_stroke = "#000000"
        text_color = "#000000"
        arrow_color = "#000000"
    elif theme == "lightgray":
        bg_color = "#FFFFFF"
        box_fill = "#F5F5F5"
        box_stroke = "#222222"
        text_color = "#111111"
        arrow_color = "#222222"
    else:
        bg_color = "white"
        box_fill = "#FFFFFF"
        box_stroke = "#000000"
        text_color = "#000000"
        arrow_color = "#000000"

    # Layout parameters (scaled)
    width = int(720 * scale)
    left_margin = int(80 * scale)
    right_margin = int(80 * scale)
    usable_width = width - left_margin - right_margin
    box_width = int(usable_width)
    box_height = int(110 * scale)
    box_radius = int(14 * scale)
    v_gap = int(26 * scale)
    top_margin = int(40 * scale)
    bottom_margin = int(40 * scale)

    num_steps = len(steps)
    total_boxes_height = num_steps * box_height
    total_gaps_height = (num_steps - 1) * v_gap
    height = (
        top_margin + total_boxes_height + total_gaps_height + bottom_margin
    )

    # Create canvas
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Fonts
    title_font = load_font(int(24 * scale))  # Top "Methods" header
    box_title_font = load_font(int(22 * scale))
    box_subtitle_font = load_font(int(16 * scale))

    # Optional small title at the top
    title_text = "Methods"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    title_x = (width - title_w) // 2
    title_y = int(10 * scale)
    draw.text((title_x, title_y), title_text, fill=text_color, font=title_font)

    # Adjust starting Y to account for title spacing
    current_top = max(top_margin, title_y + title_h + int(10 * scale))

    # Draw boxes and arrows
    center_x = left_margin + box_width // 2
    for idx, item in enumerate(steps):
        left = left_margin
        top = current_top
        right = left + box_width
        bottom = top + box_height

        # Box
        draw.rounded_rectangle(
            [left, top, right, bottom],
            radius=box_radius,
            fill=box_fill,
            outline=box_stroke,
            width=int(2 * scale),
        )
        # Labels (sanitize if ASCII-only)
        title_text = item.get("title", "").upper()
        subtitle_text = item.get("subtitle", "")
        if ascii_only:
            title_text = sanitize_text(title_text)
            subtitle_text = sanitize_text(subtitle_text)
        draw_centered_title_subtitle(
            draw,
            (left, top, right, bottom),
            title_text,
            subtitle_text,
            box_title_font,
            box_subtitle_font,
            text_color,
            line_spacing=int(6 * scale),
        )

        # Arrow to next
        if idx < len(steps) - 1:
            arrow_start_y = bottom
            arrow_end_y = bottom + v_gap
            draw_arrow_down(
                draw,
                center_x,
                arrow_start_y + int(6 * scale),
                arrow_end_y - int(6 * scale),
                arrow_color,
                shaft_width=int(3 * scale),
                head_size=int(8 * scale),
            )

        current_top = bottom + v_gap

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a compact Methods diagram (no Mermaid)"
    )
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=Path("diagrams/methods_diagram_poster.png"),
        help="Output image path (.png)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for the image size (e.g., 0.8 small, 1.2 larger)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="mono",
        choices=["mono", "lightgray"],
        help="Color theme",
    )
    parser.add_argument(
        "--unicode",
        action="store_true",
        help=(
            "Allow Unicode symbols in labels (may show as boxes on some"
            " systems)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ascii_only = not args.unicode
    generate_methods_diagram(
        args.output,
        scale=args.scale,
        theme=args.theme,
        ascii_only=ascii_only,
    )
    print(f"\u2713 Saved diagram to: {args.output}")


if __name__ == "__main__":
    main()
