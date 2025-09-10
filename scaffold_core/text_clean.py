"""
Text normalization utilities for cleaning OCR and line-wrap artifacts.

These utilities are safe for both input (extracted PDF text) and output
(LLM responses). For output usage, prefer `clean_model_output` which
conservatively normalizes casing/spaces without altering semantics.
"""

import re


_ACRONYM_WHITELIST = {
    "HVAC", "ASHRAE", "LEED", "LCA", "EPD", "EUI", "PV", "BIM",
    "CO2", "GHG", "IAQ", "RFI", "RFP",
}


def _fix_mixed_caps_token(token: str) -> str:
    """Reduce mid-word capitalization while preserving acronyms.

    Examples:
    - sustainABILITY -> sustainability
    - ENERGY eFFICIENCY (tokenized) -> normalize each token
    - Keep full acronyms (HVAC, ASHRAE) unchanged
    """
    if not token or token.isspace():
        return token

    # Preserve pure acronyms / all-caps short tokens
    if token.isupper() and len(token) <= 6 and token in _ACRONYM_WHITELIST:
        return token

    # If token mixes cases and tail is mostly uppercase, lower the tail
    if any(c.islower() for c in token) and any(c.isupper() for c in token):
        head = token[0]
        tail = token[1:]
        upper_count = sum(1 for c in tail if c.isupper())
        if upper_count >= max(2, len(tail) // 2):
            return head + tail.lower()
    return token


def _remove_inword_commas(text: str) -> str:
    """Fix comma-in-word typos: spell,ing -> spelling."""
    return re.sub(r"([A-Za-z]),([A-Za-z])", r"\1\2", text)


def _merge_hyphen_linebreaks(text: str) -> str:
    """Merge hyphenated line breaks: sustain-
    ability -> sustainability.
    """
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _normalize_newlines(text: str) -> str:
    """Collapse single newlines within paragraphs to spaces.

    Preserve blank lines.
    """
    # Replace single newlines (not double) with space
    # Use a compiled regex to avoid an overly long literal
    _single_newline_rx = re.compile(
        r"(?<!\n)\n(?!\n)"
    )
    return _single_newline_rx.sub(" ", text)


def normalize_extracted_text(text: str) -> str:
    """Normalize raw PDF-extracted text to reduce downstream artifacts."""
    if not text:
        return text

    text = _merge_hyphen_linebreaks(text)
    text = _normalize_newlines(text)
    text = _remove_inword_commas(text)

    # Token-wise mixed-caps correction, preserving punctuation separators
    tokens = re.split(r"(\W+)", text)
    tokens = [_fix_mixed_caps_token(tok) for tok in tokens]
    cleaned = "".join(tokens)

    # Collapse excessive spaces
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _has_mixed_caps_or_inword_splits(text: str) -> bool:
    """Detect obvious mid-word capitalization and in-word splits.

    - Mixed caps like sustainABILITY or ENERGy
    - In-word splits like Fl uid or cur ricular (tokenized)
    """
    if not text:
        return False

    # Mixed caps within a token: letters with multiple uppers in tail
    mixed_caps = re.search(r"\b[A-Za-z]+[A-Z]{2,}[a-z]*\b", text)

    # In-word splits: short-fragment splits that shouldn't be separate
    inword_split = re.search(r"\b([A-Za-z]{1,2})\s+([A-Za-z]{2,})\b", text)

    # Comma inside a word
    inword_comma = re.search(r"[A-Za-z],[A-Za-z]", text)

    return bool(mixed_caps or inword_split or inword_comma)


def clean_model_output(text: str) -> str:
    """Conservatively normalize LLM output to improve readability.

    - Fix in-word commas
    - Collapse single newlines -> spaces, preserve blank lines
    - Collapse multiple spaces
    - Normalize mid-word capitalization heuristically while preserving acronyms
    """
    if not isinstance(text, str) or not text:
        return text

    # Apply conservative fixes
    text = _remove_inword_commas(text)

    # Token-wise mixed-caps correction, preserving punctuation separators
    tokens = re.split(r"(\W+)", text)
    tokens = [_fix_mixed_caps_token(tok) for tok in tokens]
    cleaned = "".join(tokens)

    # Normalize spaces/newlines
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


__all__ = [
    "normalize_extracted_text",
    "clean_model_output",
    "_has_mixed_caps_or_inword_splits",
]
