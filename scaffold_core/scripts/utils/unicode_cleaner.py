import unicodedata
import re

def clean_text(text):
    """Clean text by normalizing, removing control characters, and filtering unwanted Unicode ranges."""
    # Normalize to NFKC form — handles ligatures, superscripts, etc.
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters, private use, surrogates, etc.
    text = ''.join(
        c for c in text
        if unicodedata.category(c)[0] not in ('C', 'S') and ord(c) >= 32
    )

    # Remove specific Unicode ligatures or math symbols if still present
    bad_ranges = [
        (0x2000, 0x206F),   # General Punctuation (e.g. \u200b)
        (0x2100, 0x214F),   # Letterlike Symbols (e.g. ℓ, ℮)
        (0x2200, 0x22FF),   # Mathematical Operators (e.g. ∑, √)
        (0x2500, 0x257F),   # Box Drawing
        (0x1D400, 0x1D7FF), # Mathematical Alphanumeric Symbols
    ]
    def remove_bad_chars(char):
        cp = ord(char)
        return not any(start <= cp <= end for start, end in bad_ranges)

    text = ''.join(filter(remove_bad_chars, text))

    # Remove repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_text_preserve_math(text):
    """Clean text while preserving mathematical symbols and ranges."""
    # Normalize to NFKC form — handles ligatures, superscripts, etc.
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters, private use, surrogates, etc.
    text = ''.join(
        c for c in text
        if unicodedata.category(c)[0] not in ('C', 'S') and ord(c) >= 32
    )

    # Define ranges to preserve mathematical symbols
    math_ranges = [
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x2300, 0x23FF),  # Miscellaneous Technical
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
        (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    ]

    def preserve_math_chars(char):
        cp = ord(char)
        return any(start <= cp <= end for start, end in math_ranges)

    # Filter out unwanted characters while preserving math symbols
    text = ''.join(
        c for c in text
        if preserve_math_chars(c) or unicodedata.category(c)[0] not in ('C', 'S')
    )

    # Remove repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Example usage
def example_usage():
    sample_text = """Logic\nPaper type Research paper\n1. Introduction\nUnsustainable pr there are new lines what is the issue"""
    print("Original Text:")
    print(sample_text)

    cleaned_text = clean_text(sample_text)
    print("\nCleaned Text:")
    print(cleaned_text)

    math_sample = """E = mc² is a famous equation. Let's not forget about π (pi) which is approximately 3.14159."""
    print("\nMath Sample Text:")
    print(math_sample)

    math_cleaned = clean_text_preserve_math(math_sample)
    print("\nCleaned Math Text:")
    print(math_cleaned)

if __name__ == "__main__":
    example_usage()