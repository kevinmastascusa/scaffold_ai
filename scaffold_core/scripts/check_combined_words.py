import json
import re
from pathlib import Path

CLEANED_PATH = Path("outputs/chunked_text_extracts_cleaned.json")

def find_combined_words(text):
    # Patterns for CamelCase, PascalCase, and long words
    patterns = [
        r'\b[a-z]+[A-Z][a-z]+\b',  # camelCase
        r'\b[A-Z]+[a-z]+[A-Z][a-z]+\b',  # PascalCase
        r'\b[a-z]+[A-Z]{2,}[a-z]*\b',  # words with multiple caps
        r'\b\w{15,}\b',  # very long words (15+ chars, likely combined)
    ]
    combined = []
    for pattern in patterns:
        combined.extend(re.findall(pattern, text))
    return combined

def main():
    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    found = []
    for chunk in chunks:
        text = chunk.get("text", "")
        combined = find_combined_words(text)
        if combined:
            found.extend(combined)
    print(f"Total combined words found: {len(found)}")
    if found:
        from collections import Counter
        counter = Counter(found)
        print("Most common combined words (top 20):")
        for word, count in counter.most_common(20):
            print(f"  {word}: {count} occurrences")
    else:
        print("No combined words found!")

if __name__ == "__main__":
    main() 