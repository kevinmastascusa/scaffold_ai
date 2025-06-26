#!/usr/bin/env python3
"""
Post-process chunked_text_extracts.json to fix combined words and save a cleaned version.
"""
import json
import re
import sys
from pathlib import Path

# Try to import wordninja for advanced splitting
try:
    import wordninja
    HAS_WORDNINJA = True
except ImportError:
    HAS_WORDNINJA = False
    print("[INFO] wordninja not installed. Only custom and CamelCase splitting will be used.")

# Add scaffold_core to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scaffold_core.config import CHUNKED_TEXT_EXTRACTS_JSON

INPUT_PATH = Path(CHUNKED_TEXT_EXTRACTS_JSON)
OUTPUT_PATH = INPUT_PATH.parent / "chunked_text_extracts_cleaned.json"

# Custom domain-specific replacements
DOMAIN_REPLACEMENTS = {
    "environmentalsustainability": "environmental sustainability",
    "sustainabilityeducation": "sustainability education",
    "engineeringeducation": "engineering education",
    "environmentallysustainable": "environmentally sustainable",
    "highereducation": "higher education",
    "environmentalengineering": "environmental engineering",
    "socialsustainability": "social sustainability",
    "civilengineering": "civil engineering",
    "mechanicalengineering": "mechanical engineering",
    "bioengineering": "bioengineering",
    # Add more as needed
}

def split_camel_case(text):
    # Split CamelCase and PascalCase
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

def fix_combined_words(text):
    # Apply domain-specific replacements
    for k, v in DOMAIN_REPLACEMENTS.items():
        text = text.replace(k, v)
    # Split CamelCase
    text = split_camel_case(text)
    # Optionally use wordninja for further splitting
    if HAS_WORDNINJA:
        # Only split long words (8+ chars, no spaces)
        def split_long_words(match):
            word = match.group(0)
            if len(word) > 8:
                return ' '.join(wordninja.split(word))
            return word
        text = re.sub(r'\b\w{8,}\b', split_long_words, text)
    return text

def main():
    print(f"Loading chunks from {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Processing {len(chunks)} chunks...")
    for chunk in chunks:
        if 'text' in chunk:
            chunk['text'] = fix_combined_words(chunk['text'])
    print(f"Saving cleaned chunks to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Done!")

if __name__ == "__main__":
    main() 