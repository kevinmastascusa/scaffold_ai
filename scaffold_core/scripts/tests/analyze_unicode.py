#!/usr/bin/env python3
"""
A comprehensive, consolidated script to analyze and report on Unicode
characters in specified JSON output files.

This script identifies non-ASCII characters, categorizes them, and provides a
detailed report directly to the console. It serves as the primary tool for
debugging and validating text encoding and content across the data processing
pipeline.
"""
import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict

# Ensure the project root is in the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

from scaffold_core.config import (
    CHUNKED_TEXT_EXTRACTS_JSON,
    FULL_TEXT_EXTRACTS_JSON,
    VECTOR_PROCESSED_JSON,
)


def analyze_text(text: str) -> dict:
    """
    Analyzes a string of text to identify and categorize Unicode characters.

    Args:
        text: The text to analyze.

    Returns:
        A dictionary containing the analysis results, including character
        categories, a list of non-ASCII characters, and control characters.
    """
    char_categories = defaultdict(int)
    non_ascii_chars = []
    control_chars = []

    for char in text:
        category = unicodedata.category(char)
        char_categories[category] += 1

        if ord(char) > 127:  # Non-ASCII
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                char_name = f"Unassigned (U+{ord(char):04X})"
            non_ascii_chars.append((char, char_name, category, ord(char)))

        if category.startswith('C'):  # Control characters
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                char_name = f"Unassigned (U+{ord(char):04X})"
            control_chars.append((char, char_name, ord(char)))

    return {
        'categories': dict(char_categories),
        'non_ascii_chars': list(set(non_ascii_chars)),  # Unique chars
        'control_chars': list(set(control_chars))  # Unique chars
    }


def analyze_file(file_path: str) -> dict:
    """
    Analyzes all text content within a given JSON file for Unicode issues.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A dictionary summarizing the findings for the entire file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {'error': 'File not found.'}
    except json.JSONDecodeError:
        return {'error': 'Invalid JSON format.'}
    except Exception as e:
        return {'error': str(e)}

    all_non_ascii = set()
    all_control_chars = set()
    documents_with_issues = 0
    total_docs = len(data)
    total_text_length = 0

    for item in data:
        text = item.get('text', item.get('full_text', ''))
        total_text_length += len(text)
        analysis = analyze_text(text)

        if analysis['non_ascii_chars']:
            documents_with_issues += 1
            all_non_ascii.update(analysis['non_ascii_chars'])
            all_control_chars.update(analysis['control_chars'])

    return {
        'file_path': file_path,
        'total_documents': total_docs,
        'documents_with_issues': documents_with_issues,
        'total_text_length': total_text_length,
        'unique_non_ascii_chars': sorted(
            list(all_non_ascii), key=lambda x: x[3]
        ),
        'unique_control_chars': sorted(
            list(all_control_chars), key=lambda x: x[2]
        ),
    }


def print_report(results: dict):
    """
    Prints a formatted analysis report to the console.

    Args:
        results: The analysis results from the `analyze_file` function.
    """
    if results.get('error'):
        print(
            f"✗ Error analyzing {results.get('file_path', 'file')}: "
            f"{results['error']}"
        )
        return

    print("\n" + "=" * 80)
    print(
        f"📊 UNICODE ANALYSIS REPORT FOR: "
        f"{os.path.basename(results['file_path'])}"
    )
    print("=" * 80)

    # Print summary stats
    doc_percent = (
        (results['documents_with_issues'] / results['total_documents'] * 100)
        if results['total_documents'] > 0 else 0
    )
    print(f"  - Documents Analyzed:      {results['total_documents']:,}")
    print(
        f"  - Documents with Non-ASCII: "
        f"{results['documents_with_issues']:,} ({doc_percent:.2f}%)"
    )
    print(f"  - Total Characters:        {results['total_text_length']:,}")
    print(
        f"  - Unique Non-ASCII Types:  "
        f"{len(results['unique_non_ascii_chars']):,}"
    )
    print(
        f"  - Unique Control Chars:    "
        f"{len(results['unique_control_chars']):,}"
    )

    # Print Non-ASCII character details
    if results['unique_non_ascii_chars']:
        print("\n" + "-" * 40)
        print("🔍 Found Non-ASCII Characters:")
        print("-" * 40)
        for char, name, category, code in results['unique_non_ascii_chars']:
            print(
                f"  - '{char}' (U+{code:04X}) | Category: {category} "
                f"| Name: {name}"
            )
    else:
        print("\n✅ No Non-ASCII characters found.")

    # Print Control character details
    if results['unique_control_chars']:
        print("\n" + "-" * 40)
        print("⚠️ Found Control Characters:")
        print("-" * 40)
        for char, name, code in results['unique_control_chars']:
            print(f"  - U+{code:04X} | Repr: {repr(char)} | Name: {name}")

    print("\n" + "=" * 80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze Unicode characters in specified JSON files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='*',
        default=[],
        help="One or more paths to JSON files to analyze.\n"
             "If no files are provided, it will analyze the default files:\n"
             f"- {os.path.basename(str(CHUNKED_TEXT_EXTRACTS_JSON))}\n"
             f"- {os.path.basename(str(FULL_TEXT_EXTRACTS_JSON))}\n"
             f"- {os.path.basename(str(VECTOR_PROCESSED_JSON))}"
    )
    args = parser.parse_args()

    files_to_analyze = args.files
    if not files_to_analyze:
        files_to_analyze = [
            str(CHUNKED_TEXT_EXTRACTS_JSON),
            str(FULL_TEXT_EXTRACTS_JSON),
            str(VECTOR_PROCESSED_JSON)
        ]

    print(f"Starting Unicode analysis for {len(files_to_analyze)} file(s)...")

    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            results = analyze_file(file_path)
            print_report(results)
        else:
            print(f"\n✗ File not found: {file_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main() 