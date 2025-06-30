#!/usr/bin/env python3
"""
Simple Diagram Generation Script for Scaffold AI Data Pipeline
Generates a Mermaid flow chart and converts it to an image file using multiple methods.
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_pipeline_diagram():
    """Generate the complete data pipeline flow chart in Mermaid format."""
    
    mermaid_diagram = '''flowchart TD
    %% Input Stage
    A[ACADEMIC PDFS<br/>273 Documents<br/>14.9M+ Characters] --> B[PROJECT SETUP]
    
    %% Setup Stage
    B --> B1[python setup.py]
    B1 --> B2[Create Directory Structure]
    B2 --> B3[Validate Configuration]
    B3 --> B4[Centralized Config Ready]
    
    %% Document Processing Stage
    B4 --> C[DOCUMENT PROCESSING]
    C --> C1[ChunkTest.py<br/>Page-based Chunking]
    C --> C2[ChunkTest_Math.py<br/>Math-aware Processing]
    C1 --> C3[Extract Text by Page]
    C2 --> C4[Detect Math Content]
    C3 --> C5[Extract Metadata<br/>Authors, DOIs, Titles]
    C4 --> C6[Math Analysis]
    C5 --> C7[4,859 Text Chunks]
    C6 --> C7
    
    %% Text Analysis Stage
    C7 --> D[TEXT ANALYSIS & CLEANING]
    D --> D1[generate_unicode_report.py]
    D --> D2[postprocess_combined_words.py]
    D --> D3[generate_combined_words_report.py]
    D1 --> D4[Unicode Analysis<br/>44,367 Unicode Chars<br/>0.30% Content]
    D2 --> D5[Fix Combined Words<br/>10,262 Total<br/>7,383 Unique]
    D3 --> D6[Generate Analysis Reports]
    D4 --> D7[Clean Text Data]
    D5 --> D7
    D6 --> D8[Quality Reports]
    
    %% Vectorization Stage
    D7 --> E[VECTORIZATION]
    E --> E1[transformVector.py]
    E1 --> E2[Load Cleaned Chunks]
    E2 --> E3[Final Unicode Cleaning]
    E3 --> E4[Generate Embeddings<br/>sentence-transformers<br/>all-MiniLM-L6-v2]
    E4 --> E5[Create FAISS Index]
    E5 --> E6[Vector Database Ready]
    
    %% Output Files
    E6 --> F1[vector_outputs/embeddings.npy]
    E6 --> F2[vector_outputs/faiss_index.bin]
    E6 --> F3[vector_outputs/chunk_metadata.json]
    
    %% Search Stage
    F1 --> G[SEMANTIC SEARCH & RETRIEVAL]
    F2 --> G
    F3 --> G
    G --> G1[query.py - Semantic Search]
    G1 --> G2[FAISS Similarity Search]
    G2 --> G3[Cross-Encoder Reranking]
    G3 --> G4[Top 10 Relevant Chunks]
    
    %% LLM Integration Stage
    G4 --> I[LLM ANSWER GENERATION]
    I --> I1[llm.py - LLMManager]
    I1 --> I2[Mistral-7B-Instruct-v0.2<br/>Hugging Face Transformers]
    I2 --> I3[Generate Grounded Response]
    I3 --> I4[Curriculum Recommendations<br/>with Citations]
    
    %% Reports and Analysis
    D8 --> H[Analysis Reports]
    H --> H1[outputs/unicode_report.txt]
    H --> H2[outputs/combined_words_analysis_report.txt]
    H --> H3[Processing Statistics]
    
    %% Styling with better contrast
    classDef stage1 fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
    classDef stage2 fill:#f0f0f0,stroke:#666666,stroke-width:2px
    classDef stage3 fill:#e6ffe6,stroke:#006600,stroke-width:2px
    classDef stage4 fill:#fff2e6,stroke:#cc6600,stroke-width:2px
    classDef stage5 fill:#f0f0ff,stroke:#6600cc,stroke-width:2px
    classDef stage6 fill:#f5f5f5,stroke:#333333,stroke-width:2px
    classDef stage7 fill:#ffe6f0,stroke:#cc0066,stroke-width:2px
    
    class A,B,B1,B2,B3,B4 stage1
    class C,C1,C2,C3,C4,C5,C6,C7 stage2
    class D,D1,D2,D3,D4,D5,D6,D7,D8 stage3
    class E,E1,E2,E3,E4,E5,E6 stage4
    class F1,F2,F3 stage5
    class G,G1,G2,G3,G4,H,H1,H2,H3 stage6
    class I,I1,I2,I3,I4 stage7'''
    
    return mermaid_diagram

def generate_image_with_playwright(mermaid_code, output_path="pipeline_diagram_simple.png"):
    """Generate image using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ 
            margin: 0; 
            padding: 40px; 
            background: white; 
            font-family: Arial, sans-serif;
        }}
        .mermaid {{ 
            text-align: center; 
            max-width: 100%;
            overflow-x: auto;
        }}
        .title {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 30px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="title">Scaffold AI Data Pipeline</div>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true
            }}
        }});
    </script>
</body>
</html>'''
        
        temp_html = "temp_diagram_simple.html"
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{os.path.abspath(temp_html)}")
            page.wait_for_selector('.mermaid svg')
            # Wait a bit more for rendering to complete
            page.wait_for_timeout(2000)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
        
        # Clean up temp file
        os.remove(temp_html)
        print(f"✓ Image generated using Playwright: {output_path}")
        return True
        
    except ImportError:
        print("Playwright not installed. Install with: pip install playwright")
        return False
    except Exception as e:
        print(f"Playwright method failed: {e}")
        return False

def save_mermaid_file(mermaid_code, output_path="pipeline_diagram_simple.mmd"):
    """Save the Mermaid code to a file for manual processing."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"✓ Mermaid code saved to: {output_path}")
    return output_path

def main():
    """Main function to generate the pipeline diagram image."""
    print("Generating Scaffold AI Data Pipeline Diagram (Simple Version)...")
    
    # Generate the Mermaid diagram
    mermaid_code = generate_pipeline_diagram()
    
    # Try to generate image with Playwright
    image_generated = generate_image_with_playwright(mermaid_code)
    
    # Fallback: Save Mermaid code
    if not image_generated:
        mmd_file = save_mermaid_file(mermaid_code)
        print(f"\nCould not generate image automatically.")
        print(f"Mermaid code saved to: {mmd_file}")
        print("\nTo generate an image manually:")
        print("1. Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
        print("2. Run: mmdc -i pipeline_diagram_simple.mmd -o pipeline_diagram_simple.png")
        print("3. Or use online Mermaid editor: https://mermaid.live/")
    
    print(f"\nDiagram generation complete!")

if __name__ == "__main__":
    main() 