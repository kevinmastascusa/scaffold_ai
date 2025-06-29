#!/usr/bin/env python3
"""
Model Architecture Diagram Generation Script
Generates visual diagrams of the confirmed working models in the Scaffold AI pipeline.
"""

import os
import sys
from pathlib import Path

def generate_complete_pipeline_diagram():
    """Generate the complete model pipeline architecture diagram."""
    
    mermaid_diagram = '''flowchart TD
    %% Input Stage
    A[PDF Documents<br/>273 files, 4,859 chunks] --> B[Text Processing]
    
    %% Text Processing Stage
    B --> B1[ChunkTest.py<br/>Page-based chunking]
    B --> B2[Text cleaning & Unicode normalization]
    B --> B3[Combined words fixing]
    B3 --> B4[Clean text chunks]
    
    %% Embedding Stage
    B4 --> C[Sentence Embedding Model]
    C --> C1[all-MiniLM-L6-v2<br/>sentence-transformers]
    C1 --> C2[Generate embeddings<br/>4,859 vectors]
    C2 --> C3[FAISS Index Creation]
    C3 --> C4[Vector Database Ready]
    
    %% Query Processing Stage
    D[User Query] --> E[Query Embedding]
    E --> E1[all-MiniLM-L6-v2<br/>sentence-transformers]
    E1 --> E2[Query Vector]
    
    %% Retrieval Stage
    E2 --> F[FAISS Similarity Search]
    C4 --> F
    F --> F1[Top 50 candidates]
    
    %% Reranking Stage
    F1 --> G[Cross-Encoder Reranking]
    G --> G1[cross-encoder/ms-marco-MiniLM-L-6-v2<br/>sentence-transformers]
    G1 --> G2[Top 10 reranked chunks]
    
    %% LLM Answer Generation
    G2 --> H[LLM Answer Generation]
    H --> H1[Mistral via Ollama<br/>http://localhost:11434]
    H1 --> H2[Grounded answer with citations]
    
    %% Output
    H2 --> I[Final Response<br/>Curriculum Recommendations]
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef embedding fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef retrieval fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef llm fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef output fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    
    class A,D input
    class B,B1,B2,B3,B4 processing
    class C,C1,C2,C3,C4,E,E1,E2 embedding
    class F,F1,G,G1,G2 retrieval
    class H,H1,H2 llm
    class I output'''
    
    return mermaid_diagram

def generate_embedding_model_diagram():
    """Generate the embedding model details diagram."""
    
    mermaid_diagram = '''flowchart LR
    A[Text Chunk] --> B[all-MiniLM-L6-v2]
    B --> C[384-dimensional vector]
    C --> D[FAISS Index]
    
    subgraph "Model Details"
        E[Library: sentence-transformers]
        F[Model Size: ~80MB]
        G[Speed: Fast inference]
        H[Quality: Good for semantic search]
    end
    
    classDef model fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef vector fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef details fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class B model
    class C,D vector
    class E,F,G,H details'''
    
    return mermaid_diagram

def generate_cross_encoder_diagram():
    """Generate the cross-encoder model details diagram."""
    
    mermaid_diagram = '''flowchart LR
    A[Query] --> C[Cross-Encoder]
    B[Text Chunk] --> C
    C --> D[Relevance Score]
    
    subgraph "Cross-Encoder Details"
        E[Library: sentence-transformers]
        F[Model: cross-encoder/ms-marco-MiniLM-L-6-v2]
        G[Purpose: Reranking]
        H[Input: Query + Chunk pair]
        I[Output: Relevance score 0-1]
    end
    
    classDef model fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef score fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef details fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class C model
    class D score
    class E,F,G,H,I details'''
    
    return mermaid_diagram

def generate_llm_diagram():
    """Generate the LLM model details diagram."""
    
    mermaid_diagram = '''flowchart LR
    A[User Query] --> D[Mistral LLM]
    B[Top 10 Chunks] --> D
    C[System Prompt] --> D
    D --> E[Grounded Answer]
    
    subgraph "LLM Details"
        F[Model: Mistral-7B]
        G[Platform: Ollama]
        H[Endpoint: localhost:11434]
        I[Role: Fact-checking assistant]
        J[Constraint: Quote only from chunks]
    end
    
    classDef llm fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef answer fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef details fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class D llm
    class E answer
    class F,G,H,I,J details'''
    
    return mermaid_diagram

def generate_data_flow_diagram():
    """Generate the data flow between models diagram."""
    
    mermaid_diagram = '''flowchart TD
    %% Data Sources
    A[PDF Documents] --> B[Text Chunks]
    B --> C[Embeddings]
    C --> D[FAISS Index]
    
    %% Query Flow
    E[User Query] --> F[Query Embedding]
    F --> G[Similarity Search]
    D --> G
    G --> H[Top 50 Results]
    
    %% Reranking Flow
    H --> I[Cross-Encoder Reranking]
    I --> J[Top 10 Results]
    
    %% LLM Flow
    J --> K[LLM Processing]
    K --> L[Final Answer]
    
    %% Data Types
    subgraph "Data Types"
        M[Text: String]
        N[Embeddings: Float32 384-dim]
        O[Scores: Float32]
        P[Metadata: JSON]
    end
    
    %% Model Interactions
    subgraph "Model Interactions"
        Q[Embedding Model<br/>all-MiniLM-L6-v2]
        R[Cross-Encoder<br/>ms-marco-MiniLM-L-6-v2]
        S[LLM<br/>Mistral via Ollama]
    end
    
    classDef data fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef flow fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class A,B,C,D,E,F,G,H,I,J,K,L data
    class Q,R,S model
    class M,N,O,P flow'''
    
    return mermaid_diagram

def generate_dependencies_diagram():
    """Generate the model dependencies diagram."""
    
    mermaid_diagram = '''graph TD
    A[sentence-transformers] --> B[all-MiniLM-L6-v2]
    A --> C[cross-encoder/ms-marco-MiniLM-L-6-v2]
    D[torch] --> A
    E[transformers] --> A
    F[faiss-cpu] --> G[FAISS Index]
    H[ollama] --> I[Mistral LLM]
    
    subgraph "Python Dependencies"
        J[numpy]
        K[requests]
        L[PyMuPDF]
    end
    
    classDef library fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class A,D,E,F,J,K,L library
    class B,C,G,I model
    class H external'''
    
    return mermaid_diagram

def generate_image_with_playwright(mermaid_code, output_path, title):
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
    <div class="title">{title}</div>
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
        
        temp_html = f"temp_{title.lower().replace(' ', '_')}.html"
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{os.path.abspath(temp_html)}")
            page.wait_for_selector('.mermaid svg')
            page.wait_for_timeout(2000)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
        
        # Clean up temp file
        os.remove(temp_html)
        print(f"✓ Generated: {output_path}")
        return True
        
    except ImportError:
        print("Playwright not installed. Install with: pip install playwright")
        return False
    except Exception as e:
        print(f"Playwright method failed: {e}")
        return False

def main():
    """Main function to generate all model architecture diagrams."""
    print("Generating Model Architecture Diagrams...")
    
    # Create diagrams directory if it doesn't exist
    diagrams_dir = Path("diagrams")
    diagrams_dir.mkdir(exist_ok=True)
    
    # Generate all diagrams
    diagrams = [
        ("Complete Pipeline Architecture", generate_complete_pipeline_diagram(), "complete_pipeline_architecture.png"),
        ("Embedding Model Details", generate_embedding_model_diagram(), "embedding_model_details.png"),
        ("Cross-Encoder Model Details", generate_cross_encoder_diagram(), "cross_encoder_details.png"),
        ("LLM Model Details", generate_llm_diagram(), "llm_model_details.png"),
        ("Data Flow Between Models", generate_data_flow_diagram(), "data_flow_diagram.png"),
        ("Model Dependencies", generate_dependencies_diagram(), "model_dependencies.png")
    ]
    
    success_count = 0
    for title, mermaid_code, filename in diagrams:
        output_path = diagrams_dir / filename
        print(f"\nGenerating {title}...")
        
        if generate_image_with_playwright(mermaid_code, str(output_path), title):
            success_count += 1
        else:
            print(f"Failed to generate {filename}")
    
    print(f"\n✓ Successfully generated {success_count}/{len(diagrams)} diagrams")
    print(f"Diagrams saved to: {diagrams_dir}")
    
    if success_count < len(diagrams):
        print("\nTo generate remaining diagrams manually:")
        print("1. Install Playwright: pip install playwright")
        print("2. Install browsers: playwright install chromium")
        print("3. Run this script again")

if __name__ == "__main__":
    main() 