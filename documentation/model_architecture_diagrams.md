# Model Architecture Diagrams

**Last Updated:** June 29, 2025  
**Status:** Updated for Hugging Face Integration

This document contains visual diagrams of the confirmed working models in the Scaffold AI pipeline, based on the actual implemented codebase.

---

## 1. Complete Model Pipeline Architecture

```mermaid
flowchart TD
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
    H --> H1[Mistral-7B-Instruct-v0.2<br/>Hugging Face Transformers]
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
    class I output
```

---

## 2. Model Details and Specifications

### Embedding Model: all-MiniLM-L6-v2
```mermaid
flowchart LR
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
    class E,F,G,H details
```

### Cross-Encoder Model: ms-marco-MiniLM-L-6-v2
```mermaid
flowchart LR
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
    class E,F,G,H,I details
```

### LLM Model: Mistral via Hugging Face
```mermaid
flowchart LR
    A[User Query] --> D[Mistral LLM]
    B[Top 10 Chunks] --> D
    C[System Prompt] --> D
    D --> E[Grounded Answer]
    
    subgraph "LLM Details"
        F[Model: Mistral-7B-Instruct-v0.2]
        G[Platform: Hugging Face Transformers]
        H[Integration: Python API]
        I[Role: Fact-checking assistant]
        J[Constraint: Quote only from chunks]
    end
    
    classDef llm fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef answer fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef details fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class D llm
    class E answer
    class F,G,H,I,J details
```

---

## 3. Data Flow Between Models

```mermaid
flowchart TD
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
        S[LLM<br/>Mistral via Hugging Face]
    end
    
    classDef data fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef flow fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class A,B,C,D,E,F,G,H,I,J,K,L data
    class Q,R,S model
    class M,N,O,P flow
```

---

## 4. Model Configuration (Current Settings)

| Model | Configuration | Location | Status |
|-------|---------------|----------|--------|
| **Embedding** | `all-MiniLM-L6-v2` | `scaffold_core/config.py` | ✅ Working |
| **Cross-Encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `scaffold_core/vector/query.py` | ✅ Working |
| **LLM** | `mistralai/Mistral-7B-Instruct-v0.2` (Hugging Face) | `scaffold_core/config.py` | ✅ Working |
| **FAISS Index** | `IndexFlatL2` | `scaffold_core/vector/transformVector.py` | ✅ Working |

---

## 5. Performance Characteristics

### Embedding Model
- **Speed**: ~1000 chunks/second
- **Memory**: ~80MB model size
- **Quality**: Good semantic similarity
- **Vector Dimension**: 384

### Cross-Encoder Model
- **Speed**: ~100 pairs/second
- **Memory**: ~80MB model size
- **Quality**: High reranking accuracy
- **Input**: Query-chunk pairs

### LLM Model
- **Speed**: ~2-5 seconds per query
- **Memory**: ~7GB (Mistral-7B-Instruct-v0.2)
- **Quality**: High-quality reasoning with constraints
- **Platform**: Hugging Face Transformers (Python API)

---

## 6. Model Dependencies

```mermaid
graph TD
    A[sentence-transformers] --> B[all-MiniLM-L6-v2]
    A --> C[cross-encoder/ms-marco-MiniLM-L-6-v2]
    D[torch] --> A
    E[transformers] --> A
    F[faiss-cpu] --> G[FAISS Index]
    H[transformers] --> I[Mistral LLM]
    J[huggingface-hub] --> I
    
    subgraph "Python Dependencies"
        K[numpy]
        L[requests]
        M[PyMuPDF]
        N[sentencepiece]
        O[accelerate]
    end
    
    classDef library fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class A,D,E,F,H,J,K,L,M,N,O library
    class B,C,G,I model
```

---

*These diagrams reflect the current working implementation as of June 29, 2025. All models shown are confirmed to be functional and integrated into the pipeline. The system has been successfully migrated from Ollama to Hugging Face Transformers for LLM functionality.* 