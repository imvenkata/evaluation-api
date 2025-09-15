# Technical Design Document: Synthetic Ground Truth Generation Package

**Date:** September 15, 2025
**Version:** 1.0
**Status:** Design

---

### 1. Executive Summary

This document outlines the technical architecture and design for a new Python package: the **Synthetic Ground Truth Generation** module. This module is an extension of the existing `evaluation_api` package, designed to solve the primary bottleneck in search evaluation: the lack of high-quality, diverse, and challenging test data.

The system ingests a corpus of documents or pre-processed chunks and generates a "golden" dataset of `(query, expected_doc_ids)` pairs. This output dataset is formatted to be consumed directly by the existing `metrics_evaluator.py` module, creating a closed-loop system for evaluating and improving search performance.

The core differentiator of this solution is its "validation-first" and "distractor-aware" architecture. It does not simply generate simple (context -> query) pairs. Instead, it programmatically identifies "hard negatives" (semantically similar but incorrect chunks) and generates queries that *specifically* test a search model's ability to **differentiate**, not just recall. This results in a far more rigorous and realistic evaluation of search metrics like NDCG and MRR.

### 2. System Architecture

The package is designed as a modular, configuration-driven pipeline executed via a command-line interface (CLI). The architecture provides flexibility by allowing the user to start from either raw documents or pre-computed chunks.

#### 2.1. Architectural Flow

The pipeline executes in a series of discrete, testable stages:

```text
[generation_config.py]
    (Defines input_type, thresholds, paths, prompts)
            │
            ▼
[generation/cli.py] (Orchestrator)
            │
            ├─ IF input_type == "documents" ───► [data_loader.py] (Loads raw docs)
            │                                           │
            │                                           ▼
            │                                    [chunker.py] (using LlamaIndex)
            │                                           │
            │                                           ▼
            │                                    [embedder.py] (using utils/client)
            │
            └─ IF input_type == "chunks" ────► [data_loader.py] (Loads pre-processed chunks)
            │
            ▼
(Standardized `list[ChunkData]` object)
            │
            ▼
[generation/chunk_validator.py]
    (Filters by token length, stopword ratio, duplicate similarity)
            │
            ▼
[generation/chunk_selector.py]
    (Uses FAISS to find a "Golden Chunk" + "Distractor Chunks")
            │
            ▼
[generation/query_generator.py] (STUBBED)
    (Uses "Distractor-Aware" prompt to generate a query)
            │
            ▼
[generation/evaluation_layer.py] (STUBBED)
    (Uses Ragas/DeepEval to validate the (Query, Golden Chunk) pair)
            │
            ▼
[output/synthetic_ground_truth.jsonl]
    (Ready for consumption by metrics_evaluator.py)
```

#### 2.2. Module Responsibilities

This solution is integrated directly into the `evaluation_api` as a new `generation` module.

  * **`generation_config.py`**: **The Control Panel.** A Python-based configuration file (matching the existing `metrics_config.py`) that defines all parameters: input paths, `input_type` ("chunks" or "documents"), validation thresholds, LLM prompt templates, and evaluation cutoffs.

  * **`generation/cli.py`**: **The Orchestrator.** A CLI entry point (e.g., `python -m evaluation_api.generation.cli --config ...`) that reads the config and executes each step of the pipeline in sequence.

  * **`generation/models.py`**: **The Data Contracts.** A set of Python `dataclasses` (e.g., `ChunkData`, `SelectionBundle`, `ValidatedGroundTruth`) that define the standardized objects passed between modules.

  * **`generation/data_loader.py`**: **The Flexible Entry Point.** Reads the `input_type` from the config.

      * `if "chunks"`: Loads pre-processed JSONL files into `ChunkData` objects. (This is the "chunks-first" implementation).
      * `if "documents"`: (Placeholder) Will use `LlamaIndex` to read raw files (PDF, .md, etc.) and orchestrate chunking and embedding.

  * **`generation/chunk_validator.py`**: **The Quality Gate.** This is the *first* validation step. It iterates over all loaded chunks and filters out low-quality data (e.g., too short/long, high stopword ratio) and near-duplicates (using `scikit-learn`'s `cosine_similarity`).

  * **`generation/chunk_selector.py`**: **The Strategic Heart.** This is the core of our "hard negative" logic. It builds a `FAISS` vector index of all valid chunks. For each potential query, it selects:

    1.  **Golden Chunk(s):** The chunk(s) that will serve as the "correct" answer.
    2.  **Distractor Chunks(s):** The top-k nearest neighbors (by embedding similarity) that are *not* the golden chunk. These are the "hard negatives."

  * **`generation/query_generator.py`**: **The Creative Engine.** This module (currently stubbed) receives the `SelectionBundle` (golden + distractors) from the selector. It constructs a "distractor-aware" prompt and calls the LLM (e.g., Azure OpenAI).

      * **Sample Prompt:** `"Generate a user query that is ONLY answered by the 'Golden Context' and NOT by the 'Distractor Context'. \n\n ## Golden Context: [text]... \n\n ## Distractor Context: [text]..."`

  * **`generation/evaluation_layer.py`**: **The Final Approval.** This is the *second* validation step (currently stubbed). After a query is generated, this module uses SOTA RAG evaluation frameworks like `Ragas` and `DeepEval` to score the (query, golden\_chunk) pair for metrics like `Faithfulness` and `Context Relevance`. If the pair fails to meet the config's quality threshold, it is discarded.

### 3. Core Tools and Packages

This solution is built by intelligently combining best-in-class open-source libraries with custom logic.

| Category | Package(s) | Purpose |
| :--- | :--- | :--- |
| **Core Framework** | `Python 3.10+` | The runtime environment. |
| **Orchestration** | `argparse` | For the `cli.py` entry point. |
| **Data Handling** | `numpy`, `pandas` | For high-performance data manipulation and math. |
| **Vector Search** | `faiss-cpu` | High-speed similarity search in `chunk_selector.py` to find distractors. |
| **Validation** | `scikit-learn` | For `cosine_similarity` in `chunk_validator.py` to find duplicates. |
| **LLM Interface** | `openai` | Client library for interfacing with Azure OpenAI (or other) models. |
| **RAG Evaluation** | `ragas`, `deepeval` | (For `evaluation_layer.py`) SOTA frameworks to validate the quality of generated QA pairs. |
| **Doc Processing** | `llama-index` | (For `data_loader.py` "documents" path) SOTA framework for document loading and chunking. |

### 4. Justification: Why This is the Best Solution

This design is superior to a simple generation script because it is **flexible, rigorous, and integrated.**

1.  **Flexibility (Dual-Input Architecture)**
    The system does not force a "one-size-fits-all" approach. By supporting both raw documents and pre-processed chunks, it empowers all users. Teams with an existing chunking pipeline can use it; teams starting from scratch can rely on our built-in `LlamaIndex` capabilities.

2.  **Focus on Quality, Not Quantity (Dual-Validation Gates)**
    A large, low-quality test set is useless. Our solution incorporates two critical validation gates:

      * **Pre-Validation (`chunk_validator.py`):** Ensures we don't waste time and money generating queries from "junk" chunks.
      * **Post-Validation (`evaluation_layer.py`):** Uses SOTA libraries like `Ragas` and `DeepEval` to *prove* that the generated query is faithful, relevant, and grounded in the context. This builds a *trustworthy* dataset.

3.  **The "Distractor-Aware" Differentiator (Solves for "Hard Negatives")**
    This is the single most important feature of this architecture.

    >   * **The Problem:** Most generators create simple (Context -\> Query) pairs. A basic search model can "pass" these tests by matching keywords, resulting in inflated metrics.

    >   * **Our Solution:** We solve this by finding *semantically similar* "distractor" chunks (`chunk_selector.py`) and using a "distractor-aware" prompt (`query_generator.py`). This creates queries that force the search model to **differentiate** between highly similar results. This is the *only* way to truly test the ranking precision (NDCG, MRR) of a sophisticated vector or hybrid search system.

4.  **Modular and Integrated**
    This is not a standalone Jupyter notebook. It is a production-ready Python module designed to live inside the existing `evaluation_api` package. It follows existing patterns (e.g., `configs/` files), is testable (each module is discrete), and its output (`.jsonl`) is the direct input for the rest of the package, creating a seamless, end-to-end evaluation workflow.
