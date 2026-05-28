# Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity

This repository contains the source code and configuration files for the **Privacy-Preserving RAG (PPR)** framework, designed to mitigate privacy leakage in Retrieval-Augmented Generation systems while preserving contextual fidelity.

---

## 🌟 Key Features

*   **Multi-Agent Task Decomposition**: Offloads privacy sanitization to three independent specialized agents, bypassing the prompt conflict issues and performance trade-offs of single-agent prompting.
*   **Synergistic Privacy Extraction**: Combines deterministic rules (regex/NER) for explicit identifiers with generative LLM inference for implicit quasi-identifiers.
*   **Structured Attribute Deconstruction**: Parses documents into subject-predicate-value knowledge slots, decoupling semantic context from original linguistic leak vectors.
*   **Fine-Grained Conflict Routing**: Dynamically resolves overlap conflicts between private identifiers and critical facts via either placeholder substitution or high-level abstraction.
*   **Asymmetric Retrieval Architecture**: Decouples retrieval indexing from generation payloads. All queries are retrieved using original documents, but only offline-sanitized contexts are served to the generation model.
*   **Zero Online Latency Penalty**: Operates asynchronously as an offline database ingestion pipeline, introducing no overhead to the real-time inference loop.

---

## 🏗️ System Architecture

The following diagram illustrates the lifecycle of our framework, showcasing the decoupling of data ingestion/sanitization (offline) and asymmetric retrieval/generation (online).

![System Architecture](assets/structure.png)

---

## 📂 Codebase Structure

Below is an overview of the key directories and files in this repository. Click the links below to navigate to the folders and files:

*   **[`configs/`](file:///c:/code/ppr-code/configs)**: Contains the multi-agent system configurations (`agents_config.yaml`).
*   **[`agents/`](file:///c:/code/ppr-code/agents)**: Core directory containing agent implementation modules (System coordinator, LLM client, Privacy Extraction, Semantic Analysis, and Reconstruction agents).
*   **[`prompts/`](file:///c:/code/ppr-code/prompts)**: Prompt templates used by the respective agents.
*   **[`data_preparation/`](file:///c:/code/ppr-code/data_preparation)**: Scripts to download, parse, and partition ChatDoctor and Wiki-PII datasets.
*   **[`retrieval/`](file:///c:/code/ppr-code/retrieval)**: Code for generating dense and sparse BM25 retrieval contexts.
*   **[`run_pipeline.py`](file:///c:/code/ppr-code/run_pipeline.py)**: The main entry point to execute the offline sanitization pipeline.
*   **[`requirements.txt`](file:///c:/code/ppr-code/requirements.txt)**: Python environment package declarations.

---

## 🛠️ Quick Start

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
pip install pymupdf
```

### 2. Configuration
Configure model paths and settings (API mode or offline vLLM mode) in **[`configs/agents_config.yaml`](file:///c:/code/ppr-code/configs/agents_config.yaml)**.

### 3. Usage
Run a quick test or execute the sanitization pipeline:
```bash
# Run step-by-step test runner on a sample text
python agents/system.py --config configs/agents_config.yaml

# Run the complete sanitization pipeline
python run_pipeline.py --dataset chatdoctor --qtype target_questions
```
Sanitization outputs are saved as Parquet files under the `results/` directory.
