# HALO - Hallucination Audit Layer for Official Documents

> An open-source hallucination detection framework for high-stakes documents.  
> Implements the **Legal Fidelity Score** from published ICCIC-2025 research.

---

## The Problem

Large Language Models (LLMs) hallucinate. In most contexts this is inconvenient.  
In legal, medical, financial, and clinical documents - it is dangerous.

A payment term changed from 30 days to 15 days. A renewal clause invented from nothing. A liability limit silently altered. These are not hypothetical — they are the kinds of errors that cause lawsuits, regulatory violations, and financial loss.

Existing NLP metrics like ROUGE and BLEU measure word overlap. They cannot detect semantic hallucinations. A summary can score high on ROUGE and still contain fabricated facts.

**HALO solves this.**

---

## What HALO Does

HALO takes two inputs:
- An **original document** (legal contract, medical record, financial agreement, clinical protocol)
- An **AI-generated summary** of that document

It returns:
- A **Legal Fidelity Score** (0–100%) measuring how grounded the summary is
- A **Hallucination Rate** showing the percentage of fabricated claims
- A **clause-level audit trail** — every claim verified against the source with evidence

---

## Architecture
Original Document

│

▼

┌─────────────────┐

│  Clause Chunker │  ← RecursiveCharacterTextSplitter

└────────┬────────┘

│

▼

┌─────────────────────┐

│  Embedding & Store  │  ← Sentence Transformers + ChromaDB

└────────┬────────────┘

│

▼                    AI-Generated Summary

┌─────────────────┐                   │

│ Claim Extractor │ ◄─────────────────┘

└────────┬────────┘

│

▼

┌──────────────────────┐

│  Semantic Retrieval  │  ← Similarity search per claim

└────────┬─────────────┘

│

▼

┌──────────────────────┐

│    LLM Judge         │  ← LLaMA 3.3 via Groq API

│  SUPPORTED /         │

│  PARTIALLY_SUPPORTED │

│  NOT_SUPPORTED       │

└────────┬─────────────┘

│

▼

┌──────────────────────┐

│  Legal Fidelity      │  ← Novel metric from ICCIC-2025

│  Scorer              │

└────────┬─────────────┘

│

▼

┌──────────────────────┐

│  Audit Report UI     │  ← Streamlit

└──────────────────────┘
---

## Research Foundation

HALO is the first open-source implementation of the **Legal Fidelity Score** introduced in:

> *"A Survey and Framework for Context-Aware Legal Contract Summarization Using Large Language Models"*  
> Raesa Razeen · ICCIC-2025 · First Author · Peer Reviewed & Published

The Legal Fidelity Score addresses a critical gap in NLP evaluation - standard metrics measure lexical overlap, not legal accuracy. HALO extends this metric into a production-grade, domain-agnostic verification system.

---

## Results

On test contracts with planted hallucinations:

| Metric | Result |
|--------|--------|
| Hallucination Detection Rate | 100% |
| False Positive Rate | 0% |
| Supported Claim Accuracy | 100% |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Claim Extraction | Python · Regex |
| Document Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| LLM Judge | LLaMA 3.3 70B via Groq API |
| Scoring | Custom Legal Fidelity Score |
| UI | Streamlit |

---

## Installation

```bash
git clone https://github.com/raesa-razeen/halo-hallucination-detector.git
cd halo-hallucination-detector
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers groq streamlit pypdf python-dotenv
```

Create a `.env` file:

GROQ_API_KEY=your_groq_api_key_here
Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Paste your document and summary. Click **Run HALO Analysis**.

---

## Roadmap

**v0.1 - Current**
- 3-stage pipeline: ingestion → verification → scoring
- Legal Fidelity Score implementation
- Streamlit web interface

**v0.2 - In Development**
- Legal clause segmentation replacing naive chunking
- Clause importance weighting
- Hallucination taxonomy: Fabrication / Distortion / Exaggeration / Cross-reference error

**v1.0 - Planned**
- Cross-clause dependency graph
- Multi-judge ensemble verification
- Benchmark mode on CUAD dataset
- LegalHalluBench — open benchmark dataset release

---

## Author

**Raesa Razeen**  
PhD Research Scholar · Presidency University, Bangalore  
Published AI Researcher · Legal NLP · LLM Systems  

[LinkedIn](https://www.linkedin.com/in/raesa-razeen-200260209/) · [GitHub](https://github.com/raesa-razeen)

---

## License

MIT License - open for research and commercial use.
