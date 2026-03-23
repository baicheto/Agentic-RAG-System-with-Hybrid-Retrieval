# Agentic RAG System with Hybrid Retrieval

An intelligent question-answering system that combines local document retrieval and live web search to answer user queries. Built on a self-correcting agentic pipeline, the system automatically decides whether to consult a local PDF knowledge base, search the web, or both depending on the nature of the question.

---

## How It Works

Unlike traditional RAG systems that always retrieve from a single source, this pipeline is **agentic**, it makes decisions at every step and corrects itself when something goes wrong.

```
Query comes in
    ↓
Route: local / web / both?
    ↓
Retrieve local docs from PDF
    ↓
Are they relevant?
    No → Rewrite query → Try again (up to 3x)
    Still no → Fall back to web search
    ↓
Generate answer from merged context
    ↓
Is the answer grounded?
    No → Flag with warning 
    Yes → Return clean answer 
```

---

## Key Features

- **Adaptive routing**: the LLM decides whether to use local docs, the web, or both for each query
- **Query rewriting**: if local retrieval fails, the query is automatically rewritten and retried up to 3 times
- **Web fallback**: if local docs are not relevant, the system falls back to live web search automatically
- **Multi-agent web retrieval**: a CrewAI crew of two agents (search + scraping) handles web retrieval
- **Answer grounding verification**: the final answer is checked against the retrieved context to detect hallucinations
- **Hybrid context fusion**: local and web knowledge are merged before final answer generation
- **Streamlit UI**: upload any PDF and ask questions through a simple web interface
- **FAISS caching**: vector database is saved to disk after the first run for faster subsequent queries

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (routing, grading, answering) | Groq - LLaMA 3.1 8B Instant |
| Vector Database | FAISS |
| Embeddings | HuggingFace - all-MiniLM-L6-v2 |
| PDF Loading | LangChain PyPDFLoader |
| Web Search | DuckDuckGo Search (free, no API key) |
| Web Scraping | CrewAI + ScrapeWebsiteTool |
| Multi-Agent Framework | CrewAI |
| UI | Streamlit |

---

## Project Structure

```
agentic-rag/
├── agentic_rag.py       # Core pipeline logic
├── app.py               # Streamlit UI
├── .env                 # API keys (not committed)
└── README.md
```

---

##  Installation

**1. Install dependencies:**
```bash
pip install langchain langchain-community langchain-huggingface crewai crewai-tools python-dotenv pypdf faiss-cpu sentence-transformers duckduckgo-search litellm fastapi streamlit langchain-groq
```

**2. Set up your `.env` file:**
```
GROQ_API_KEY=your_free_groq_key
```
---

## Running the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, upload a PDF, and start asking questions.

> **Note:** Use Chrome or Firefox, Safari is not fully compatible with Streamlit.

---


## Pipeline Components

### Router
Uses the LLM to classify each query as `local`, `web`, or `both` based on a summary of the PDF content. Falls back to `both` if unsure.

### Local Retrieval
Splits the PDF into overlapping chunks, embeds them using a sentence transformer, and stores them in a FAISS vector index. Retrieves the top-5 most semantically similar chunks for each query.

### Relevance Grader
Asks the LLM to judge whether the retrieved chunks are relevant enough to answer the question. Returns `relevant` or `not_relevant`.

### Query Rewriter
If local docs are not relevant, rewrites the query to be more retrieval-friendly and retries — up to 3 times.

### Web Crew (CrewAI)
Two agents working in sequence:
1. **Search Agent** — finds the best web source using DuckDuckGo
2. **Scraper Agent** — extracts and summarizes the page content

### Answer Generator
Merges local and web context and generates a grounded answer using the LLM.

### Grounding Checker
Verifies that the final answer is supported by the retrieved context. Flags unverified answers with a warning instead of silently returning them.
