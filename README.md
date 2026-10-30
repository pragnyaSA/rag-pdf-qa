# PDF RAG Question Answering (LangChain + FastAPI)

## Project Overview
A Retrieval-Augmented Generation app that answers questions from a PDF
(the survey *"Retrieval-Augmented Generation for Large Language Models: A Survey"*).

Pipeline: PDF loading -> text extraction -> chunking -> embeddings -> Chroma vector DB
-> hybrid search (BM25 + dense) -> cross-encoder reranking -> neighbor-chunk expansion -> LLM answer
grounded in the retrieved context -> answer returned with page and chunk references through a
FastAPI `POST /ask` endpoint. The query flow is orchestrated as a **LangGraph** workflow.

```
Indexing: PDF -> PyPDFLoader -> strip References -> RecursiveCharacterTextSplitter -> HF Embeddings (MiniLM) -> ChromaDB

Query (LangGraph):
  retrieve (BM25 + dense, fused with RRF)
      -> rerank (cross-encoder)
      -> route --[top score >= threshold]--> build_context (neighbor chunks) --+
             \--[top score <  threshold]--> web_search (LLM tool calling,      |
                                            Wikipedia API)  ------------------+--> generate -> answer + sources
```

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- LangChain (document loaders, text splitters, prompts, chains)
- PyPDF (PDF text extraction)
- LangGraph (workflow), LangSmith (optional tracing)
- sentence-transformers `all-MiniLM-L6-v2` (embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)
- rank-bm25 (keyword search for hybrid retrieval)
- Wikipedia API (web tool, no API key needed)
- ChromaDB (vector database)
- Groq API with `openai/gpt-oss-20b` (LLM)

## Setup
```bash
git clone https://github.com/pragnyaSA/rag-pdf-qa.git
cd rag-pdf-qa
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
# Windows: copy .env.example .env     Mac/Linux: cp .env.example .env
```
Open `.env` and set your `GROQ_API_KEY` (free key at console.groq.com).
The PDF is included at `data/rag_survey.pdf`; change `PDF_PATH` in `.env` to use another file.
The first run downloads the embedding model (about 90 MB) and the reranker (about 90 MB).

## Run the FastAPI Server
```bash
python ingest.py                          # builds the vector index (it is also built automatically on first start)
python -m uvicorn app.main:app --reload
```
Open the Swagger UI at http://127.0.0.1:8000/docs

If you change the embedding model or the chunk settings, delete the `chroma_db` folder and run
`python ingest.py` again, because vectors from different models cannot be mixed.

## API Endpoint
### `POST /ask`
Request:
```json
{ "question": "What is Retrieval-Augmented Generation?" }
```
Response (answer found in the PDF):
```json
{
  "answer": "...",
  "route": "document",
  "sources": [
    { "type": "document", "page": 1, "chunk_id": 0, "score": 7.31, "snippet": "..." }
  ]
}
```
Response (question not covered by the PDF, answered through the Wikipedia tool):
```json
{
  "answer": "Not found in the PDF - based on Wikipedia: ...",
  "route": "web",
  "sources": [
    { "type": "web", "title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris", "snippet": "..." }
  ]
}
```
- `route`: `document` (answered from the PDF) or `web` (PDF had no confident match, web tool used)
- `page`: page number in the PDF (1-based)
- `chunk_id`: index of the chunk in the vector store
- `score`: cross-encoder rerank score (higher = more relevant)
- `snippet`: first 200 characters of the chunk or web extract

### `GET /health`
Returns `{"status": "ok"}`.

Example call:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What are the three paradigms of RAG?"}'
```

## Example Questions and Responses

**1. What is Retrieval-Augmented Generation?**
> RAG is a paradigm that enhances large language models by retrieving relevant document chunks from
> external databases and incorporating that knowledge into generation. It improves accuracy, reduces
> hallucinations, and allows continuous knowledge updates.

Sources: page 1 (chunks 0, 3), page 3 (chunk 20), page 11 (chunks 92, 93)

**2. What are the three paradigms of RAG?**
> Naive RAG, Advanced RAG and Modular RAG.

Sources: page 2 (chunks 7, 10), page 1 (chunk 5)

**3. What are the three quality scores used to evaluate RAG?**
> Context Relevance (precision of the retrieved context), Answer Faithfulness (the answer stays true to
> the retrieved context) and Answer Relevance (the answer addresses the question).

Sources: page 12 (chunks 104, 106, 107), page 14 (chunks 115-117)

**4. How is RAG different from fine-tuning?**
> RAG adds an external knowledge source at inference time, giving real-time updates and high
> interpretability at the cost of higher latency. Fine-tuning changes the model's weights, is static,
> needs retraining to update and suits replicating specific styles or formats. Across knowledge-intensive
> benchmarks, RAG outperforms unsupervised fine-tuning.

Sources: page 5 (section "D. RAG vs Fine-tuning"), pages 1-2, page 14

**5. Who won the 2022 FIFA World Cup?** (out of scope for the PDF)
> Before the optional features, the app answered "I could not find this in the document." Now the
> reranker's top score falls below `RERANK_MIN_SCORE`, the graph routes to the web tool, the LLM calls
> `search_wikipedia`, and the answer is labelled "Not found in the PDF - based on Wikipedia" with the
> article URL in `sources` (`"route": "web"`). Set `WEB_FALLBACK=false` to restore the strict
> PDF-only refusal.

## Model Choices and Reasons
- **Embeddings: `all-MiniLM-L6-v2` (normalized)** - small, fast, free and runs locally on CPU.
  Normalizing the vectors keeps distance scores consistent.
- **Vector DB: ChromaDB** - simple, persists on disk, needs no separate server and integrates directly
  with LangChain.
- **Chunking: recursive, 700 characters / 150 overlap** - smaller, more focused chunks than the common
  1000/200 default, which improved retrieval of section-level questions. The References section is
  removed before indexing so citation lists do not pollute the results.
- **Top-k = 6 plus neighbor expansion** - after similarity search, the chunks around the top 3 matches
  are added to the context so a matched heading comes with its body text.
- **LLM: `openai/gpt-oss-20b` via Groq** - free tier and fast. Groq retired Llama 3.1 8B for free
  accounts, so the model is configurable through `LLM_MODEL` in `.env`. Temperature is 0 for factual answers.
- **Hybrid search (BM25 + dense, RRF):** dense embeddings capture meaning but can miss exact terms such as
  "RAGAS" or "HyDE"; BM25 does the opposite. Reciprocal Rank Fusion merges both rankings without needing to
  normalise their different score scales.
- **Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`** - reads question and chunk together, so it is more precise
  than bi-encoder similarity. It only scores a shortlist of about 15 chunks, so latency stays low. It also provides
  the confidence score used for routing.
- **Prompt** restricts the LLM to the retrieved context and asks for a refusal when the answer is not
  there, which reduces hallucination.

## Improvements Made During Development
- Problem: "How is RAG different from fine-tuning?" initially answered "I could not find this in the
  document", although the answer is on page 5.
- Fixes: removed the bibliography from the index, reduced chunks to 700/150, normalized the embeddings,
  and added neighbor-chunk expansion. The answer then covered the full comparison from the survey.

## Optional Features Implemented
| Feature | Where | How it works |
|---|---|---|
| LangGraph workflow | `app/graph.py` | 5-node graph (retrieve, rerank, build_context, web_search, generate) with a conditional edge that routes on reranker confidence |
| Hybrid search | `app/retrieval.py` | BM25 + Chroma dense search fused with Reciprocal Rank Fusion |
| Reranking | `app/retrieval.py` | Cross-encoder rescoring of the fused shortlist; falls back to fused order if the model cannot load |
| Tool calling | `app/graph.py` | The LLM is bound to the `search_wikipedia` tool and chooses the search query itself |
| Web API tool usage | `app/tools.py` | Wikipedia public API (no key), with timeouts and graceful failure |
| Caching | `app/graph.py` | In-memory LRU cache (size `CACHE_SIZE`, default 256) for repeated questions |
| Evaluation metrics | `evaluate.py` | Hit Rate@k and MRR for dense vs hybrid vs hybrid + rerank, plus a routing check |
| LangSmith tracing | `.env` | Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`; each request appears as a `pdf_rag_ask` run with a span per graph node |

### Evaluation
```bash
python evaluate.py     # compares dense-only vs hybrid vs hybrid + rerank on 9 labelled questions
pytest -q              # offline tests with stubbed models (no API key or downloads needed)
```
Baseline measured before the optional features (dense search only, 6 questions): Hit Rate@6 = 1.00, MRR = 0.72.
The script now reports the same two metrics for all three retrieval modes on a 9-question set, so the effect of
hybrid search and reranking can be read directly from its output. It is a small, page-level sanity check rather
than a full benchmark.

`evaluate.py` also prints the top rerank score for in-scope and out-of-scope questions. If questions are routed to
the wrong place, set `RERANK_MIN_SCORE` in `.env` to a value between those two scores.

## Limitations
- Single PDF, indexed at startup; there is no upload endpoint.
- Tables, figures and equations are poorly extracted by text-only parsing.
- The web fallback depends on `RERANK_MIN_SCORE`, which is a heuristic threshold that should be calibrated with
  `python evaluate.py`. A wrong value sends in-scope questions to the web or keeps out-of-scope ones in the PDF route.
- Web answers rely on Wikipedia's top search hits and intro extracts only, so they are shallow and can be off-topic.
- The reranker adds a second model to load and some latency per uncached question.
- The cache is in-process memory: it is lost on restart and not shared between workers.
- The `sources` list shows the top-k matched chunks; the neighbor chunks that also fed the answer
  are not listed.
- The small LLM occasionally adds minor inferences that are not stated in the context.
- No conversation memory; each question is independent.
- Evaluation covers retrieval only (9 questions, page-level matching); answer quality is not scored automatically.

## Author
Pragnya Amminabhavi