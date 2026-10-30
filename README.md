# PDF RAG Question Answering (LangChain + LangGraph + FastAPI)

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
- LangGraph (workflow), LangSmith (optional tracing)
- PyPDF (PDF text extraction)
- sentence-transformers `all-MiniLM-L6-v2` (embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)
- rank-bm25 (keyword search for hybrid retrieval)
- ChromaDB (vector database)
- Groq API with `openai/gpt-oss-20b` (LLM)
- Wikipedia API (web tool, no API key needed)
- pytest (offline tests)

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
    { "type": "document", "page": 16, "chunk_id": 135, "score": 9.0383, "snippet": "..." }
  ],
  "context_chunk_ids": [20, 21, 22, 23, 24, 25, 26, 135]
}
```
Response (question not covered by the PDF, answered through the Wikipedia tool):
```json
{
  "answer": "Not found in the PDF - based on Wikipedia: Argentina won the 2022 FIFA World Cup.",
  "route": "web",
  "sources": [
    { "type": "web", "title": "2022 FIFA World Cup", "url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup", "snippet": "..." }
  ],
  "context_chunk_ids": []
}
```
- `route`: `document` (answered from the PDF) or `web` (the PDF had no confident match, so the web tool was used)
- `sources`: the best-matching chunks (or web pages), best first
- `page`: page number in the PDF (1-based)
- `chunk_id`: index of the chunk in the vector store
- `score`: cross-encoder rerank score (higher = more relevant)
- `snippet`: first 200 characters of the chunk or web extract
- `context_chunk_ids`: every chunk that was sent to the LLM, including the neighbor chunks added around the top matches

### `GET /health`
Returns `{"status": "ok"}`.

Example call:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What are the three paradigms of RAG?"}'
```

## Example Questions and Responses

**1. What are the three paradigms of RAG?** (route: `document`)
> The paper identifies three RAG paradigms:
> - **Naive RAG** - the earliest, basic retrieval-generation pipeline.
> - **Advanced RAG** - adds pre-retrieval and post-retrieval optimizations (e.g., refined indexing, query rewriting, reranking).
> - **Modular RAG** - a flexible architecture with interchangeable modules (e.g., iterative/adaptive retrieval, fine-tuned retrievers).

Sources (page / chunk / rerank score): 16 / 135 / 9.04, 4 / 23 / 8.48, 2 / 12 / 7.32, 2 / 14 / 7.17, 1 / 7 / 6.70, 1 / 2 / 6.52

**2. What is Retrieval-Augmented Generation?** (route: `document`)
> RAG is a paradigm that enhances large language models by retrieving relevant document chunks from
> external databases and incorporating that knowledge into generation. It improves accuracy, reduces
> hallucinations, and allows continuous knowledge updates.

**3. What are the three quality scores used to evaluate RAG?** (route: `document`)
> Context Relevance (precision of the retrieved context), Answer Faithfulness (the answer stays true to
> the retrieved context) and Answer Relevance (the answer addresses the question).

**4. How is RAG different from fine-tuning?** (route: `document`)
> RAG adds an external knowledge source at inference time, giving real-time updates and high
> interpretability at the cost of higher latency. Fine-tuning changes the model's weights, is static,
> needs retraining to update and suits replicating specific styles or formats. Across knowledge-intensive
> benchmarks, RAG outperforms unsupervised fine-tuning.

**5. Who won the 2022 FIFA World Cup?** (not in the PDF, route: `web`)
> Not found in the PDF - based on Wikipedia: Argentina won the 2022 FIFA World Cup.

Sources: Wikipedia "2022 FIFA World Cup" and "2026 FIFA World Cup knockout stage" (URLs returned in `sources`).
The best PDF chunk scored about -10.9, far below the threshold, so the graph routed to the web tool and the LLM
chose the search query by calling `search_wikipedia`. Set `WEB_FALLBACK=false` to restore a strict PDF-only
refusal ("I could not find this in the document.").

## Model Choices and Reasons
- **Embeddings: `all-MiniLM-L6-v2` (normalized)** - small, fast, free and runs locally on CPU.
  Normalizing the vectors keeps distance scores consistent.
- **Vector DB: ChromaDB** - simple, persists on disk, needs no separate server and integrates directly
  with LangChain.
- **Chunking: recursive, 700 characters / 150 overlap** - smaller, more focused chunks than the common
  1000/200 default, which improved retrieval of section-level questions. The References section is
  removed before indexing so citation lists do not pollute the results.
- **Hybrid search (BM25 + dense, RRF):** dense embeddings capture meaning but can miss exact terms such as
  "RAGAS" or "HyDE"; BM25 does the opposite. Reciprocal Rank Fusion merges both rankings without needing to
  normalise their different score scales.
- **Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`** - reads question and chunk together, so it is more precise
  than bi-encoder similarity in principle. It only scores a shortlist of about 15 chunks, so latency stays low.
  It also provides the confidence score used for routing.
- **Top-k = 6 plus neighbor expansion** - after reranking, the chunks around the top 3 matches
  are added to the context so a matched heading comes with its body text.
- **LLM: `openai/gpt-oss-20b` via Groq** - free tier, fast, and supports tool calling. Groq retired Llama 3.1 8B
  for free accounts, so the model is configurable through `LLM_MODEL` in `.env`. Temperature is 0 for factual answers.
- **Prompt** restricts the LLM to the retrieved context and asks for a refusal when the answer is not
  there, which reduces hallucination.

## Improvements Made During Development
- **Section-level questions.** "How is RAG different from fine-tuning?" initially answered "I could not find this in
  the document", although the answer is on page 5. Fixes: removed the bibliography from the index, reduced chunks to
  700/150, normalized the embeddings, and added neighbor-chunk expansion.
- **Keyword misses in dense search.** Added BM25 and fused it with the dense ranking (RRF). MRR rose from 0.74 to 0.91.
- **Web answers missing the facts.** The first version of the Wikipedia tool truncated each article to 900 characters,
  which cut off the sentence naming the winner, so the answer to the World Cup question said it could not find it.
  Returning the full article introduction (capped at 2500 characters) fixed it.
- **Transparency.** `context_chunk_ids` now lists every chunk the LLM saw, not just the top matches.

## Optional Features Implemented
| Feature | Where | How it works |
|---|---|---|
| LangGraph workflow | `app/graph.py` | 5-node graph (retrieve, rerank, build_context, web_search, generate) with a conditional edge that routes on reranker confidence |
| Hybrid search | `app/retrieval.py` | BM25 + Chroma dense search fused with Reciprocal Rank Fusion |
| Reranking | `app/retrieval.py` | Cross-encoder rescoring of the fused shortlist; falls back to fused order if the model cannot load |
| Tool calling | `app/graph.py` | The LLM is bound to the `search_wikipedia` tool and chooses the search query itself |
| Web API tool usage | `app/tools.py` | Wikipedia public API (no key), with timeouts and graceful failure |
| Caching | `app/graph.py` | LRU cache (size `CACHE_SIZE`, default 256) for repeated questions, saved to `answer_cache.json` so it survives restarts; `python ingest.py` clears it because cached answers belong to the old index |
| Evaluation metrics | `evaluate.py` | Hit Rate@k and MRR for dense vs hybrid vs hybrid + rerank, a routing check, and an optional answer-level keyword check (`--answers`) |
| LangSmith tracing | `.env` | Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`; each request appears as a `pdf_rag_ask` run with a span per graph node |

### Evaluation
```bash
python evaluate.py             # retrieval: dense-only vs hybrid vs hybrid + rerank on 9 labelled questions, plus routing check
python evaluate.py --answers   # also checks the generated answers contain the expected facts (calls the LLM)
pytest -q                      # 13 offline tests with stubbed models (no API key or downloads needed)
```

Results on 9 hand-labelled questions (a hit means one of the top-6 chunks comes from a page that contains the answer):

| Method | Hit Rate@6 | MRR |
|---|---|---|
| Dense only | 1.00 | 0.74 |
| Hybrid (BM25 + dense, RRF) | 1.00 | 0.91 |
| Hybrid + rerank | 1.00 | 0.81 |

Answer check (`python evaluate.py --answers`): 9/9 in-scope answers pass (from the PDF route, with the expected facts), and 2/2 out-of-scope questions are sent to the web tool.

- Hybrid search improved MRR from 0.74 to 0.91, so the first relevant chunk is usually ranked first or second.
- The cross-encoder did **not** improve MRR on this set (0.91 to 0.81). Matching is at page level, which is coarse,
  and the ms-marco model is trained on web passages, so it sometimes prefers a chunk from a neighboring page.
- The reranker's main value here is **routing**: the lowest top score for an in-scope question was 0.17 and the
  highest for an out-of-scope question was -10.86, so `RERANK_MIN_SCORE=-2.0` separated them cleanly
  (9/9 sent to the PDF, 2/2 sent to the web).

- The reranker is a second model that adds some latency to each uncached question (it scores about 15 chunks);
  repeated questions skip it through the cache.
- `--answers` adds an end-to-end check: each in-scope answer must come from the PDF route and mention the expected
  facts (for example "Naive", "Advanced" and "Modular" for the paradigms question), and out-of-scope questions must
  be sent to the web tool. It is a coarse keyword check, not a replacement for human review.

This is a small sanity check, not a full benchmark.

## Limitations
- **Scope:** one PDF, indexed at startup (to use another, change `PDF_PATH` and run `python ingest.py`); there is no
  upload endpoint and no conversation memory, so every question is answered independently.
- **PDF parsing:** tables, figures and equations are poorly extracted by text-only parsing.
- **Web fallback:** answers rely on Wikipedia's top search hits and introduction text, so they are shallow and can
  include loosely related articles (the World Cup answer also returned a 2026 article).
- **LLM:** the small model occasionally adds minor inferences that are not stated in the context.
- **Evaluation size:** 9 in-scope and 2 out-of-scope questions with page-level matching and keyword answer checks.
  The routing threshold (`RERANK_MIN_SCORE`) separated them with a wide gap, but questions that are only loosely
  related to the PDF could still land near it.

## Author
Pragnya Amminabhavi