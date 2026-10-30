# PDF RAG Question Answering (LangChain + FastAPI)

## Project Overview
A Retrieval-Augmented Generation app that answers questions from a PDF
(the survey *"Retrieval-Augmented Generation for Large Language Models: A Survey"*).

Pipeline: PDF loading -> text extraction -> chunking -> embeddings -> Chroma vector DB
-> similarity search -> neighbor-chunk expansion -> LLM answer grounded in the retrieved context
-> answer returned with page and chunk references through a FastAPI `POST /ask` endpoint.

```
PDF -> PyPDFLoader -> strip References -> RecursiveCharacterTextSplitter -> HF Embeddings (MiniLM) -> ChromaDB
Question -> embed -> top-k similarity search -> add neighboring chunks -> prompt(context + question) -> Groq LLM -> answer + sources
```

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- LangChain (document loaders, text splitters, prompts, chains)
- PyPDF (PDF text extraction)
- sentence-transformers `all-MiniLM-L6-v2` (embeddings)
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
The first run downloads the embedding model (about 90 MB).

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
Response:
```json
{
  "answer": "...",
  "sources": [
    { "page": 1, "chunk_id": 0, "score": 0.74, "snippet": "..." }
  ]
}
```
- `page`: page number in the PDF (1-based)
- `chunk_id`: index of the chunk in the vector store
- `score`: Chroma distance (lower = more similar)
- `snippet`: first 200 characters of the chunk

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

**5. Who won the 2022 FIFA World Cup?** (out of scope)
> I could not find this in the document.

The retrieved chunks have high distance scores (about 1.7-1.8, versus about 0.5-0.7 for in-scope
questions), so the model correctly refuses instead of answering from outside knowledge.

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
- **Prompt** restricts the LLM to the retrieved context and asks for a refusal when the answer is not
  there, which reduces hallucination.

## Improvements Made During Development
- Problem: "How is RAG different from fine-tuning?" initially answered "I could not find this in the
  document", although the answer is on page 5.
- Fixes: removed the bibliography from the index, reduced chunks to 700/150, normalized the embeddings,
  and added neighbor-chunk expansion. The answer then covered the full comparison from the survey.

## Limitations
- Single PDF, indexed at startup; there is no upload endpoint.
- Tables, figures and equations are poorly extracted by text-only parsing.
- Dense retrieval only (no hybrid/BM25 search and no reranking), so some section-level questions can
  still retrieve loosely related chunks.
- The `sources` list shows the top-k matched chunks; the neighbor chunks that also fed the answer
  are not listed.
- The small LLM occasionally adds minor inferences that are not stated in the context.
- No conversation memory; each question is independent.
- No automated evaluation metrics yet.

## Author
Pragnya Amminabhavi