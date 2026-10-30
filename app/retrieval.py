"""Hybrid retrieval: BM25 (keyword) + dense (Chroma) fused with Reciprocal Rank Fusion,
followed by cross-encoder reranking.

Why: dense embeddings are good at meaning but can miss exact terms (e.g. "RAGAS", "HyDE");
BM25 is the opposite. Fusing both and reranking the shortlist with a cross-encoder improves
precision at the top of the list.
"""
import os
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.rag import get_vectorstore, TOP_K

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DENSE_K = int(os.getenv("DENSE_K", 12))   # candidates taken from the vector search
BM25_K = int(os.getenv("BM25_K", 12))     # candidates taken from keyword search
FUSED_K = int(os.getenv("FUSED_K", 15))   # shortlist size passed to the reranker
RRF_K = 60                                 # standard RRF constant

_chunks: list[Document] | None = None
_bm25: BM25Okapi | None = None
_reranker = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load_bm25():
    """Build the BM25 index once from the chunks already stored in Chroma (no re-ingest needed)."""
    global _chunks, _bm25
    if _bm25 is None:
        got = get_vectorstore().get()
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(got["documents"], got["metadatas"])
        ]
        _chunks = sorted(docs, key=lambda d: d.metadata["chunk_id"])
        _bm25 = BM25Okapi([_tokenize(d.page_content) for d in _chunks])
    return _chunks, _bm25


def bm25_search(question: str, k: int = BM25_K) -> list[Document]:
    chunks, bm25 = _load_bm25()
    scores = bm25.get_scores(_tokenize(question))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top if scores[i] > 0]


def dense_search(question: str, k: int = DENSE_K) -> list[Document]:
    return get_vectorstore().similarity_search(question, k=k)


def reciprocal_rank_fusion(rank_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """Merge several ranked lists: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    by_id: dict[int, Document] = {}
    for ranked in rank_lists:
        for rank, doc in enumerate(ranked, start=1):
            cid = doc.metadata["chunk_id"]
            by_id[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return [by_id[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


def hybrid_search(question: str, k: int = FUSED_K) -> list[Document]:
    """BM25 + dense search fused with RRF; returns the top-k shortlist."""
    return reciprocal_rank_fusion([dense_search(question), bm25_search(question)])[:k]


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # lazy: heavy import
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(question: str, docs: list[Document], top_n: int = TOP_K):
    """Score (question, chunk) pairs with a cross-encoder.

    Returns [(Document, score)] sorted best-first. Higher score = more relevant.
    If the reranker cannot be loaded (e.g. offline), falls back to the fused order
    with score=None so the API keeps working.
    """
    if not docs:
        return []
    try:
        scores = _get_reranker().predict([(question, d.page_content) for d in docs])
    except Exception as e:  # pragma: no cover - defensive fallback
        print(f"[warn] reranker unavailable ({e}); using fused order")
        return [(d, None) for d in docs[:top_n]]
    ranked = sorted(zip(docs, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
