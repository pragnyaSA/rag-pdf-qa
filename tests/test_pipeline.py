"""Offline tests: embeddings, cross-encoder, LLM and Wikipedia are stubbed so no network,
API key or model download is needed. They exercise the real BM25, RRF, LangGraph routing,
tool-calling, caching and FastAPI code paths.

Run: pytest -q
"""
import re

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import graph, rag, retrieval, tools

PAGES = [
    "Retrieval-Augmented Generation (RAG) enhances large language models with external knowledge.",
    "Naive RAG, Advanced RAG and Modular RAG are the three paradigms of RAG.",
    "RAGAS, ARES and TruLens are tools used to evaluate RAG systems automatically.",
    "Fine-tuning changes model weights while RAG retrieves knowledge at inference time.",
    "Lost in the middle: LLMs tend to focus on the start and end of long contexts.",
]


STOPWORDS = {"what", "is", "the", "of", "are", "and", "a", "to", "in", "on", "with", "while"}


def _words(text):
    return set(re.findall(r"\w+", text.lower())) - STOPWORDS


class FakeCrossEncoder:
    """Score = word overlap, shifted so unrelated pairs are negative (like ms-marco logits)."""
    def predict(self, pairs):
        return [len(_words(q) & _words(d)) * 2.0 - 3.0 for q, d in pairs]


class FakeLLM(RunnableLambda):
    """Echo-style LLM; bind_tools() returns a model that always emits a search_wikipedia call."""
    def __init__(self):
        super().__init__(lambda x: AIMessage(content="LLM_ANSWER"))

    def bind_tools(self, _tools):
        return RunnableLambda(lambda _: AIMessage(
            content="",
            tool_calls=[{"name": "search_wikipedia", "args": {"query": "paris capital"}, "id": "1"}],
        ))


@pytest.fixture(autouse=True)
def stubbed(tmp_path, monkeypatch):
    docs = [
        Document(page_content=t, metadata={"page": i + 1, "chunk_id": i})
        for i, t in enumerate(PAGES)
    ]
    store = Chroma.from_documents(
        docs, DeterministicFakeEmbedding(size=64), persist_directory=str(tmp_path / "db")
    )
    monkeypatch.setattr(rag, "_vectorstore", store)
    monkeypatch.setattr(retrieval, "_bm25", None)
    monkeypatch.setattr(retrieval, "_reranker", FakeCrossEncoder())
    monkeypatch.setattr(graph, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(graph, "_graph", None)
    graph._cache.clear()
    calls = []
    monkeypatch.setattr(
        graph, "wikipedia_search",
        lambda q, limit=2: calls.append(q) or [
            {"title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris", "extract": "Paris is the capital of France."}
        ],
    )
    return calls


def test_bm25_finds_exact_keyword():
    top = retrieval.bm25_search("RAGAS TruLens")[0]
    assert top.metadata["chunk_id"] == 2


def test_rrf_rewards_agreement_between_lists():
    a, b, c = (Document(page_content=x, metadata={"chunk_id": i}) for i, x in enumerate("abc"))
    fused = retrieval.reciprocal_rank_fusion([[a, b], [b, c]])
    assert fused[0].metadata["chunk_id"] == 1  # b is ranked by both lists


def test_rerank_puts_most_relevant_first():
    cands = retrieval.hybrid_search("three paradigms of RAG")
    ranked = retrieval.rerank("three paradigms of RAG", cands, top_n=3)
    assert ranked[0][0].metadata["chunk_id"] == 1
    assert ranked[0][1] >= ranked[1][1]


def test_in_scope_question_uses_document_route(stubbed):
    out = graph.answer_question("What are the three paradigms of RAG?")
    assert out["route"] == "document"
    assert out["answer"] == "LLM_ANSWER"
    assert all(s["type"] == "document" and s["page"] for s in out["sources"])
    # every matched chunk is part of the context that was sent to the LLM
    assert {s["chunk_id"] for s in out["sources"]} <= set(out["context_chunk_ids"])
    assert stubbed == []  # web tool never called


def test_out_of_scope_question_uses_web_tool_calling(stubbed):
    out = graph.answer_question("What is the capital of France?")
    assert out["route"] == "web"
    assert stubbed == ["paris capital"]  # query chosen by the LLM's tool call
    assert out["sources"][0]["type"] == "web" and out["sources"][0]["url"].startswith("https://")
    assert out["context_chunk_ids"] == []


def test_web_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setattr(graph, "WEB_FALLBACK", False)
    assert graph.answer_question("What is the capital of France?")["route"] == "document"


def test_no_web_results_returns_not_found(monkeypatch):
    monkeypatch.setattr(graph, "wikipedia_search", lambda q, limit=2: [])
    out = graph.answer_question("What is the capital of France?")
    assert out["answer"] == graph.NOT_FOUND


def test_cache_returns_same_object_and_skips_pipeline(monkeypatch):
    first = graph.answer_question("What are the three paradigms of RAG?")
    monkeypatch.setattr(graph, "_graph", None)
    monkeypatch.setattr(graph, "build_graph", lambda: pytest.fail("pipeline re-ran"))
    assert graph.answer_question("  WHAT ARE THE THREE PARADIGMS OF RAG?  ") is first


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(graph, "CACHE_SIZE", 2)
    for q in ("what are rags", "three paradigms rag", "tools evaluate rag"):
        graph.answer_question(q)
    assert len(graph._cache) == 2


def test_wikipedia_tool_degrades_gracefully(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(tools.requests, "get", boom)
    assert tools.wikipedia_search("anything") == []


def test_ask_endpoint_contract():
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "What are the three paradigms of RAG?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"answer", "route", "sources", "context_chunk_ids"}
        assert "title" not in body["sources"][0]  # null fields are omitted
        assert body["sources"][0]["page"] >= 1
        assert client.post("/ask", json={"question": "hi"}).status_code == 422