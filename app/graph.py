"""LangGraph workflow for the PDF RAG pipeline.

    retrieve (hybrid: BM25 + dense, RRF)
        -> rerank (cross-encoder)
        -> route ──(confident)──> build_context (neighbor expansion) ──┐
              └──(low score)───> web_search (LLM tool calling)  ───────┴─> generate -> END

If the best reranked chunk scores below RERANK_MIN_SCORE the PDF probably does not
contain the answer, so the LLM is given a Wikipedia search tool (tool calling) and the
answer is clearly labelled as coming from the web instead of the PDF.
"""
import os
from collections import OrderedDict
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from app.rag import PROMPT, TOP_K, expand_with_neighbors, get_llm, _format_docs
from app.retrieval import hybrid_search, rerank
from app.tools import search_wikipedia, wikipedia_search

# Cross-encoder (ms-marco) logits: relevant chunks are usually > 0, unrelated ones strongly negative.
# Tune with `python evaluate.py` (it prints the top score for in-scope and out-of-scope questions).
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", -2.0))
WEB_FALLBACK = os.getenv("WEB_FALLBACK", "true").lower() == "true"
CACHE_SIZE = int(os.getenv("CACHE_SIZE", 256))

NOT_FOUND = "I could not find this in the document."

WEB_PROMPT = ChatPromptTemplate.from_template(
    """The user's question is NOT covered by the PDF document, so Wikipedia was searched instead.
Use ONLY the Wikipedia excerpts below. Begin your answer with "Not found in the PDF - based on Wikipedia:".
Answer concisely. If the excerpts do not answer the question, say "I could not find this in the document."

Wikipedia excerpts:
{context}

Question: {question}

Answer:"""
)

TOOL_PROMPT = (
    "You are the web-lookup step of a PDF question-answering system. The PDF does not cover "
    "the user's question. Call the search_wikipedia tool with a short, focused query.\n\n"
    "Question: {question}"
)


class RAGState(TypedDict, total=False):
    question: str
    candidates: list[Document]
    ranked: list  # [(Document, score | None)]
    top_score: Optional[float]
    route: str  # "document" | "web"
    context: str
    sources: list[dict]
    prompt_kind: str
    answer: str


# ---------------------------------------------------------------- nodes
def retrieve_node(state: RAGState) -> dict:
    return {"candidates": hybrid_search(state["question"])}


def rerank_node(state: RAGState) -> dict:
    ranked = rerank(state["question"], state["candidates"], top_n=TOP_K)
    top = ranked[0][1] if ranked else None
    return {"ranked": ranked, "top_score": top}


def route_after_rerank(state: RAGState) -> str:
    """Conditional edge: go to the web only when the reranker is clearly unconvinced."""
    top = state.get("top_score")
    if WEB_FALLBACK and top is not None and top < RERANK_MIN_SCORE:
        return "web_search"
    return "build_context"


def build_context_node(state: RAGState) -> dict:
    ranked = state["ranked"]
    docs = expand_with_neighbors(ranked)
    sources = [
        {
            "type": "document",
            "page": d.metadata["page"],
            "chunk_id": d.metadata["chunk_id"],
            "score": round(score, 4) if score is not None else None,  # rerank score: higher = better
            "snippet": d.page_content[:200].replace("\n", " ") + "...",
        }
        for d, score in ranked
    ]
    return {"route": "document", "context": _format_docs(docs), "sources": sources, "prompt_kind": "document"}


def web_search_node(state: RAGState) -> dict:
    """Tool calling: the LLM decides the search query by calling `search_wikipedia`."""
    question = state["question"]
    queries = []
    try:
        llm_with_tools = get_llm().bind_tools([search_wikipedia])
        msg = llm_with_tools.invoke([HumanMessage(TOOL_PROMPT.format(question=question))])
        queries = [
            tc["args"].get("query", question)
            for tc in (msg.tool_calls or [])
            if tc["name"] == "search_wikipedia"
        ]
    except Exception as e:
        print(f"[warn] tool calling failed ({e}); searching with the raw question")
    results = []
    for q in queries[:2] or [question]:  # fall back to the raw question if the LLM made no tool call
        results.extend(wikipedia_search(q))
    seen, unique = set(), []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    context = "\n\n".join(f"[{r['title']}] {r['extract']}" for r in unique)
    sources = [
        {"type": "web", "title": r["title"], "url": r["url"], "snippet": r["extract"][:200] + "..."}
        for r in unique
    ]
    return {"route": "web", "context": context, "sources": sources, "prompt_kind": "web"}


def generate_node(state: RAGState) -> dict:
    if not state["context"].strip():
        return {"answer": NOT_FOUND}
    prompt = WEB_PROMPT if state["prompt_kind"] == "web" else PROMPT
    chain = prompt | get_llm() | StrOutputParser()
    return {"answer": chain.invoke({"context": state["context"], "question": state["question"]})}


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("build_context", build_context_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", route_after_rerank, ["build_context", "web_search"])
    g.add_edge("build_context", "generate")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = None
_cache: "OrderedDict[str, dict]" = OrderedDict()


def _answer_uncached(question: str) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    # run_name/tags show up in LangSmith when LANGSMITH_TRACING=true
    out = _graph.invoke(
        {"question": question},
        config={"run_name": "pdf_rag_ask", "tags": ["pdf-rag", "langgraph"]},
    )
    return {"answer": out["answer"], "route": out["route"], "sources": out["sources"]}


def answer_question(question: str) -> dict:
    """Run the graph, with a bounded LRU cache for repeated questions."""
    key = question.strip().lower()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    result = _answer_uncached(question)
    _cache[key] = result
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return result
