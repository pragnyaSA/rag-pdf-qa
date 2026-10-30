"""Retrieval evaluation: compares dense-only, hybrid (BM25 + dense) and hybrid + rerank.

Run: python evaluate.py            (retrieval metrics + routing check, no API key needed)
     python evaluate.py --answers  (also checks the generated answers; calls the LLM about 11 times)

Metrics (on a small hand-labelled set; a 'hit' = any top-k chunk comes from an expected page):
  - Hit Rate@k : fraction of questions with at least one relevant chunk in the top k
  - MRR        : mean of 1 / rank of the first relevant chunk

It also prints the top rerank score for out-of-scope questions so RERANK_MIN_SCORE
(the threshold that sends a question to the web tool) can be calibrated.
"""
import re
import sys

from app.graph import RERANK_MIN_SCORE
from app.rag import TOP_K, get_vectorstore
from app.retrieval import dense_search, hybrid_search, rerank

# (question, pages where the answer is found in the PDF)
EVAL_SET = [
    ("What is Retrieval-Augmented Generation?", [1]),
    ("What are the three paradigms of RAG?", [1, 2]),
    ("How is RAG different from fine-tuning?", [5]),
    ("What are the three quality scores used to evaluate RAG?", [12]),
    ("Which tools are used to evaluate RAG?", [14]),
    ("What is the lost in the middle problem?", [10]),
    ("What is HyDE?", [5, 8, 9]),
    ("How does Self-RAG decide when to retrieve?", [11, 12]),
    ("What chunk sizes are commonly used for splitting documents?", [8]),
]

# Questions the PDF cannot answer: these should be routed to the web tool.
OUT_OF_SCOPE = [
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of France?",
]

# Facts a correct answer must mention. A tuple means "any one of these is fine".
# This is a coarse, deterministic check, not a full answer-quality benchmark.
ANSWER_KEYWORDS = {
    "What is Retrieval-Augmented Generation?": ["retriev", "external"],
    "What are the three paradigms of RAG?": ["naive", "advanced", "modular"],
    "How is RAG different from fine-tuning?": ["fine-tun", "retriev"],
    "What are the three quality scores used to evaluate RAG?": ["context relevance", "faithfulness", "answer relevance"],
    "Which tools are used to evaluate RAG?": ["ragas", "ares", "trulens"],
    "What is the lost in the middle problem?": ["middle", "context"],
    "What is HyDE?": ["hypothetical"],
    "How does Self-RAG decide when to retrieve?": ["token"],
    "What chunk sizes are commonly used for splitting documents?": [("100", "256", "512")],
}

MODES = {
    "dense only": lambda q: [d for d in dense_search(q, k=TOP_K)],
    "hybrid (BM25 + dense, RRF)": lambda q: hybrid_search(q)[:TOP_K],
    "hybrid + rerank": lambda q: [d for d, _ in rerank(q, hybrid_search(q), top_n=TOP_K)],
}


def evaluate(retrieve, verbose=False):
    hits, rr_sum = 0, 0.0
    for question, expected in EVAL_SET:
        pages = [d.metadata["page"] for d in retrieve(question)]
        rank = next((i + 1 for i, p in enumerate(pages) if p in expected), None)
        if rank:
            hits += 1
            rr_sum += 1 / rank
        if verbose:
            print(f"  {'HIT ' if rank else 'MISS'} rank={rank} pages={pages} | {question}")
    n = len(EVAL_SET)
    return hits / n, rr_sum / n


def _normalize(text: str) -> str:
    """Lowercase and turn Unicode hyphens/dashes (e.g. U+2011) into a plain '-'."""
    text = text.lower().replace("\u00ad", "")  # soft hyphen: invisible, just drop it
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text)


def check_answers():
    """End-to-end check: in-scope answers must mention the expected facts, out-of-scope ones must use the web."""
    from app.graph import answer_question  # imported here so the retrieval-only run needs no API key

    passed = 0
    print("\nAnswer check (in-scope questions must contain the expected facts):")
    for question, _ in EVAL_SET:
        out = answer_question(question)
        text = _normalize(out["answer"])
        missing = [
            g for g in ANSWER_KEYWORDS[question]
            if not any(alt in text for alt in (g if isinstance(g, tuple) else (g,)))
        ]
        ok = out["route"] == "document" and not missing
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'} route={out['route']} missing={missing} | {question}")
        if not ok:
            print(f"       answer: {out['answer'][:300]!r}")
    print(f"  In-scope answers passing: {passed}/{len(EVAL_SET)}")

    routed = 0
    for question in OUT_OF_SCOPE:
        out = answer_question(question)
        routed += out["route"] == "web"
        print(f"  {'PASS' if out['route'] == 'web' else 'FAIL'} route={out['route']} | {question}")
    print(f"  Out-of-scope questions sent to the web tool: {routed}/{len(OUT_OF_SCOPE)}")


def main():
    get_vectorstore()
    print(f"{len(EVAL_SET)} questions, k={TOP_K}\n")
    print(f"{'method':32s} Hit@{TOP_K}   MRR")
    for name, retrieve in MODES.items():
        hit, mrr = evaluate(retrieve)
        print(f"{name:32s} {hit:5.2f}   {mrr:.2f}")

    print("\nPer-question detail for the full pipeline (hybrid + rerank):")
    evaluate(MODES["hybrid + rerank"], verbose=True)

    print(f"\nRouting check (threshold RERANK_MIN_SCORE = {RERANK_MIN_SCORE}; below it -> web tool)")
    in_scope = [rerank(q, hybrid_search(q), top_n=1)[0][1] for q, _ in EVAL_SET]
    out_scope = [rerank(q, hybrid_search(q), top_n=1)[0][1] for q in OUT_OF_SCOPE]
    print(f"  in-scope top score     min={min(in_scope):.2f}  -> routed to document: "
          f"{sum(s >= RERANK_MIN_SCORE for s in in_scope)}/{len(in_scope)}")
    print(f"  out-of-scope top score max={max(out_scope):.2f}  -> routed to web:      "
          f"{sum(s < RERANK_MIN_SCORE for s in out_scope)}/{len(out_scope)}")
    print("  Pick a threshold between those two values if routing looks wrong.")

    if "--answers" in sys.argv:
        check_answers()


if __name__ == "__main__":
    main()