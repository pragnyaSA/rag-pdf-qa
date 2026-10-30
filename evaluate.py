"""Retrieval evaluation: Hit Rate@k and MRR on a small hand-labelled question set.
Run: python evaluate.py
A question is a 'hit' if any of the top-k retrieved chunks comes from an expected page."""
from app.rag import get_vectorstore, TOP_K

# (question, pages where the answer is found in the PDF)
EVAL_SET = [
    ("What is Retrieval-Augmented Generation?", [1]),
    ("What are the three paradigms of RAG?", [1, 2]),
    ("How is RAG different from fine-tuning?", [5]),
    ("What are the three quality scores used to evaluate RAG?", [12]),
    ("Which tools are used to evaluate RAG?", [14]),
    ("What is the lost in the middle problem?", [10]),
]


def main():
    store = get_vectorstore()
    hits, rr_sum = 0, 0.0
    for question, expected in EVAL_SET:
        results = store.similarity_search(question, k=TOP_K)
        pages = [d.metadata["page"] for d in results]
        rank = next((i + 1 for i, p in enumerate(pages) if p in expected), None)
        if rank:
            hits += 1
            rr_sum += 1 / rank
        print(f"{'HIT ' if rank else 'MISS'} rank={rank} pages={pages} | {question}")
    n = len(EVAL_SET)
    print(f"\nHit Rate@{TOP_K}: {hits}/{n} = {hits / n:.2f}")
    print(f"MRR: {rr_sum / n:.2f}")


if __name__ == "__main__":
    main()