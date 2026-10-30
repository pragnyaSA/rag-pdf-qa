"""RAG pipeline: load PDF -> chunk -> embed -> store in Chroma -> retrieve -> generate."""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "data/rag_survey.pdf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("TOP_K", 6))

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a PDF document.
Use ONLY the context below. If the answer is not in the context, say
"I could not find this in the document."
Answer concisely in a short paragraph or a few bullet points. Do not add facts that are not in the context.

Context:
{context}

Question: {question}

Answer:"""
)

_embeddings = None
_vectorstore = None
_chain = None
_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def _strip_references(pages):
    """Drop the bibliography so citation lists don't pollute retrieval."""
    kept = []
    for p in pages:
        idx = p.page_content.find("REFERENCES")
        if idx != -1:
            p.page_content = p.page_content[:idx]
            if p.page_content.strip():
                kept.append(p)
            break
        kept.append(p)
    return kept



def build_index(pdf_path: str = PDF_PATH) -> Chroma:
    """Load PDF, split into chunks, embed and persist in Chroma."""
    pages = PyPDFLoader(pdf_path).load()  # one Document per page, with page metadata
    pages = _strip_references(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1  # 1-based page number
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")
    return Chroma.from_documents(
        chunks, get_embeddings(), persist_directory=CHROMA_DIR
    )


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            _vectorstore = Chroma(
                persist_directory=CHROMA_DIR, embedding_function=get_embeddings()
            )
        else:
            _vectorstore = build_index()
    return _vectorstore


def _format_docs(docs):
    return "\n\n".join(f"[Page {d.metadata['page']}] {d.page_content}" for d in docs)


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0)
    return _llm


def get_chain():
    """Document-QA chain: prompt -> LLM -> string."""
    global _chain
    if _chain is None:
        _chain = PROMPT | get_llm() | StrOutputParser()
    return _chain


def expand_with_neighbors(results, top_n=3, window=3):
    """Add the chunks before/after the top_n best matches, in document order."""
    wanted = {d.metadata["chunk_id"] for d, _ in results}
    for d, _ in results[:top_n]:
        cid = d.metadata["chunk_id"]
        wanted.update(range(max(0, cid - window), cid + window + 1))
    got = get_vectorstore().get(where={"chunk_id": {"$in": sorted(wanted)}})
    docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(got["documents"], got["metadatas"])
    ]
    return sorted(docs, key=lambda d: d.metadata["chunk_id"])
