from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Literal, Optional
from pydantic import BaseModel, Field
from app.rag import get_vectorstore
from app.graph import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_vectorstore()  # builds the index on first run, loads it afterwards
    yield


app = FastAPI(title="PDF RAG Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is Retrieval-Augmented Generation?"])


class Source(BaseModel):
    type: Literal["document", "web"] = "document"
    # document sources
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    score: Optional[float] = None  # cross-encoder rerank score (higher = more relevant)
    # web sources
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    route: Literal["document", "web"]  # where the answer came from
    sources: list[Source]  # best-matching chunks (or web pages)
    context_chunk_ids: list[int] = []  # every chunk sent to the LLM, incl. neighbor chunks


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(req: AskRequest):
    try:
        return answer_question(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))