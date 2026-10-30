from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.rag import answer_question, get_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_vectorstore()  # builds the index on first run, loads it afterwards
    yield


app = FastAPI(title="PDF RAG Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is Retrieval-Augmented Generation?"])


class Source(BaseModel):
    page: int
    chunk_id: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return answer_question(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
