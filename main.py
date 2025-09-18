from fastapi import FastAPI
from api import routes_chat, routes_files

app = FastAPI(
    title="Agentic RAG System",
    description="A minimal RAG system with LangGraph, Pinecone, and OpenAI."
)

app.include_router(routes_files.router, tags=["files"], prefix="/files")
app.include_router(routes_chat.router, tags=["chat"], prefix="/chat")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Agentic RAG System API!"}