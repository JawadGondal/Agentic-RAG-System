from fastapi import FastAPI
from api import routes_chat, routes_files
import uvicorn


app = FastAPI(title="Agentic RAG - LangGraph + Pinecone + OpenAI")


app.include_router(routes_files.router, prefix="/files")
app.include_router(routes_chat.router, prefix="/chat")


@app.get("/")
async def root():
    return {"message": "Agentic RAG service running. See /docs for API"}

if __name__ == "__main__":
    uvicorn.run("rag_app.main:app", host="0.0.0.0", port=8080, reload=True)