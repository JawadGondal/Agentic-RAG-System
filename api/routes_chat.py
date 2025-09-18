from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.embeddings_service import get_embeddings
from services.vectordb_service import query_pinecone
from core.config import settings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

router = APIRouter()

# ---- Request & State ----
class ChatRequest(BaseModel):
    query: str

class AgentState(TypedDict):
    query: str
    documents: list[str]
    response: str

# ---- Nodes ----
def retrieve(state: AgentState):
    query_embedding = get_embeddings([state['query']])[0]
    retrieved_docs = query_pinecone(query_embedding)
    return {"documents": retrieved_docs}

def generate(state: AgentState):
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=settings.OPENAI_API_KEY)
    context = "\n\n".join(state["documents"])
    prompt = (
        f"Based on the following context, answer the user's query.\n\n"
        f"Context: {context}\n\n"
        f"Query: {state['query']}\n\n"
        f"Answer:"
    )
    response = llm.invoke(prompt).content
    return {"response": response}

# ---- Build LangGraph ----
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")   # entry point
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)     # finish point

app = builder.compile()

# ---- FastAPI Route ----
@router.post("/chat")
def chat_with_docs(request: ChatRequest):
    try:
        final_state = app.invoke({"query": request.query})
        return {"answer": final_state["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")
