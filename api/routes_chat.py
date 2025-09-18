# ---- rag_app/api/routes_chat.py ----
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from services.embeddings_service import get_embeddings
from services.vectordb_service import query_vectors
from services.llm_service import llm_answer


router = APIRouter()


class ChatRequest(BaseModel):
    query: str


# --- LangGraph RAG State ---
class RAGState(TypedDict):
    query: str
    query_embedding: List[float]
    retrieved_docs: List[Dict[str, Any]]
    answer: str


# --- Graph Nodes ---
def embed_query(state: RAGState) -> RAGState:
    q_emb = get_embeddings([state["query"]])[0]
    state["query_embedding"] = q_emb
    return state


def retrieve_docs(state: RAGState) -> RAGState:
    results = query_vectors(state["query_embedding"], top_k=4)
    state["retrieved_docs"] = results
    return state


def generate_answer(state: RAGState) -> RAGState:
    context_texts = [r['metadata'].get('text', '') for r in state["retrieved_docs"]]
    prompt = "You are a helpful assistant. Use the following document context:\n\n"
    for i, t in enumerate(context_texts):
        prompt += f"[{i}] {t}\n"
    prompt += f"\nUser question: {state['query']}\nAnswer concisely."
    state["answer"] = llm_answer(prompt)
    return state


# --- Build Graph once at startup ---
graph = StateGraph(RAGState)
graph.add_node("embed_query", embed_query)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("generate_answer", generate_answer)

graph.set_entry_point("embed_query")
graph.add_edge("embed_query", "retrieve_docs")
graph.add_edge("retrieve_docs", "generate_answer")
graph.add_edge("generate_answer", END)

rag_graph = graph.compile()


# --- FastAPI endpoint ---
@router.post("/")
async def chat(req: ChatRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    init_state = {"query": req.query}
    final_state = rag_graph.invoke(init_state)

    return {
        "answer": final_state["answer"],
        "sources": [r["id"] for r in final_state["retrieved_docs"]],
    }
