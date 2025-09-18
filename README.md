# Agentic RAG System (FastAPI + LangGraph + Pinecone + OpenAI)

This project implements a **Retrieval-Augmented Generation (RAG)** system using:

- **LangGraph** → Agentic orchestration (retriever → LLM nodes flow)  
- **FastAPI** → REST API service  
- **Pinecone** → Vector database  
- **OpenAI** → Embeddings + LLM (ChatGPT)  
- **PDF Support only** → Document ingestion pipeline  

---

## Features

✅ Upload a **PDF file** → extract + chunk text → embed → store in Pinecone  
✅ Query the system → retrieve relevant chunks from Pinecone → context-aware LLM answer  
✅ Update or delete documents by `file_id`  
✅ LangGraph workflow for modular, agentic orchestration  
✅ FastAPI endpoints for easy integration  

---

## Project Structure

rag_app/
│
├── main.py # FastAPI entrypoint
│
├── api/ # API routes
│ ├── init.py
│ ├── routes_chat.py # /chat endpoint (LangGraph)
│ ├── routes_files.py # /add_file, /delete_file, /update_file endpoints
│
├── core/ # Core configs
│ ├── init.py
│ ├── config.py # Load env vars (OpenAI + Pinecone)
│
├── services/ # Business logic
│ ├── init.py
│ ├── data_injestion_service.py # PDF text extraction + chunking
│ ├── embeddings_service.py # OpenAI embeddings
│ ├── llm_service.py # OpenAI llm for QA
│ ├── vectordb_service.py # Pinecone insert, delete, query
│
├── utils/ # (optional utilities)
│ ├── init.py
│ ├── logger.py # Centralized logging
│
├── requirements.txt # Python dependencies
├── README.md # Documentation
├── .env # Environment variables

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/JawadGondal/Agentic-RAG-Syste

```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure environment variables

Create a .env file in the root directory:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=...
PINECONE_CLOUD = aws
PINECONE_REGION = us-east-1
PINECONE_INDEX=rag-index
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## ▶️ Running the Server

Start the FastAPI server:

```bash
uvicorn main:app --reload --port 8000
```

API docs available at:
👉 Swagger UI: http://127.0.0.1:8000/docs
👉 ReDoc: http://127.0.0.1:8000/redoc

## 📌 API Endpoints

### 1. Add File (PDF ingestion)

```bash
POST /add_file
```

Description: Upload a PDF → extract text → chunk → embed → upsert into Pinecone.

Request:

Multipart file upload

Response:

```bash
{
  "file_id": "myfile-123",
  "message": "File successfully added."
}
```

### 2. Delete File

```bash
DELETE /delete_file/{file_id}
```

Description: Remove all vectors belonging to a file_id from Pinecone.

**Request**

```bash
file_id = myfile-123
```

Response:

```bash
{
  "message": "File myfile-123 deleted successfully."
}
```

### 3. Update File

```bash
PUT /update_file/{file_id}
```

Description: Replace an existing file with a new PDF (re-ingestion).

**Request**

```bash
file_id = myfile-123
```

Response:

```bash
{
  "message": "File myfile-123 updated successfully."
}
```

### 4. Chat (RAG Query)

```bash
POST /chat
```

Description: Retrieve relevant chunks from Pinecone + answer query via LangGraph → OpenAI.

Request:

```bash
{
  "query": "What does the document say about data privacy?"
}
```


Response:

```bash
{
  "response": "The document states that data privacy is ensured through..."
}
```

🧠 LangGraph Workflow

The ```bash/chat``` endpoint runs a LangGraph state machine:

User Query → [Retriever Node] → [LLM Node with Context] → Answer


Retriever Node → pulls relevant chunks from Pinecone

LLM Node → generates answer with context

StateGraph → ensures smooth data flow
