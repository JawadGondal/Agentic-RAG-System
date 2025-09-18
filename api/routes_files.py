from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.data_injestion_service import extract_text_from_pdf, chunk_text
from services.embeddings_service import get_embeddings
from services.vectordb_service import upsert_vectors, delete_vectors_by_file, create_index_if_not_exists
import uuid


router = APIRouter()

# ensure index exists at startup
create_index_if_not_exists()


@router.post("/add_file")
async def add_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    file_id = str(uuid.uuid4())
    filename = file.filename  # original uploaded file name
    content = await file.read()
    text = extract_text_from_pdf(content)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    # prepare vectors: id includes file_id prefix
    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            'id': f"{file_id}_chunk_{i}",
            'values': emb,
            'metadata': {'file_id': file_id, 'title':filename, 'chunk_index': i, 'text': chunks[i][:500]}
            })
    upsert_vectors(vectors)
    return JSONResponse({"file_id": file_id, "message": "PDF ingested and vectors stored"})


@router.delete("/delete_file/{file_id}")
async def delete_file(file_id: str):
    deleted = delete_vectors_by_file(file_id)
    if deleted:
        return JSONResponse({"file_id": file_id, "message": "Vectors deleted"})
    else:
        raise HTTPException(status_code=404, detail="No vectors found for file_id")


@router.put("/update_file/{file_id}")
async def update_file(file_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await file.read()
    text = extract_text_from_pdf(content)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    # delete old vectors first
    delete_vectors_by_file(file_id)
    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            'id': f"{file_id}_chunk_{i}",
            'values': emb,
            'metadata': {'file_id': file_id, 'chunk_index': i, 'text': chunks[i][:500]}
            })
    upsert_vectors(vectors)
    return JSONResponse({"file_id": file_id, "message": "File updated and vectors replaced"})