from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import os
from services.data_injestion_service import extract_text_from_pdf, split_text_into_chunks
from services.embeddings_service import get_embeddings
from services.vectordb_service import add_vectors_to_pinecone, delete_vectors_by_file_id

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/add_file")
async def add_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, file_id + ".pdf")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        text = extract_text_from_pdf(file_path)
        chunks = split_text_into_chunks(text)
        embeddings = get_embeddings(chunks)
        add_vectors_to_pinecone(chunks, embeddings, file_id)
        
        return {"file_id": file_id, "message": "File uploaded and processed successfully."}
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

@router.delete("/delete_file/{file_id}")
def delete_file(file_id: str):
    try:
        delete_vectors_by_file_id(file_id)
        return {"message": f"Vectors for file_id {file_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

@router.put("/update_file/{file_id}")
async def update_file(file_id: str, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        delete_vectors_by_file_id(file_id)
        
        file_path = os.path.join(UPLOAD_DIR, file_id + ".pdf")
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        text = extract_text_from_pdf(file_path)
        chunks = split_text_into_chunks(text)
        embeddings = get_embeddings(chunks)
        add_vectors_to_pinecone(chunks, embeddings, file_id)
        
        return {"message": f"File with file_id {file_id} updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update file: {e}")