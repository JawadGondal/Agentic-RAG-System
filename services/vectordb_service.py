from pinecone import Pinecone, ServerlessSpec
from core.config import settings
import uuid

pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)

def get_or_create_index():
    index_name = settings.PINECONE_INDEX_NAME
    if index_name not in pinecone.list_indexes().names:
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Dimension for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    return pinecone.Index(index_name)

def add_vectors_to_pinecone(texts: list[str], embeddings: list, file_id: str):
    index = get_or_create_index()
    vectors = []
    for text, embedding in zip(texts, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {"text": text, "file_id": file_id}
        })
    index.upsert(vectors=vectors)

def delete_vectors_by_file_id(file_id: str):
    index = get_or_create_index()
    index.delete(filter={"file_id": {"$eq": file_id}})
    
def query_pinecone(query_embedding: list, top_k: int = 5):
    index = get_or_create_index()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match['metadata']['text'] for match in results.matches]