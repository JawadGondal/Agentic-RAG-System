from pinecone import Pinecone, ServerlessSpec
from core.config import PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION, PINECONE_INDEX

# Global Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index_if_not_exists(dim: int = 1536):
    try:
        indexes = pc.list_indexes().names()
        if PINECONE_INDEX not in indexes:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_INDEX, region=PINECONE_REGION)
            )
            print(f"Created index {PINECONE_INDEX}")
    except Exception as e:
        print("Pinecone init/create failed:", e)


def upsert_vectors(vectors: list[dict]):
    """Upsert vectors to Pinecone index."""
    idx = pc.Index(PINECONE_INDEX)
    # pinecone expects list of (id, vector, metadata)
    to_upsert = [(v["id"], v["values"], v.get("metadata", {})) for v in vectors]
    res = idx.upsert(vectors=to_upsert)
    return res


def query_vectors(query_embedding: list[float], top_k: int = 4) -> list[dict]:
    """Query vectors from Pinecone index."""
    idx = pc.Index(PINECONE_INDEX)
    qres = idx.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    results = []
    for m in qres.matches:
        results.append(
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata,
            }
        )
    return results


def delete_vectors_by_file(file_id: str) -> bool:
    """Delete all vectors related to a given file_id."""
    idx = pc.Index(PINECONE_INDEX)
    try:
        idx.delete(filter={"file_id": {"$eq": file_id}})
        return True
    except Exception as e:
        print("Delete failed:", e)
        return False
