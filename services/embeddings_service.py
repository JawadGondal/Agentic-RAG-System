from openai import OpenAI
from core.config import OPENAI_API_KEY, EMBEDDING_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    # Uses OpenAI embeddings endpoint
    # returns list of vectors (one per text)
    if not texts:
        return []
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = [item.embedding for item in resp.data]
    return embeddings