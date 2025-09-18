import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY : str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY : str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME : str = os.getenv("PINECONE_INDEX_NAME","rag-app-index")

settings = Settings()