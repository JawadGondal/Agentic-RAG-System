from typing import List
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CHUNK_SIZE, CHUNK_OVERLAP




def extract_text_from_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)




def chunk_text(text: str) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer larger semantic breaks
    )

    chunks = splitter.split_text(text)
    return chunks