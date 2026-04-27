import os
from ollama import Client
from dotenv import load_dotenv
from app.rag.chunker import Chunk

load_dotenv()

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _get_client() -> Client:
    return Client(host=_OLLAMA_BASE_URL)


def embed_chunks(chunks: list[Chunk]) -> list[dict]:
    """
    Embed a list of Chunk objects using Ollama nomic-embed-text.
    Returns list of dicts ready for ChromaDB insertion:
        { "id": str, "text": str, "embedding": list[float], "metadata": dict }
    """
    client = _get_client()
    embedded = []

    for chunk in chunks:
        response = client.embed(model=_EMBED_MODEL, input=chunk.text)
        embedding: list[float] = response.embeddings[0]

        session_id = chunk.metadata.get("session_id", "unknown")
        embedded.append({
            "id":        f"{session_id}_{chunk.chunk_index}",
            "text":      chunk.text,
            "embedding": embedding,
            "metadata":  chunk.metadata,
        })

    return embedded


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string at retrieval time.
    Used by the RAG retriever when an agent sends a question to ChromaDB.
    """
    client = _get_client()
    response = client.embed(model=_EMBED_MODEL, input=query)
    return response.embeddings[0]


if __name__ == "__main__":
    print(f"Testing embedder with model: {_EMBED_MODEL}")
    test_vec = embed_query("Python developer with 5 years experience")
    print(f"Embedding dimensions: {len(test_vec)}")
    print(f"First 5 values: {test_vec[:5]}")
