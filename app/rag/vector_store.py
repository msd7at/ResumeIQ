import os
import chromadb
from chromadb import Collection
from dotenv import load_dotenv

load_dotenv()

_CHROMA_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
_COLLECTION_NAME = "resumes"


def _get_collection() -> Collection:
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_embeddings(embedded_chunks: list[dict]) -> int:
    """
    Insert embedded chunks into ChromaDB.
    Input: output of embed_chunks() — list of {id, text, embedding, metadata}
    Returns: number of chunks stored.
    """
    if not embedded_chunks:
        return 0

    collection = _get_collection()
    collection.add(
        ids        = [c["id"]        for c in embedded_chunks],
        documents  = [c["text"]      for c in embedded_chunks],
        embeddings = [c["embedding"] for c in embedded_chunks],
        metadatas  = [c["metadata"]  for c in embedded_chunks],
    )
    return len(embedded_chunks)


def retrieve_chunks(
    query_embedding: list[float],
    session_id: str,
    n_results: int = 5,
    section: str | None = None,
) -> list[dict]:
    """
    Find the top-n most similar chunks for a query embedding.
    Always filters to a single session so resumes don't bleed into each other.
    Optionally narrow to a specific section (e.g. "SKILLS").
    Returns: list of {text, metadata, distance}  (distance: lower = more similar)
    """
    collection = _get_collection()

    where: dict = {"session_id": session_id}
    if section:
        where = {"$and": [{"session_id": session_id}, {"section": section}]}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def delete_session(session_id: str) -> None:
    """Delete all chunks belonging to a session (cleanup on re-upload)."""
    collection = _get_collection()
    collection.delete(where={"session_id": session_id})


def count_chunks(session_id: str) -> int:
    """Return how many chunks are stored for a session."""
    collection = _get_collection()
    result = collection.get(where={"session_id": session_id}, include=[])
    return len(result["ids"])


if __name__ == "__main__":
    print(f"ChromaDB path : {_CHROMA_DIR}")
    col = _get_collection()
    print(f"Collection    : {col.name}")
    print(f"Total vectors : {col.count()}")
