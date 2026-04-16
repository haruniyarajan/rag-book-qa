"""
vector_store.py
---------------
Stores chunk embeddings in ChromaDB and retrieves relevant ones for queries.

ChromaDB is a free local vector database — it saves to disk so you only
embed once and reuse on every run.
"""

import os
import chromadb
from chromadb.config import Settings
from src.embeddings import embed_texts, embed_query

DB_DIR          = os.path.join("data", "chroma_db")
COLLECTION_NAME = "book_chunks"


def get_client():
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,          # We supply our own vectors
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict], force_reindex: bool = False) -> None:
    """Embed all chunks and store in ChromaDB."""
    print("\n=== 🗄️  Indexing ===")
    client     = get_client()
    collection = get_collection(client)
    existing   = collection.count()

    if existing > 0 and not force_reindex:
        print(f"   ⚡ Already indexed {existing} chunks. Skipping.")
        print("      Pass force_reindex=True to rebuild.\n")
        return

    if force_reindex and existing > 0:
        print(f"   🗑️  Clearing {existing} existing chunks…")
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection(client)

    texts     = [c["text"]       for c in chunks]
    ids       = [c["chunk_id"]   for c in chunks]
    metadatas = [{"page_number": c["page_number"], "source": c["source"],
                  "chunk_index": c["chunk_index"]} for c in chunks]

    print(f"   🔢 Embedding {len(texts)} chunks…")
    embeddings = embed_texts(texts)

    BATCH = 500
    for i in range(0, len(texts), BATCH):
        collection.add(
            ids=ids[i:i+BATCH],
            documents=texts[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )
    print(f"✅ Indexed {len(chunks)} chunks.\n")


def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Find the most relevant chunks for a query."""
    client     = get_client()
    collection = get_collection(client)

    if collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run indexing first.")

    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text":        doc,
            "page_number": meta.get("page_number", "?"),
            "source":      meta.get("source", "?"),
            "distance":    round(dist, 4),
            "relevance":   round(1 - dist, 4),
        })
    return retrieved


def get_stats() -> dict:
    client     = get_client()
    collection = get_collection(client)
    return {"total_chunks": collection.count(), "db_path": os.path.abspath(DB_DIR)}
