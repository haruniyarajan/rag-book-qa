"""
embeddings.py
-------------
Converts text into vectors (embeddings) using a free local model.

What is an embedding?
  A list of numbers [0.12, -0.45, ...] that captures the MEANING of text.
  Similar text = similar numbers = easy to search.

Model: all-MiniLM-L6-v2 (free, runs locally, no API cost)
"""

from sentence_transformers import SentenceTransformer

_model = None  # Loaded once, reused forever


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load embedding model (only on first call)."""
    global _model
    if _model is None:
        print(f"⚙️  Loading embedding model: {model_name}…")
        _model = SentenceTransformer(model_name)
        print("   Model ready ✓")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Convert a list of strings into embedding vectors."""
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
    ).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
