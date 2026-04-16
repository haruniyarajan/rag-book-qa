"""
rag_pipeline.py
---------------
Core RAG logic: retrieve relevant chunks → ask OpenAI → return answer.

Flow:
  Question → embed → find similar chunks → build prompt → GPT-4o → answer
"""

import os
from openai import OpenAI
from src.vector_store import retrieve_chunks

OPENAI_MODEL = "gpt-4o"


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build a grounded prompt from retrieved chunks."""
    context_parts = [
        f"[Passage {i} — Page {c['page_number']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    context_block = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions strictly based on the provided book passages.

Here are the most relevant passages from the book:

{context_block}

---

User Question: {query}

Instructions:
- Answer using ONLY the information from the passages above.
- Be clear, structured, and beginner-friendly.
- Use bullet points or numbered steps when listing information.
- Mention which page(s) the information comes from (e.g., "According to page 45...").
- If the passages don't contain enough information, say so honestly.
- Do NOT make up information that isn't in the passages.

Answer:"""


def ask_question(
    query:   str,
    top_k:   int  = 5,
    api_key: str  = None,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline: question → retrieve → GPT-4o → answer.

    Returns:
        {question, answer, retrieved_chunks, model, tokens_used}
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found.\n"
            "Set it in .env:  OPENAI_API_KEY=sk-...\n"
            "Or pass it:      ask_question(query, api_key='sk-...')"
        )

    print(f"\n🔍 Retrieving top {top_k} passages…")
    chunks = retrieve_chunks(query, top_k=top_k)

    if verbose:
        print("\n── Retrieved Passages ──")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] Page {c['page_number']} | Relevance: {c['relevance']:.2%}")
            print(f"       {c['text'][:120]}…")

    print("🤖 Asking GPT-4o…")
    client   = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model    = OPENAI_MODEL,
        max_tokens = 1500,
        messages = [{"role": "user", "content": build_prompt(query, chunks)}],
    )

    return {
        "question":         query,
        "answer":           response.choices[0].message.content,
        "retrieved_chunks": chunks,
        "model":            OPENAI_MODEL,
        "tokens_used":      response.usage.total_tokens,
    }


def print_answer(result: dict) -> None:
    """Pretty-print a result dict."""
    print("\n" + "=" * 70)
    print(f"❓ QUESTION:\n   {result['question']}")
    print("=" * 70)
    print(f"\n💡 ANSWER:\n\n{result['answer']}")
    print("\n" + "-" * 70)
    print("📚 Sources used:")
    for c in result["retrieved_chunks"]:
        print(f"   • Page {c['page_number']} (relevance: {c['relevance']:.2%})")
    print(f"\n🔢 Tokens: {result['tokens_used']} | Model: {result['model']}")
    print("=" * 70 + "\n")
