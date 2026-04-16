"""
main.py — CLI interface (alternative to the web UI)

Usage:
  python main.py index --pdf data/book.pdf
  python main.py ask "How do I build wealth?"
  python main.py chat
  python main.py stats
"""

import argparse, os, sys
from dotenv import load_dotenv
load_dotenv()

from src.pdf_processor import process_pdf
from src.vector_store  import index_chunks, get_stats
from src.rag_pipeline  import ask_question, print_answer


def cmd_index(args):
    if not os.path.exists(args.pdf):
        print(f"❌ PDF not found: {args.pdf}"); sys.exit(1)
    chunks = process_pdf(args.pdf, chunk_size=args.chunk_size, overlap=args.overlap)
    index_chunks(chunks, force_reindex=args.reindex)
    stats = get_stats()
    print(f"📦 DB: {stats['db_path']}")
    print(f"📊 Chunks: {stats['total_chunks']}")


def cmd_ask(args):
    result = ask_question(args.question, top_k=args.top_k, verbose=args.verbose)
    print_answer(result)


def cmd_chat(args):
    print("\n🗣️  CHAT MODE — type 'exit' to quit\n")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!"); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("👋 Bye!"); break
        result = ask_question(q, top_k=args.top_k, verbose=args.verbose)
        print_answer(result)


def cmd_stats(_):
    stats = get_stats()
    print(f"\n📊 Chunks: {stats['total_chunks']}\n   DB: {stats['db_path']}\n")


def main():
    p = argparse.ArgumentParser(description="📚 RAG Book Q&A CLI")
    sub = p.add_subparsers(dest="command", required=True)

    ip = sub.add_parser("index")
    ip.add_argument("--pdf",        required=True)
    ip.add_argument("--chunk-size", type=int, default=500)
    ip.add_argument("--overlap",    type=int, default=100)
    ip.add_argument("--reindex",    action="store_true")

    ap = sub.add_parser("ask")
    ap.add_argument("question")
    ap.add_argument("--top-k",   type=int, default=5)
    ap.add_argument("--verbose", action="store_true")

    cp = sub.add_parser("chat")
    cp.add_argument("--top-k",   type=int, default=5)
    cp.add_argument("--verbose", action="store_true")

    sub.add_parser("stats")

    args = p.parse_args()
    {"index": cmd_index, "ask": cmd_ask, "chat": cmd_chat, "stats": cmd_stats}[args.command](args)


if __name__ == "__main__":
    main()
