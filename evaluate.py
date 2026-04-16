"""
evaluate.py — Run the RAG evaluation suite

Usage:
  python evaluate.py
  python evaluate.py --top-k 6
  python evaluate.py --questions "What is wealth?" "How does luck work?"
"""

import argparse, os, sys
from dotenv import load_dotenv
load_dotenv()

from src.evaluator import run_evaluation, DEFAULT_TEST_QUESTIONS


def main():
    p = argparse.ArgumentParser(description="🧪 RAG Evaluation")
    p.add_argument("--questions", nargs="+", default=None)
    p.add_argument("--top-k",     type=int,  default=5)
    p.add_argument("--output-dir",           default="eval_results")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Add it to .env"); sys.exit(1)

    run_evaluation(
        questions  = args.questions or DEFAULT_TEST_QUESTIONS,
        top_k      = args.top_k,
        output_dir = args.output_dir,
    )


if __name__ == "__main__":
    main()
