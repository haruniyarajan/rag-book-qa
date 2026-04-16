"""
evaluator.py
------------
Scores the RAG system using GPT-4o as judge.

Metrics (each scored 1-5):
  - Retrieval Relevance  : Are the right chunks being found?
  - Faithfulness         : Is the answer grounded in the context?
  - Answer Relevancy     : Does the answer address the question?
  - Context Coverage     : Does the answer use all available context?
"""

import json
import os
import time
from openai import OpenAI
from src.rag_pipeline import ask_question

JUDGE_MODEL = "gpt-4o"

DEFAULT_TEST_QUESTIONS = [
    "What is the main lesson about wealth and getting rich?",
    "How does luck influence financial success?",
    "What does the author say about saving money?",
    "How should one think about risk in investing?",
    "What is the relationship between money and happiness?",
    "What does the author mean by 'tails, you win'?",
    "How important is staying wealthy versus getting wealthy?",
    "What role does compounding play in building wealth?",
]


def evaluate_single(question: str, answer: str, chunks: list[dict], api_key: str) -> dict:
    """Score a single Q&A pair on 4 dimensions (1-5 each)."""
    context = "\n---\n".join(
        [f"[Page {c['page_number']}] {c['text'][:300]}" for c in chunks]
    )
    prompt = f"""You are an expert RAG evaluator. Score the Q&A below 1-5 for each criterion.

QUESTION: {question}
RETRIEVED CONTEXT: {context}
GENERATED ANSWER: {answer}

Return ONLY valid JSON (no markdown fences):
{{
  "retrieval_relevance": <1-5>,
  "faithfulness": <1-5>,
  "answer_relevancy": <1-5>,
  "context_coverage": <1-5>,
  "explanation": "<one short paragraph>"
}}"""

    client   = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=JUDGE_MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        scores = json.loads(raw.strip())
    except Exception:
        scores = {"retrieval_relevance": 0, "faithfulness": 0,
                  "answer_relevancy": 0, "context_coverage": 0,
                  "explanation": f"Parse error: {raw[:100]}"}

    dims  = ["retrieval_relevance", "faithfulness", "answer_relevancy", "context_coverage"]
    valid = [scores[d] for d in dims if isinstance(scores.get(d), (int, float)) and scores[d] > 0]
    scores["overall_score"] = round(sum(valid) / len(valid), 2) if valid else 0.0
    return scores


def run_evaluation(
    questions:  list[str] = None,
    top_k:      int        = 5,
    api_key:    str        = None,
    output_dir: str        = "eval_results",
) -> dict:
    """Run full evaluation and save report."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set.")
    if questions is None:
        questions = DEFAULT_TEST_QUESTIONS

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🧪 Evaluating {len(questions)} questions…")

    all_results = []
    totals = {m: 0 for m in ["retrieval_relevance","faithfulness",
                               "answer_relevancy","context_coverage","overall_score"]}

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q[:65]}…")
        rag    = ask_question(q, top_k=top_k, api_key=key)
        scores = evaluate_single(q, rag["answer"], rag["retrieved_chunks"], key)
        all_results.append({"question": q, "answer": rag["answer"],
                             "scores": scores, "sources": rag["retrieved_chunks"]})
        for m in totals:
            totals[m] += scores.get(m, 0)
        print(f"   ✅ Overall: {scores['overall_score']}/5.0 | "
              f"Faith: {scores['faithfulness']} | Rel: {scores['answer_relevancy']}")
        if i < len(questions):
            time.sleep(0.5)

    n       = len(questions)
    summary = {
        "num_questions":           n,
        "avg_overall_score":       round(totals["overall_score"]       / n, 2),
        "avg_faithfulness":        round(totals["faithfulness"]        / n, 2),
        "avg_answer_relevancy":    round(totals["answer_relevancy"]    / n, 2),
        "avg_retrieval_relevance": round(totals["retrieval_relevance"] / n, 2),
        "avg_context_coverage":    round(totals["context_coverage"]    / n, 2),
        "per_question_results":    all_results,
    }

    with open(f"{output_dir}/evaluation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable summary
    with open(f"{output_dir}/evaluation_summary.txt", "w") as f:
        f.write("RAG EVALUATION REPORT\n" + "="*50 + "\n\n")
        f.write(f"Questions       : {n}\n")
        f.write(f"Overall Score   : {summary['avg_overall_score']} / 5.0\n")
        f.write(f"Faithfulness    : {summary['avg_faithfulness']} / 5.0\n")
        f.write(f"Answer Relevancy: {summary['avg_answer_relevancy']} / 5.0\n")
        f.write(f"Retrieval Rel.  : {summary['avg_retrieval_relevance']} / 5.0\n")
        f.write(f"Context Coverage: {summary['avg_context_coverage']} / 5.0\n\n")
        for r in all_results:
            f.write(f"\nQ: {r['question']}\n")
            f.write(f"   Score: {r['scores']['overall_score']}/5 — {r['scores']['explanation'][:150]}\n")

    print(f"\n{'='*60}")
    print(f"📊 Avg Overall Score    : {summary['avg_overall_score']} / 5.0 ⭐")
    print(f"   Faithfulness         : {summary['avg_faithfulness']}")
    print(f"   Answer Relevancy     : {summary['avg_answer_relevancy']}")
    print(f"   Retrieval Relevance  : {summary['avg_retrieval_relevance']}")
    print(f"   Context Coverage     : {summary['avg_context_coverage']}")
    print(f"\n   Report saved → {output_dir}/evaluation_report.json")
    print(f"{'='*60}\n")
    return summary
