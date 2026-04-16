"""
app.py
------
Streamlit web UI for the RAG Book Q&A system.

Run it with:
  streamlit run app.py

Features:
  - Upload any PDF book directly in the browser
  - Auto-indexes on upload (no terminal needed)
  - Chat interface to ask questions
  - Shows source pages used for each answer
  - Evaluation tab with live scores
  - Sidebar with settings (top-k, model, API key)
"""

import os
import time
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "📚 RAG Book Q&A",
    page_icon  = "📚",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .main-header h1 {
    font-family: 'Playfair Display', serif;
    color: #e2c27d;
    font-size: 2.2rem;
    margin: 0 0 0.3rem 0;
  }
  .main-header p {
    color: rgba(255,255,255,0.6);
    margin: 0;
    font-size: 0.95rem;
  }

  /* Chat bubbles */
  .chat-user {
    background: #0f3460;
    color: #fff;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 4rem;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .chat-assistant {
    background: #1a1a2e;
    color: #e8e8e8;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 4rem 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
    border: 1px solid rgba(226,194,125,0.15);
  }
  .chat-label-user {
    text-align: right;
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 2px;
  }
  .chat-label-bot {
    font-size: 0.75rem;
    color: #e2c27d;
    margin-bottom: 2px;
  }

  /* Source cards */
  .source-card {
    background: rgba(15,52,96,0.3);
    border: 1px solid rgba(226,194,125,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #aaa;
  }
  .source-card strong { color: #e2c27d; }

  /* Score badges */
  .score-badge {
    display: inline-block;
    background: rgba(226,194,125,0.15);
    border: 1px solid rgba(226,194,125,0.3);
    color: #e2c27d;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem;
  }

  /* Status pill */
  .status-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .status-ready   { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
  .status-pending { background: rgba(234,179,8,0.15);  color: #fbbf24; border: 1px solid rgba(234,179,8,0.3); }
  .status-error   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

  /* Streamlit overrides */
  .stButton > button {
    background: linear-gradient(135deg, #e2c27d, #c9a84c);
    color: #1a1a2e;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(226,194,125,0.4);
  }
  div[data-testid="stSidebar"] {
    background: #0d0d1a;
  }
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: #1a1a2e;
    color: #e8e8e8;
    border: 1px solid rgba(226,194,125,0.2);
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "indexed"       not in st.session_state: st.session_state.indexed       = False
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = None
if "eval_results"  not in st.session_state: st.session_state.eval_results  = None
if "chunk_count"   not in st.session_state: st.session_state.chunk_count   = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type     = "password",
        value    = os.environ.get("OPENAI_API_KEY", ""),
        help     = "Get yours at platform.openai.com/api-keys",
        placeholder = "sk-...",
    )

    st.divider()

    # Retrieval settings
    top_k = st.slider(
        "📎 Passages to retrieve",
        min_value = 2, max_value = 10, value = 5,
        help = "More passages = more context for GPT-4o, but slower"
    )

    chunk_size = st.slider(
        "✂️ Chunk size (words)",
        min_value = 200, max_value = 1000, value = 500, step = 50,
        help = "Words per chunk. Smaller = more precise. Larger = more context."
    )

    st.divider()

    # Upload PDF
    st.markdown("### 📄 Upload Book PDF")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type = ["pdf"],
        help = "The PDF will be indexed automatically after upload."
    )

    if uploaded_file:
        reindex = st.checkbox("🔄 Re-index (if already indexed)", value=False)

        if st.button("📥 Index Book", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter your OpenAI API key first.")
            else:
                with st.spinner("Processing PDF…"):
                    # Save uploaded PDF to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        from src.pdf_processor import process_pdf
                        from src.vector_store  import index_chunks, get_stats

                        os.makedirs("data", exist_ok=True)
                        chunks = process_pdf(tmp_path, chunk_size=chunk_size)
                        index_chunks(chunks, force_reindex=reindex)
                        stats = get_stats()

                        st.session_state.indexed     = True
                        st.session_state.pdf_name    = uploaded_file.name
                        st.session_state.chunk_count = stats["total_chunks"]
                        st.session_state.messages    = []  # Clear old chat
                        st.success(f"✅ Indexed {stats['total_chunks']} chunks!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    finally:
                        os.unlink(tmp_path)

    st.divider()

    # Status
    st.markdown("### 📊 Status")
    if st.session_state.indexed:
        st.markdown(f'<span class="status-pill status-ready">✅ Ready</span>', unsafe_allow_html=True)
        st.caption(f"📚 {st.session_state.pdf_name}")
        st.caption(f"🗂️ {st.session_state.chunk_count} chunks indexed")
    else:
        # Check if already indexed from a previous run
        try:
            from src.vector_store import get_stats
            stats = get_stats()
            if stats["total_chunks"] > 0:
                st.session_state.indexed     = True
                st.session_state.chunk_count = stats["total_chunks"]
                st.markdown(f'<span class="status-pill status-ready">✅ Ready</span>', unsafe_allow_html=True)
                st.caption(f"🗂️ {stats['total_chunks']} chunks from previous session")
            else:
                st.markdown(f'<span class="status-pill status-pending">⏳ No book indexed</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown(f'<span class="status-pill status-pending">⏳ No book indexed</span>', unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📚 RAG Book Q&A</h1>
  <p>Upload any PDF book · Ask questions · Get answers grounded in the actual text</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_eval, tab_about = st.tabs(["💬 Chat", "🧪 Evaluation", "ℹ️ About"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if not st.session_state.indexed:
        st.info("👈 Upload a PDF book in the sidebar and click **Index Book** to get started.")
    else:
        # Render chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-label-user">You</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label-bot">📚 Assistant</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
                # Show sources if available
                if "sources" in msg:
                    with st.expander("📄 View source passages", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>Page {src["page_number"]}</strong> &nbsp;·&nbsp; '
                                f'Relevance: {src["relevance"]:.1%}<br/>'
                                f'<span style="color:#888">{src["text"][:200]}…</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                if "tokens" in msg:
                    st.caption(f"🔢 {msg['tokens']} tokens used")

        st.divider()

        # Quick question buttons
        st.markdown("**💡 Try these questions:**")
        quick_questions = [
            "What is the main lesson about building wealth?",
            "How does luck influence financial success?",
            "What does the book say about saving money?",
            "How important is compounding?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(quick_questions):
            if cols[i % 2].button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state["prefill_question"] = q
                st.rerun()

        # Question input
        prefill = st.session_state.pop("prefill_question", "")
        question = st.text_input(
            "Ask anything about the book:",
            value       = prefill,
            placeholder = "e.g. What are the key steps to financial freedom?",
            key         = "question_input",
        )

        col1, col2 = st.columns([1, 4])
        ask_clicked = col1.button("🚀 Ask", use_container_width=True)

        if ask_clicked and question.strip():
            if not api_key:
                st.error("❌ Please enter your OpenAI API key in the sidebar.")
            else:
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": question})

                with st.spinner("🔍 Searching passages & generating answer…"):
                    try:
                        from src.rag_pipeline import ask_question as rag_ask

                        result = rag_ask(question, top_k=top_k, api_key=api_key)

                        st.session_state.messages.append({
                            "role":    "assistant",
                            "content": result["answer"],
                            "sources": result["retrieved_chunks"],
                            "tokens":  result["tokens_used"],
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role":    "assistant",
                            "content": f"❌ Error: {str(e)}",
                        })

                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 🧪 RAG Evaluation")
    st.markdown(
        "Tests the RAG system on multiple questions and scores it using **GPT-4o as judge**. "
        "Each metric is rated 1–5."
    )

    if not st.session_state.indexed:
        st.info("👈 Index a book first before running evaluation.")
    else:
        # Custom questions
        st.markdown("**Test questions** (one per line):")
        from src.evaluator import DEFAULT_TEST_QUESTIONS
        default_qs = "\n".join(DEFAULT_TEST_QUESTIONS)
        custom_qs_text = st.text_area(
            "Questions",
            value  = default_qs,
            height = 200,
            label_visibility = "collapsed",
        )

        if st.button("▶️ Run Evaluation", use_container_width=False):
            if not api_key:
                st.error("❌ OpenAI API key required.")
            else:
                questions = [q.strip() for q in custom_qs_text.strip().splitlines() if q.strip()]
                progress  = st.progress(0, text="Starting evaluation…")
                status    = st.empty()

                try:
                    from src.evaluator import evaluate_single, ask_question as rag_ask

                    all_results = []
                    totals = {"retrieval_relevance":0,"faithfulness":0,
                              "answer_relevancy":0,"context_coverage":0,"overall_score":0}

                    for i, q in enumerate(questions):
                        status.markdown(f"**[{i+1}/{len(questions)}]** {q[:60]}…")
                        rag    = rag_ask(q, top_k=top_k, api_key=api_key)
                        scores = evaluate_single(q, rag["answer"], rag["retrieved_chunks"], api_key)
                        all_results.append({"question":q,"answer":rag["answer"],"scores":scores})
                        for m in totals:
                            totals[m] += scores.get(m, 0)
                        progress.progress((i+1)/len(questions), text=f"{i+1}/{len(questions)} done")
                        time.sleep(0.3)

                    n = len(questions)
                    summary = {k: round(v/n, 2) for k, v in totals.items()}
                    st.session_state.eval_results = {"summary": summary, "details": all_results}

                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")

        # Show results
        if st.session_state.eval_results:
            summary = st.session_state.eval_results["summary"]
            details = st.session_state.eval_results["details"]

            st.divider()
            st.markdown("#### 📊 Aggregate Scores")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("⭐ Overall",    f"{summary['overall_score']} / 5")
            c2.metric("🎯 Faithfulness",     f"{summary['faithfulness']} / 5")
            c3.metric("💬 Relevancy",         f"{summary['answer_relevancy']} / 5")
            c4.metric("🔍 Retrieval",         f"{summary['retrieval_relevance']} / 5")
            c5.metric("📖 Coverage",          f"{summary['context_coverage']} / 5")

            st.divider()
            st.markdown("#### 📋 Per-Question Breakdown")
            for r in details:
                with st.expander(f"**{r['question'][:70]}** — Score: {r['scores']['overall_score']}/5"):
                    st.markdown(f"**Answer:** {r['answer'][:400]}…")
                    st.markdown(
                        f'<span class="score-badge">Faith: {r["scores"]["faithfulness"]}</span>'
                        f'<span class="score-badge">Rel: {r["scores"]["answer_relevancy"]}</span>'
                        f'<span class="score-badge">Retrieval: {r["scores"]["retrieval_relevance"]}</span>'
                        f'<span class="score-badge">Coverage: {r["scores"]["context_coverage"]}</span>',
                        unsafe_allow_html=True,
                    )
                    if r["scores"].get("explanation"):
                        st.caption(r["scores"]["explanation"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### 📚 How This Works

This app uses **Retrieval-Augmented Generation (RAG)** to answer questions about any PDF book.

```
Your PDF
   ↓
CHUNK     Split into ~500-word passages
   ↓
EMBED     Convert each passage → vector using sentence-transformers (free)
   ↓
STORE     Save all vectors in ChromaDB (local database)
   ↓  ─ ─ ─ done once ─ ─ ─
   ↓
QUERY     You ask a question
   ↓
RETRIEVE  Find the 5 most similar passages
   ↓
GENERATE  Send question + passages to GPT-4o
   ↓
ANSWER ✅  Grounded answer with page citations
```

### 🛠️ Tech Stack

| Component | Tool |
|---|---|
| PDF extraction | pdfplumber |
| Embeddings | sentence-transformers (free, local) |
| Vector database | ChromaDB |
| LLM | OpenAI GPT-4o |
| Evaluation | GPT-4o as judge |
| UI | Streamlit |

### 💡 Tips

- **chunk_size 300** = more precise retrieval (good for fact-dense books)
- **chunk_size 700** = more context per passage (good for narrative books)
- **top_k 7-8** = better answers for complex multi-part questions
- Re-index with a different chunk size to improve answer quality
""")
