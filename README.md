# 📚 RAG Book Q&A — Powered by OpenAI GPT-4o

Ask questions about any PDF book and get intelligent, page-cited answers using **Retrieval-Augmented Generation (RAG)** + **GPT-4o**.

Includes a full **web UI** (Streamlit) and a **CLI** interface.

---

## 🖼️ UI Preview

The web app has 3 tabs:
- **💬 Chat** — Upload PDF, ask questions, see source passages
- **🧪 Evaluation** — Score your RAG pipeline on faithfulness, relevancy, coverage
- **ℹ️ About** — How it works

---

## 🗂️ File Structure

```
rag-book-qa/
├── app.py               ← Streamlit web UI  ⭐ START HERE
├── main.py              ← CLI alternative
├── evaluate.py          ← Run evaluation suite
├── requirements.txt
├── .env.example
├── src/
│   ├── pdf_processor.py ← Extract + chunk PDF
│   ├── embeddings.py    ← Text → vectors (free local model)
│   ├── vector_store.py  ← ChromaDB store + search
│   ├── rag_pipeline.py  ← Retrieve + GPT-4o generate
│   └── evaluator.py     ← Score with GPT-4o as judge
└── data/
    └── book.pdf         ← Your PDF goes here
```

---

## 🚀 Setup (GitHub Codespaces — No Install Needed)

### 1. Open Codespaces
- Go to your GitHub repo
- Click **Code → Codespaces → Create codespace on main**

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```
Get your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 4. Upload your PDF
- In the file panel, right-click the `data/` folder → **Upload**
- Upload your PDF and rename it:
```bash
mv data/YourBook.pdf data/book.pdf
```

### 5. Launch the web UI 🎉
```bash
streamlit run app.py
```
Codespaces will show a popup — click **"Open in Browser"**

---

## 💻 CLI Usage (Alternative to UI)

```bash
# Index the book
python main.py index --pdf data/book.pdf

# Ask a question
python main.py ask "How do I build wealth?"

# Interactive chat
python main.py chat

# Run evaluation
python evaluate.py
```

---

## 🧪 RAG Evaluation Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved text? |
| Answer Relevancy | Does the answer address the question? |
| Retrieval Relevance | Are the right passages being found? |
| Context Coverage | Does the answer use all available context? |

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| PDF extraction | pdfplumber |
| Embeddings | sentence-transformers (free, local) |
| Vector DB | ChromaDB |
| LLM | OpenAI GPT-4o |
| UI | Streamlit |

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Run `pip install -r requirements.txt` again |
| `OPENAI_API_KEY not found` | Check `.env` exists and has the right key |
| `Vector store is empty` | Index the PDF first via the UI sidebar or CLI |
| Poor answer quality | Try `top_k=7` or smaller `chunk_size=300` |
