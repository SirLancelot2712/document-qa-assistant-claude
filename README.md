# Document Q&A Assistant

Ask questions about your own documents (PDF, TXT, Markdown) and get answers that cite the exact passages they came from. Built with Python, Streamlit and the Claude API.

**How it works**
1. Uploaded files are split into overlapping passages.
2. A built-in BM25 search ranks the passages most relevant to your question (no vector database or extra service needed).
3. Only those passages go to Claude, which answers strictly from them and cites `[1]`, `[2]`, ...
4. If the answer isn't in the documents, it says so instead of guessing.

## Run it

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your-key      # Windows (or paste the key in the app sidebar)
streamlit run app.py
```

## Tests

```bash
pytest
```

The tests cover chunking, retrieval and prompt building, and need no API key.

## Typical uses

Internal policy/FAQ lookup, contract and report Q&A, customer-support knowledge bases, onboarding docs.

## Want one for your business?

I build custom document assistants and AI automations. Get in touch via my freelance profile.
