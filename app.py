"""Streamlit UI: upload documents, ask questions, get cited answers from Claude."""
import streamlit as st

import qa

st.set_page_config(page_title="Document Q&A Assistant", page_icon="💬", layout="wide")
st.title("💬 Document Q&A Assistant")
st.caption("Upload PDF/TXT/MD files and ask questions. Answers are grounded in your documents and cite their sources.")

with st.sidebar:
    st.header("Setup")
    provider = st.selectbox("Answer engine", qa.PROVIDERS)
    api_key = None
    if provider.startswith("Gemini"):
        api_key = st.text_input("Gemini API key (free)", type="password",
                                help="Get one free at aistudio.google.com/apikey. Used only for this session.")
    elif provider == "Claude":
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Used only for this session. Or set ANTHROPIC_API_KEY in your environment.")
    files = st.file_uploader("Documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    top_k = st.slider("Excerpts sent to Claude", 2, 8, 4)

if "history" not in st.session_state:
    st.session_state.history = []

chunks: list[qa.Chunk] = []
for f in files or []:
    try:
        chunks += qa.chunk_text(qa.load_text(f.name, f.getvalue()), source=f.name)
    except Exception as exc:  # unreadable/unsupported file
        st.sidebar.error(f"{f.name}: {exc}")

if files:
    st.sidebar.success(f"{len(files)} file(s), {len(chunks)} searchable passages")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        with st.expander("Sources"):
            for i, c in enumerate(turn["ctx"], 1):
                st.markdown(f"**[{i}] {c.source}**")
                st.caption(c.text)

question = st.chat_input("Ask a question about your documents")
if question:
    if not chunks:
        st.warning("Upload at least one document first.")
    else:
        with st.spinner("Thinking..."):
            ctx = qa.retrieve(question, chunks, k=top_k)
            try:
                answer = qa.answer(provider, question, ctx, api_key=api_key or None)
            except Exception as exc:
                st.error(f"Could not get an answer: {exc}")
                st.stop()
        st.session_state.history.append({"q": question, "a": answer, "ctx": ctx})
        st.rerun()
