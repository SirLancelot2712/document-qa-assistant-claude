import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import qa

DOC = (
    "Our refund policy allows returns within 30 days of purchase. Items must be unused.\n\n"
    "Shipping takes 3 to 5 business days inside the country. International shipping takes 10 days.\n\n"
    "Support is available Monday to Friday from 9am to 6pm by email and live chat."
)


def test_chunking_covers_text_and_respects_size():
    chunks = qa.chunk_text(DOC * 10, "doc.txt", size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_retrieval_finds_relevant_chunk():
    chunks = qa.chunk_text(DOC, "doc.txt", size=120, overlap=20)
    top = qa.retrieve("How long does international shipping take?", chunks, k=1)[0]
    assert "International shipping" in top.text


def test_retrieval_handles_empty_inputs():
    assert qa.retrieve("anything", [], k=3) == []


def test_load_text_txt_and_unsupported():
    assert qa.load_text("a.txt", b"hello") == "hello"
    with pytest.raises(ValueError):
        qa.load_text("a.exe", b"x")


def test_prompt_contains_numbered_excerpts_and_question():
    chunks = qa.chunk_text(DOC, "doc.txt")
    prompt = qa.build_prompt("What is the refund window?", chunks)
    assert "[1]" in prompt and "What is the refund window?" in prompt
