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


def test_extractive_answer_needs_no_key():
    chunks = qa.chunk_text(DOC, "doc.txt", size=120, overlap=20)
    out = qa.answer("No key (show passages)", "refund policy", qa.retrieve("refund policy", chunks, k=2))
    assert "refund" in out.lower()


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Gemini API key"):
        qa.ask_gemini("q", qa.chunk_text(DOC, "d.txt"))


def test_gemini_parses_response_and_sends_key_in_header(monkeypatch):
    import io
    import json

    seen = {}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen["key"] = req.get_header("X-goog-api-key")
        seen["url"] = req.full_url
        payload = {"candidates": [{"content": {"parts": [{"text": "30 days [1]"}]}}]}
        return FakeResp(json.dumps(payload).encode())

    monkeypatch.setattr(qa.urllib.request, "urlopen", fake_urlopen)
    out = qa.ask_gemini("refund window?", qa.chunk_text(DOC, "d.txt"), api_key="test-key")
    assert out == "30 days [1]"
    assert seen["key"] == "test-key"
    assert "test-key" not in seen["url"]  # key must never be placed in the URL
