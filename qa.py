"""Core logic for the document Q&A assistant: load, chunk, retrieve, ask Claude."""
from __future__ import annotations

import io
import math
import os
import re
from collections import Counter
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "for", "on", "with",
         "that", "this", "it", "as", "by", "be", "at", "from", "what", "which", "who", "how", "do",
         "does", "i", "you", "we", "they", "can", "about", "me"}


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def load_text(filename: str, data: bytes) -> str:
    """Extract text from an uploaded .pdf, .txt or .md file."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(text: str, source: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split text into overlapping character windows, preferring paragraph/sentence breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start, i = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, source, i))
            i += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _tokens(s: str) -> list[str]:
    return [w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 1]


def retrieve(question: str, chunks: list[Chunk], k: int = 4) -> list[Chunk]:
    """Rank chunks against the question with a small BM25 scorer (no external services)."""
    q = _tokens(question)
    if not q or not chunks:
        return chunks[:k]
    docs = [Counter(_tokens(c.text)) for c in chunks]
    lengths = [sum(d.values()) or 1 for d in docs]
    avg = sum(lengths) / len(lengths)
    n = len(chunks)
    k1, b = 1.5, 0.75
    scores = []
    for d, ln in zip(docs, lengths):
        s = 0.0
        for term in set(q):
            f = d.get(term, 0)
            if not f:
                continue
            df = sum(1 for x in docs if term in x)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / avg))
        scores.append(s)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:k] if scores[i] > 0] or chunks[:k]


SYSTEM = (
    "You answer questions using ONLY the provided document excerpts. "
    "Cite the excerpt numbers you used like [1] or [2]. "
    "If the excerpts do not contain the answer, say you could not find it in the document. "
    "Be concise and do not invent facts."
)


def build_prompt(question: str, context: list[Chunk]) -> str:
    excerpts = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, c in enumerate(context, 1))
    return f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def ask(question: str, context: list[Chunk], api_key: str | None = None, model: str = DEFAULT_MODEL) -> str:
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No API key. Set ANTHROPIC_API_KEY or paste it in the sidebar.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, context)}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")
