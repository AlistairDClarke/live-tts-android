import re
from dataclasses import dataclass, field

_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u2035": "'",
    "\u00ab": '"', "\u00bb": '"', "\u201e": '"', "\u201f": '"',
}

def _normalize_quotes(text: str) -> str:
    for k, v in _SMART_QUOTES.items():
        text = text.replace(k, v)
    return text


@dataclass
class TextChunk:
    text: str
    position: int
    end_position: int
    index: int


SENTENCE_END = re.compile(r"(?<=[.!?])(?=[\"']?\s+[A-Z\"'=#])")
WORD_BOUNDARY = re.compile(r'\s+')


def split_sentences(text: str) -> list[str]:
    text = _normalize_quotes(text)
    parts = SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chunk_chars: int = 200) -> list[TextChunk]:
    text = _normalize_quotes(text)
    raw = text.split("\n\n")
    paragraphs = [p.strip() for p in raw if p.strip()]
    chunks: list[TextChunk] = []
    pos = 0

    for i, para in enumerate(paragraphs):
        actual_pos = text.find(para, pos)
        if actual_pos == -1:
            actual_pos = pos
        chunks.append(TextChunk(
            text=para,
            position=actual_pos,
            end_position=actual_pos + len(para),
            index=i,
        ))
        pos = actual_pos + len(para)

    return chunks


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'(["''])\1+', r'\1', text)
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    return text


def find_chunk_at_position(chunks: list[TextChunk], position: int) -> int:
    for i, chunk in enumerate(chunks):
        if chunk.position <= position < chunk.end_position:
            return i
    for i, chunk in enumerate(chunks):
        if chunk.position >= position:
            return i
    return len(chunks) - 1
