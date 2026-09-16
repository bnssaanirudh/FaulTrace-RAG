"""
Preprocessing and chunking logic.
"""

import re
import unicodedata
from typing import TypedDict


class TextChunk(TypedDict):
    text: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    """Normalize unicode, clean whitespace, and strip boilerplate."""
    if not text:
        return ""
    # Normalize unicode to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except tab, newline, carriage return
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\t", "\n", "\r"))
    # Condense multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Condense multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def detect_language(text: str) -> str:
    """Basic mock language detection."""
    # In a real app we might use langdetect or fasttext.
    # We will assume 'en' for now since the prompt doesn't specify an external dep for this.
    return "en"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[TextChunk]:
    """
    Split text into chunks of `chunk_size` characters with `overlap`.
    Returns list of {"text": chunk, "start_char": start, "end_char": end}.
    """
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # If not the end, try to break at a natural boundary (newline or space)
        if end < text_len:
            # Look backwards up to 100 characters for a newline or space
            break_point = -1
            for separator in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(separator, start, end)
                if pos != -1 and pos > start + (chunk_size // 2):
                    break_point = pos + len(separator)
                    break

            if break_point != -1:
                end = break_point
        else:
            end = text_len

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "text": chunk_text_str,
                "start_char": start,
                "end_char": end,
            })

        if end >= text_len:
            break

        start = end - overlap
        # Prevent infinite loop if overlap is too large relative to progress
        if start <= chunks[-1]["start_char"]:
            start = chunks[-1]["start_char"] + 1

    return chunks
