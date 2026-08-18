"""
Deduplication logic for text chunks and documents.
"""

import hashlib


def compute_content_hash(text: str) -> str:
    """Computes SHA-256 hash of normalized text for exact matching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class Deduplicator:
    """Tracks seen hashes to detect duplicates."""
    def __init__(self):
        self.seen_doc_hashes: set[str] = set()
        self.seen_chunk_hashes: set[str] = set()

    def is_duplicate_doc(self, text: str) -> bool:
        """Returns True if the document text was already seen."""
        h = compute_content_hash(text)
        if h in self.seen_doc_hashes:
            return True
        self.seen_doc_hashes.add(h)
        return False

    def is_duplicate_chunk(self, text: str) -> bool:
        """Returns True if the chunk text was already seen."""
        h = compute_content_hash(text)
        if h in self.seen_chunk_hashes:
            return True
        self.seen_chunk_hashes.add(h)
        return False
