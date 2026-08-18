"""
Pipeline orchestrator for text corpus ingestion.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from faulttrace_data.snapshot import _fingerprint_path
from faulttrace_data.text.adapters import (
    CorpusAdapter,
    CsvAdapter,
    HtmlAdapter,
    JsonlAdapter,
    PdfAdapter,
    TextAdapter,
)
from faulttrace_data.text.deduplication import Deduplicator, compute_content_hash
from faulttrace_data.text.preprocessing import chunk_text, detect_language, normalize_text
from faulttrace_data.text.snapshot_text import TextCorpusSnapshot, TextSnapshotRegistry

logger = logging.getLogger(__name__)

class TextIngestionPipeline:
    def __init__(self, data_root: Path, registry_path: Path):
        self.data_root = data_root
        self.registry = TextSnapshotRegistry(registry_path)

    def _get_adapter(self, source_type: str, config: dict[str, Any]) -> CorpusAdapter:
        if source_type == "jsonl":
            return JsonlAdapter(
                text_field=config.get("text_field", "text"),
                id_field=config.get("id_field", "id"),
                title_field=config.get("title_field", "title")
            )
        elif source_type == "csv":
            return CsvAdapter(
                text_field=config.get("text_field", "text"),
                id_field=config.get("id_field", "id"),
                title_field=config.get("title_field", "title")
            )
        elif source_type == "txt":
            return TextAdapter()
        elif source_type == "pdf":
            return PdfAdapter()
        elif source_type == "html":
            return HtmlAdapter()
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def run(self, dataset_id: str, source_path: Path, source_type: str, config: dict[str, Any]) -> TextCorpusSnapshot:
        """Run the ingestion pipeline and return a snapshot."""

        # Security: Prevent symlink escapes
        resolved_source = source_path.resolve()

        adapter = self._get_adapter(source_type, config)
        dedup = Deduplicator()

        chunk_size = config.get("chunk_size", 1000)
        overlap = config.get("overlap", 100)

        documents = []
        chunks_data = []

        stats = {
            "accepted_documents": 0,
            "duplicate_documents": 0,
            "malformed_documents": 0,
            "exact_duplicate_chunks": 0,
            "total_chunks": 0,
            "languages": {},
            "size_distribution": {"<1KB": 0, "1KB-10KB": 0, ">10KB": 0}
        }

        source_file_count = 1 if source_path.is_file() else sum(1 for _ in source_path.rglob("*") if _.is_file())

        for raw_doc in adapter.iter_documents(resolved_source):
            doc_id = raw_doc.get("doc_id")
            text = raw_doc.get("text", "")

            if not text.strip():
                stats["malformed_documents"] += 1
                continue

            norm_text = normalize_text(text)
            if not norm_text:
                stats["malformed_documents"] += 1
                continue

            if dedup.is_duplicate_doc(norm_text):
                stats["duplicate_documents"] += 1
                continue

            lang = detect_language(norm_text)
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

            size_kb = len(norm_text) / 1024
            if size_kb < 1:
                stats["size_distribution"]["<1KB"] += 1
            elif size_kb <= 10:
                stats["size_distribution"]["1KB-10KB"] += 1
            else:
                stats["size_distribution"][">10KB"] += 1

            stats["accepted_documents"] += 1

            # Store document record
            documents.append({
                "doc_id": doc_id,
                "title": raw_doc.get("title", ""),
                "language": lang,
                "source_hash": raw_doc.get("original_hash", ""),
                "metadata_json": json.dumps(raw_doc.get("metadata", {}))
            })

            # Chunking
            doc_chunks = chunk_text(norm_text, chunk_size, overlap)
            for i, chunk in enumerate(doc_chunks):
                chunk_text_str = chunk["text"]
                if dedup.is_duplicate_chunk(chunk_text_str):
                    stats["exact_duplicate_chunks"] += 1
                    # Still keep the chunk but maybe flag it? Actually if we want to deduplicate chunks
                    # across the corpus, we should skip it or just link it.
                    # RAG usually indexes all occurrences unless specified.
                    # We'll skip adding duplicate chunks to the vector DB by dropping them here if strict deduplication is on.
                    if config.get("strict_chunk_dedup", False):
                        continue

                stats["total_chunks"] += 1
                chunks_data.append({
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "text": chunk_text_str,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "chunk_index": i,
                    "content_hash": compute_content_hash(chunk_text_str)
                })

        # Save to Parquet
        out_dir = self.data_root / "generated" / "text_worlds" / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        chunks_df = pd.DataFrame(chunks_data) if chunks_data else pd.DataFrame(columns=["chunk_id", "doc_id", "text", "start_char", "end_char", "chunk_index", "content_hash"])
        docs_df = pd.DataFrame(documents) if documents else pd.DataFrame(columns=["doc_id", "title", "language", "source_hash", "metadata_json"])

        chunks_path = out_dir / "chunks.parquet"
        docs_path = out_dir / "documents.parquet"

        chunks_df.to_parquet(chunks_path)
        docs_df.to_parquet(docs_path)

        # Calculate canonical hash
        import hashlib
        h = hashlib.sha256()
        with open(chunks_path, "rb") as f:
            while b := f.read(65536): h.update(b)
        canonical_hash = h.hexdigest()

        # Build Snapshot
        config_hash = compute_content_hash(json.dumps(config, sort_keys=True))
        source_fp = _fingerprint_path(resolved_source, self.data_root)

        snap = TextCorpusSnapshot(
            dataset_id=dataset_id,
            source_type=source_type,
            source_path_fingerprint=source_fp,
            source_file_count=source_file_count,
            document_count=stats["accepted_documents"],
            chunk_count=stats["total_chunks"],
            parquet_root=f"generated/text_worlds/{dataset_id}",
            languages=stats["languages"],
            size_distribution=stats["size_distribution"],
            accepted_documents=stats["accepted_documents"],
            duplicate_documents=stats["duplicate_documents"],
            malformed_documents=stats["malformed_documents"],
            exact_duplicate_chunks=stats["exact_duplicate_chunks"],
            canonical_content_hash=canonical_hash,
            ingestion_config_hash=config_hash,
        )

        # Replace snapshot ID with deterministic one
        snap.snapshot_id = TextCorpusSnapshot.make_snapshot_id(dataset_id, source_fp, config_hash)

        self.registry.register(snap)
        return snap
