"""
Adapters for reading different corpus file formats.
"""

import csv
import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Any


class CorpusAdapter(ABC):
    @abstractmethod
    def iter_documents(self, source_path: Path) -> Generator[dict[str, Any], None, None]:
        """
        Yields raw document dictionaries.
        Expected keys:
        - doc_id: str
        - text: str
        - title: Optional[str]
        - metadata: dict
        - original_hash: str
        """
        pass

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

class TextAdapter(CorpusAdapter):
    def iter_documents(self, source_path: Path) -> Generator[dict[str, Any], None, None]:
        if source_path.is_file():
            yield self._parse_file(source_path)
        else:
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.endswith(".txt"):
                        yield self._parse_file(Path(root) / file)

    def _parse_file(self, path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return {
            "doc_id": path.name,
            "text": text,
            "title": path.name,
            "metadata": {"source_type": "txt", "path": str(path.name)},
            "original_hash": self._file_hash(path),
        }

class JsonlAdapter(CorpusAdapter):
    def __init__(self, text_field: str = "text", id_field: str = "id", title_field: str = "title"):
        self.text_field = text_field
        self.id_field = id_field
        self.title_field = title_field

    def iter_documents(self, source_path: Path) -> Generator[dict[str, Any], None, None]:
        if not source_path.is_file():
            raise ValueError("JsonlAdapter expects a file path")

        file_hash = self._file_hash(source_path)
        with open(source_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(self.text_field, "")
                    doc_id = str(data.get(self.id_field, f"{source_path.name}_{i}"))
                    title = data.get(self.title_field)
                    yield {
                        "doc_id": doc_id,
                        "text": text,
                        "title": title,
                        "metadata": {k: v for k, v in data.items() if k not in [self.text_field]},
                        "original_hash": file_hash,
                    }
                except json.JSONDecodeError:
                    pass

class PdfAdapter(CorpusAdapter):
    def iter_documents(self, source_path: Path) -> Generator[dict[str, Any], None, None]:
        import PyPDF2

        if source_path.is_file():
            yield self._parse_file(source_path, PyPDF2)
        else:
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        yield self._parse_file(Path(root) / file, PyPDF2)

    def _parse_file(self, path: Path, pypdf2_module) -> dict[str, Any]:
        text = ""
        try:
            with open(path, "rb") as f:
                reader = pypdf2_module.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            pass

        return {
            "doc_id": path.name,
            "text": text,
            "title": path.name,
            "metadata": {"source_type": "pdf", "path": str(path.name)},
            "original_hash": self._file_hash(path),
        }

class HtmlAdapter(CorpusAdapter):
    def iter_documents(self, source_path: Path) -> Generator[dict[str, Any], None, None]:
        from bs4 import BeautifulSoup

        if source_path.is_file():
            yield self._parse_file(source_path, BeautifulSoup)
        else:
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith((".html", ".htm")):
                        yield self._parse_file(Path(root) / file, BeautifulSoup)

    def _parse_file(self, path: Path, bs_module) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                soup = bs_module(f.read(), "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                title = soup.title.string if soup.title else path.name
        except Exception:
            text = ""
            title = path.name

        return {
            "doc_id": path.name,
            "text": text,
            "title": title,
            "metadata": {"source_type": "html", "path": str(path.name)},
            "original_hash": self._file_hash(path),
        }

class CsvAdapter(CorpusAdapter):
    def __init__(self, text_field: str = "text", id_field: str = "id", title_field: str = "title"):
        self.text_field = text_field
        self.id_field = id_field
        self.title_field = title_field

    def iter_documents(self, source_path: Path) -> Generator[dict[str, Any], None, None]:
        if not source_path.is_file():
            raise ValueError("CsvAdapter expects a file path")

        file_hash = self._file_hash(source_path)
        with open(source_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get(self.text_field, "")
                doc_id = str(row.get(self.id_field, f"{source_path.name}_{i}"))
                title = row.get(self.title_field)
                yield {
                    "doc_id": doc_id,
                    "text": text,
                    "title": title,
                    "metadata": {k: v for k, v in row.items() if k not in [self.text_field]},
                    "original_hash": file_hash,
                }
