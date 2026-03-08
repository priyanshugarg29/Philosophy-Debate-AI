from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone

if sys.version_info >= (3, 14):
    raise RuntimeError(
        "This app currently needs Python 3.11 or 3.12 for the Chroma stack. On Streamlit Community Cloud, redeploy the app with Python 3.12 in Advanced settings."
    )

from langchain_chroma import Chroma
from langchain_core.documents import Document

from philosophy_debate.config import CorpusSpec
from philosophy_debate.document_processing import (
    PDFTextExtractor,
    list_pdf_files,
    normalize_whitespace,
)
from philosophy_debate.llm import LocalEmbeddingService
from philosophy_debate.models import IndexBuildReport, SearchResult, TextChunk


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_text(text: str, chunk_chars: int = 1400, overlap_chars: int = 250) -> list[str]:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_chars)
        if end < len(cleaned):
            split_at = cleaned.rfind("\n", start + (chunk_chars // 2), end)
            if split_at == -1:
                split_at = cleaned.rfind(". ", start + (chunk_chars // 2), end)
            if split_at > start:
                end = split_at + 1

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


class CorpusKnowledgeBase:
    def __init__(
        self,
        corpus: CorpusSpec,
        embedder: LocalEmbeddingService,
        vectorstore: Chroma,
        report: IndexBuildReport,
    ) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.report = report

    @classmethod
    def load_or_build(
        cls,
        corpus: CorpusSpec,
        embedder: LocalEmbeddingService,
        extractor: PDFTextExtractor,
        *,
        force_rebuild: bool = False,
    ) -> "CorpusKnowledgeBase":
        manifest_path = corpus.storage_dir / "manifest.json"
        chroma_dir = corpus.storage_dir / "chroma"

        if not force_rebuild and manifest_path.exists() and chroma_dir.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if cls._is_manifest_fresh(corpus, manifest, embedder):
                return cls._load(corpus, embedder, manifest)

        return cls._build(corpus, embedder, extractor)

    @staticmethod
    def _collection_name(corpus: CorpusSpec) -> str:
        return f"{corpus.key}_corpus"

    @staticmethod
    def _current_file_signature(corpus: CorpusSpec) -> list[dict[str, object]]:
        signature = []
        for pdf_path in list_pdf_files(corpus.corpus_dir):
            stat = pdf_path.stat()
            signature.append(
                {
                    "path": pdf_path.name,
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                }
            )
        return signature

    @classmethod
    def _is_manifest_fresh(
        cls,
        corpus: CorpusSpec,
        manifest: dict[str, object],
        embedder: LocalEmbeddingService,
    ) -> bool:
        return (
            manifest.get("files") == cls._current_file_signature(corpus)
            and manifest.get("embedding_model") == embedder.model_name
        )

    @classmethod
    def _load(
        cls,
        corpus: CorpusSpec,
        embedder: LocalEmbeddingService,
        manifest: dict[str, object],
    ) -> "CorpusKnowledgeBase":
        vectorstore = Chroma(
            collection_name=cls._collection_name(corpus),
            persist_directory=str(corpus.storage_dir / "chroma"),
            embedding_function=embedder.embedding_model,
        )
        report = IndexBuildReport(
            corpus_key=corpus.key,
            display_name=corpus.display_name,
            source_count=int(manifest.get("source_count", 0)),
            chunk_count=int(manifest.get("chunk_count", 0)),
            warnings=list(manifest.get("warnings", [])),
            used_ocr=bool(manifest.get("used_ocr", False)),
            built_at=str(manifest.get("built_at", "")),
            loaded_from_cache=True,
        )
        return cls(corpus=corpus, embedder=embedder, vectorstore=vectorstore, report=report)

    @classmethod
    def _build(
        cls,
        corpus: CorpusSpec,
        embedder: LocalEmbeddingService,
        extractor: PDFTextExtractor,
    ) -> "CorpusKnowledgeBase":
        pdf_files = list_pdf_files(corpus.corpus_dir)
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files were found in {corpus.corpus_dir}.")

        corpus.storage_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        documents: list[Document] = []
        document_ids: list[str] = []
        used_ocr = False
        chunk_count = 0

        for pdf_path in pdf_files:
            document = extractor.extract(pdf_path)
            warnings.extend(document.warnings)
            used_ocr = used_ocr or document.extractor == "ocr"

            for chunk_index, chunk_text in enumerate(_chunk_text(document.text), start=1):
                chunk_count += 1
                chunk_id = f"{corpus.key}-{document.source_id}-{chunk_index}"
                documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_id": chunk_id,
                            "corpus_key": corpus.key,
                            "title": document.title,
                            "file_path": document.file_path,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                document_ids.append(chunk_id)

        if not documents:
            raise RuntimeError(f"No searchable chunks were created for {corpus.display_name}.")

        chroma_dir = corpus.storage_dir / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)

        # Chroma keeps each tradition in its own persisted vector index.
        vectorstore = Chroma(
            collection_name=cls._collection_name(corpus),
            persist_directory=str(chroma_dir),
            embedding_function=embedder.embedding_model,
        )
        vectorstore.add_documents(documents=documents, ids=document_ids)

        manifest = {
            "corpus_key": corpus.key,
            "display_name": corpus.display_name,
            "embedding_model": embedder.model_name,
            "built_at": _utc_now(),
            "source_count": len(pdf_files),
            "chunk_count": chunk_count,
            "used_ocr": used_ocr,
            "warnings": sorted(set(warnings)),
            "files": cls._current_file_signature(corpus),
        }

        with (corpus.storage_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        report = IndexBuildReport(
            corpus_key=corpus.key,
            display_name=corpus.display_name,
            source_count=len(pdf_files),
            chunk_count=chunk_count,
            warnings=sorted(set(warnings)),
            used_ocr=used_ocr,
            built_at=manifest["built_at"],
            loaded_from_cache=False,
        )
        return cls(corpus=corpus, embedder=embedder, vectorstore=vectorstore, report=report)

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        results = self.vectorstore.similarity_search_with_score(query=query, k=top_k)
        search_results: list[SearchResult] = []
        for document, distance in results:
            metadata = document.metadata
            score = 1.0 / (1.0 + float(distance))
            search_results.append(
                SearchResult(
                    chunk=TextChunk(
                        chunk_id=str(metadata.get("chunk_id", "")),
                        corpus_key=str(metadata.get("corpus_key", self.corpus.key)),
                        title=str(metadata.get("title", "Unknown Source")),
                        file_path=str(metadata.get("file_path", "")),
                        text=document.page_content,
                        chunk_index=int(metadata.get("chunk_index", 0)),
                    ),
                    score=score,
                )
            )
        return search_results