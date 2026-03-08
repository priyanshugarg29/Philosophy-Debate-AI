from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

SpeakerRole = Literal["agent", "narrator", "moderator"]


@dataclass(slots=True)
class ExtractedDocument:
    source_id: str
    title: str
    file_path: str
    text: str
    extractor: str
    page_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    corpus_key: str
    title: str
    file_path: str
    text: str
    chunk_index: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TextChunk":
        return cls(**payload)


@dataclass(slots=True)
class SearchResult:
    chunk: TextChunk
    score: float


@dataclass(slots=True)
class DebateTurn:
    speaker: str
    role: SpeakerRole
    phase: str
    round_number: int
    text: str
    citations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IndexBuildReport:
    corpus_key: str
    display_name: str
    source_count: int
    chunk_count: int
    warnings: list[str]
    used_ocr: bool
    built_at: str
    loaded_from_cache: bool
