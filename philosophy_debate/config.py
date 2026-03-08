from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    key: str
    display_name: str
    folder_name: str
    agent_name: str
    tradition_note: str
    accent_color: str

    @property
    def corpus_dir(self) -> Path:
        return PROJECT_ROOT / self.folder_name

    @property
    def storage_dir(self) -> Path:
        return PROJECT_ROOT / "storage" / self.key


@dataclass(frozen=True, slots=True)
class AppConfig:
    project_root: Path
    debate_model: str
    narrator_model: str
    embedding_model: str
    embedding_device: str
    default_rounds: int
    top_k_results: int
    min_direct_text_chars: int
    ocr_language: str
    tesseract_cmd: str | None
    corpora: dict[str, CorpusSpec]


def get_config() -> AppConfig:
    corpora = {
        "stoic": CorpusSpec(
            key="stoic",
            display_name="Stoicism Corpus",
            folder_name="Stoicism Corpus",
            agent_name="Stoic Agent",
            tradition_note=(
                "A disciplined, practical voice grounded in Stoic ethics, self-mastery, "
                "virtue, and rational clarity."
            ),
            accent_color="#A14A1A",
        ),
        "vedanta": CorpusSpec(
            key="vedanta",
            display_name="Vedanta Corpus",
            folder_name="Vedanta corpus",
            agent_name="Vedantam Agent",
            tradition_note=(
                "A reflective, metaphysical voice grounded in Vedantic inquiry, self-knowledge, "
                "unity, liberation, and consciousness."
            ),
            accent_color="#1F5B6E",
        ),
        "machiavellian": CorpusSpec(
            key="machiavellian",
            display_name="Machiavellianism Corpus",
            folder_name="Machiavellianism Corpus",
            agent_name="Machiavellianism Agent",
            tradition_note=(
                "A hard-headed strategic voice grounded in political realism, power dynamics, "
                "statecraft, persuasion, and practical advantage."
            ),
            accent_color="#3D4F2A",
        ),
    }

    return AppConfig(
        project_root=PROJECT_ROOT,
        debate_model=os.getenv("DEBATE_MODEL", "llama-3.1-8b-instant"),
        narrator_model=os.getenv("NARRATOR_MODEL", "llama-3.1-8b-instant"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        default_rounds=_get_int_env("DEFAULT_DEBATE_ROUNDS", 4),
        top_k_results=_get_int_env("TOP_K_RESULTS", 4),
        min_direct_text_chars=_get_int_env("MIN_DIRECT_TEXT_CHARS", 1200),
        ocr_language=os.getenv("OCR_LANGUAGE", "eng"),
        tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
        corpora=corpora,
    )