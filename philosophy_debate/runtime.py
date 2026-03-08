from __future__ import annotations

import os
from dataclasses import dataclass

from philosophy_debate.config import AppConfig, get_config
from philosophy_debate.debate import DebateAgent, DebateOrchestrator
from philosophy_debate.document_processing import PDFTextExtractor
from philosophy_debate.llm import GroqTextGenerator, LocalEmbeddingService
from philosophy_debate.models import IndexBuildReport
from philosophy_debate.retrieval import CorpusKnowledgeBase


@dataclass(slots=True)
class RuntimeBundle:
    config: AppConfig
    reports: dict[str, IndexBuildReport]
    debate_service: DebateOrchestrator


def build_runtime(*, force_rebuild: bool = False) -> RuntimeBundle:
    config = get_config()

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

    extractor = PDFTextExtractor(
        min_direct_text_chars=config.min_direct_text_chars,
        ocr_language=config.ocr_language,
        tesseract_cmd=config.tesseract_cmd,
    )
    embedder = LocalEmbeddingService(
        model_name=config.embedding_model,
        device=config.embedding_device,
    )

    knowledge_bases: dict[str, CorpusKnowledgeBase] = {}
    reports: dict[str, IndexBuildReport] = {}

    for corpus_key, corpus in config.corpora.items():
        if not corpus.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus.corpus_dir}")
        knowledge_base = CorpusKnowledgeBase.load_or_build(
            corpus=corpus,
            embedder=embedder,
            extractor=extractor,
            force_rebuild=force_rebuild,
        )
        knowledge_bases[corpus_key] = knowledge_base
        reports[corpus_key] = knowledge_base.report

    debate_llm = GroqTextGenerator(model=config.debate_model)
    narrator_llm = GroqTextGenerator(model=config.narrator_model)

    agents = {
        "stoic": DebateAgent(
            key="stoic",
            name=config.corpora["stoic"].agent_name,
            worldview=(
                "You represent the Stoic tradition. Emphasize virtue, rational discipline, "
                "freedom through inner governance, and practical ethics for daily life."
            ),
            guidance=config.corpora["stoic"].tradition_note,
            knowledge_base=knowledge_bases["stoic"],
        ),
        "vedanta": DebateAgent(
            key="vedanta",
            name=config.corpora["vedanta"].agent_name,
            worldview=(
                "You represent the Vedanta tradition. Emphasize self-knowledge, liberation, "
                "the relation between the self and ultimate reality, and the inward search for truth."
            ),
            guidance=config.corpora["vedanta"].tradition_note,
            knowledge_base=knowledge_bases["vedanta"],
        ),
        "machiavellian": DebateAgent(
            key="machiavellian",
            name=config.corpora["machiavellian"].agent_name,
            worldview=(
                "You represent the Machiavellian tradition. Emphasize political realism, strategy, "
                "power, leverage, reputation management, and the hard constraints of human conflict."
            ),
            guidance=config.corpora["machiavellian"].tradition_note,
            knowledge_base=knowledge_bases["machiavellian"],
        ),
    }

    debate_service = DebateOrchestrator(
        agents=agents,
        debate_llm=debate_llm,
        narrator_llm=narrator_llm,
        default_top_k=config.top_k_results,
    )
    return RuntimeBundle(config=config, reports=reports, debate_service=debate_service)