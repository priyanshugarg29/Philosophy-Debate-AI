from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator

from philosophy_debate.llm import GroqTextGenerator
from philosophy_debate.models import DebateTurn, SearchResult
from philosophy_debate.retrieval import CorpusKnowledgeBase


@dataclass(slots=True)
class DebateAgent:
    key: str
    name: str
    worldview: str
    guidance: str
    knowledge_base: CorpusKnowledgeBase


@dataclass(slots=True)
class AgentScore:
    agent: str
    coherence: int
    responsiveness: int
    evidence: int
    persuasion: int
    total: int
    comment: str


class DebateOrchestrator:
    def __init__(
        self,
        agents: dict[str, DebateAgent],
        debate_llm: GroqTextGenerator,
        narrator_llm: GroqTextGenerator,
        default_top_k: int = 4,
    ) -> None:
        self.agents = agents
        self.agent_order = list(agents.keys())
        self.debate_llm = debate_llm
        self.narrator_llm = narrator_llm
        self.default_top_k = default_top_k

    def debate(
        self,
        topic: str,
        *,
        rounds: int,
        top_k: int | None = None,
    ) -> Iterator[DebateTurn]:
        transcript: list[DebateTurn] = []
        debate_memory = "No debate memory yet."
        top_k = top_k or self.default_top_k

        moderator_intro = DebateTurn(
            speaker="Moderator",
            role="moderator",
            phase="opening",
            round_number=0,
            text=(
                f'Topic: "{topic}"\n\n'
                f"Participants: {', '.join(self.agents[agent_key].name for agent_key in self.agent_order)}\n\n"
                "The discussion will move through opening statements, direct rebuttals, and final reflection. "
                "The goal is not theatrical conflict alone, but a careful attempt to discover whether these "
                "traditions can converge on a shared conclusion."
            ),
        )
        yield moderator_intro

        for agent_key in self.agent_order:
            turn = self._generate_agent_turn(
                agent=self.agents[agent_key],
                topic=topic,
                transcript=transcript,
                debate_memory=debate_memory,
                phase="opening",
                round_number=0,
                top_k=top_k,
            )
            transcript.append(turn)
            yield turn

        opening_summary = self._generate_round_summary(
            topic=topic,
            transcript=transcript,
            debate_memory=debate_memory,
            round_number=0,
        )
        yield opening_summary
        debate_memory = self._update_debate_memory(
            topic=topic,
            debate_memory=debate_memory,
            transcript=transcript,
            round_number=0,
        )

        for round_number in range(1, rounds + 1):
            for agent_key in self.agent_order:
                turn = self._generate_agent_turn(
                    agent=self.agents[agent_key],
                    topic=topic,
                    transcript=transcript,
                    debate_memory=debate_memory,
                    phase="rebuttal",
                    round_number=round_number,
                    top_k=top_k,
                )
                transcript.append(turn)
                yield turn

            summary = self._generate_round_summary(
                topic=topic,
                transcript=transcript,
                debate_memory=debate_memory,
                round_number=round_number,
            )
            yield summary
            debate_memory = self._update_debate_memory(
                topic=topic,
                debate_memory=debate_memory,
                transcript=transcript,
                round_number=round_number,
            )

        closing_turns: list[DebateTurn] = []
        for agent_key in self.agent_order:
            closing_turn = self._generate_agent_turn(
                agent=self.agents[agent_key],
                topic=topic,
                transcript=transcript,
                debate_memory=debate_memory,
                phase="closing",
                round_number=rounds + 1,
                top_k=top_k,
            )
            transcript.append(closing_turn)
            closing_turns.append(closing_turn)
            yield closing_turn

        moderator_verdict = self._generate_moderator_verdict(
            topic=topic,
            debate_memory=debate_memory,
            closing_turns=closing_turns,
        )
        yield moderator_verdict

    def _generate_agent_turn(
        self,
        *,
        agent: DebateAgent,
        topic: str,
        transcript: list[DebateTurn],
        debate_memory: str,
        phase: str,
        round_number: int,
        top_k: int,
    ) -> DebateTurn:
        evidence = agent.knowledge_base.search(
            query=self._build_retrieval_query(topic, transcript, debate_memory, agent.name, phase),
            top_k=top_k,
        )
        prompt = self._build_agent_prompt(
            agent=agent,
            topic=topic,
            transcript=transcript,
            debate_memory=debate_memory,
            phase=phase,
            round_number=round_number,
            evidence=evidence,
        )
        text = self.debate_llm.generate(
            instructions=self._agent_instructions(agent),
            prompt=prompt,
            max_output_tokens=520,
        )
        return DebateTurn(
            speaker=agent.name,
            role="agent",
            phase=phase,
            round_number=round_number,
            text=text,
            citations=self._citation_labels(evidence),
        )

    def _generate_round_summary(
        self,
        *,
        topic: str,
        transcript: list[DebateTurn],
        debate_memory: str,
        round_number: int,
    ) -> DebateTurn:
        prompt = (
            f'Topic: "{topic}"\n\n'
            f"Round number: {round_number}\n\n"
            f"Current debate memory:\n{debate_memory}\n\n"
            "Latest round transcript:\n"
            f"{self._format_recent_turns(transcript, max_turns=len(self.agent_order))}\n\n"
            "Write a balanced narrator update using these section labels exactly:\n"
            "Shared Ground:\n"
            "Main Tension:\n"
            "What Changed:\n"
            "Keep it concise but meaningful."
        )
        text = self.narrator_llm.generate(
            instructions=(
                "You are a neutral narrator for a philosophy debate. Track agreement, disagreement, "
                "and how the exchange is evolving without taking sides."
            ),
            prompt=prompt,
            max_output_tokens=260,
        )
        return DebateTurn(
            speaker="Narrator",
            role="narrator",
            phase="summary",
            round_number=round_number,
            text=text,
        )

    def _update_debate_memory(
        self,
        *,
        topic: str,
        debate_memory: str,
        transcript: list[DebateTurn],
        round_number: int,
    ) -> str:
        prompt = (
            f'Topic: "{topic}"\n\n'
            f"Round number: {round_number}\n\n"
            f"Existing memory:\n{debate_memory}\n\n"
            "Latest round transcript:\n"
            f"{self._format_recent_turns(transcript, max_turns=len(self.agent_order))}\n\n"
            "Rewrite the debate memory in a compact form. Preserve only the most important claims, agreements, "
            "disagreements, and shifts in position. Use these section labels exactly and keep the whole answer short:\n"
            "Agent Positions:\n"
            "Agreements:\n"
            "Disagreements:\n"
            "Open Questions:\n"
        )
        return self.narrator_llm.generate(
            instructions=(
                "You maintain the compressed memory of a multi-agent debate. Keep the memory short, accurate, "
                "and useful for future turns."
            ),
            prompt=prompt,
            max_output_tokens=220,
        )

    def _generate_moderator_verdict(
        self,
        *,
        topic: str,
        debate_memory: str,
        closing_turns: list[DebateTurn],
    ) -> DebateTurn:
        agent_names = [self.agents[agent_key].name for agent_key in self.agent_order]
        scores = self._generate_scorecard(
            topic=topic,
            debate_memory=debate_memory,
            closing_turns=closing_turns,
            agent_names=agent_names,
        )
        winner_text = self._build_winner_text(scores)
        score_lines = "\n".join(
            (
                f"- {score.agent} | Coh {score.coherence}/10 | Resp {score.responsiveness}/10 | "
                f"Evidence {score.evidence}/10 | Pers {score.persuasion}/10 | Total {score.total}/40 | "
                f"Comment: {score.comment}"
            )
            for score in scores
        )

        prompt = (
            f'Topic: "{topic}"\n\n'
            f"Debaters: {', '.join(agent_names)}\n\n"
            f"Compressed debate memory:\n{debate_memory}\n\n"
            "Closing statements:\n"
            f"{self._format_turns(closing_turns)}\n\n"
            "Moderator scorecard:\n"
            f"{score_lines}\n\n"
            f"Winner guidance:\n{winner_text}\n\n"
            "Act as the moderator closing the debate. Summarize the debate fairly.\n"
            "Use these section labels exactly:\n"
            "Outcome:\n"
            "Moderator Summary:\n"
            "Final Reflection:\n"
            "Requirements:\n"
            "- the outcome should be short\n"
            "- keep the moderator summary to 3 sentences or less\n"
            "- keep the final reflection to 2 sentences or less\n"
            "- do not use markdown bold markers\n"
        )
        sections_text = self.narrator_llm.generate(
            instructions=(
                "You are the moderator of a philosophy debate. Be fair but decisive. Judge the debate as a whole "
                "without repeating the score lines. Do not invent facts."
            ),
            prompt=prompt,
            max_output_tokens=220,
        )
        sections = self._parse_labeled_sections(
            sections_text,
            labels=["Outcome", "Moderator Summary", "Final Reflection"],
        )
        text = (
            f"Outcome:\n{sections['Outcome']}\n\n"
            f"Moderator Summary:\n{sections['Moderator Summary']}\n\n"
            f"Scorecard:\n{score_lines}\n\n"
            f"Winner:\n{winner_text}\n\n"
            f"Final Reflection:\n{sections['Final Reflection']}"
        )
        return DebateTurn(
            speaker="Moderator",
            role="moderator",
            phase="verdict",
            round_number=closing_turns[-1].round_number if closing_turns else 0,
            text=text,
        )

    def _generate_scorecard(
        self,
        *,
        topic: str,
        debate_memory: str,
        closing_turns: list[DebateTurn],
        agent_names: list[str],
    ) -> list[AgentScore]:
        expected_agents = "\n".join(f"- {agent_name}" for agent_name in agent_names)
        prompt = (
            f'Topic: "{topic}"\n\n'
            f"Debaters:\n{expected_agents}\n\n"
            f"Compressed debate memory:\n{debate_memory}\n\n"
            "Closing statements:\n"
            f"{self._format_turns(closing_turns)}\n\n"
            "Return JSON only. Use this exact schema:\n"
            '{\n'
            '  "scores": [\n'
            '    {\n'
            '      "agent": "Stoic Agent",\n'
            '      "coherence": 8,\n'
            '      "responsiveness": 8,\n'
            '      "evidence": 8,\n'
            '      "persuasion": 8,\n'
            '      "comment": "Short explanation."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Requirements:\n"
            "- include every debater exactly once\n"
            "- every score must be an integer from 1 to 10\n"
            "- keep each comment under 20 words\n"
            "- do not include markdown fences\n"
        )
        raw_scores = self.narrator_llm.generate(
            instructions=(
                "You are the moderator scoring a philosophy debate. Return JSON only, with one entry for each debater."
            ),
            prompt=prompt,
            max_output_tokens=260,
        )
        return self._parse_scorecard(raw_scores, agent_names)

    def _agent_instructions(self, agent: DebateAgent) -> str:
        return (
            f"You are {agent.name}. {agent.guidance}\n"
            "Use the debate memory, recent transcript, and retrieved evidence faithfully.\n"
            "Do not invent citations, books, or quotations.\n"
            "Engage the other agents charitably but directly.\n"
            "Avoid repetition and keep your reasoning cumulative across turns."
        )

    def _build_retrieval_query(
        self,
        topic: str,
        transcript: list[DebateTurn],
        debate_memory: str,
        agent_name: str,
        phase: str,
    ) -> str:
        latest_turns = self._format_recent_turns(transcript, max_turns=len(self.agent_order) + 1)
        return (
            f"Topic: {topic}\n"
            f"Speaker preparing next turn: {agent_name}\n"
            f"Phase: {phase}\n"
            f"Debate memory: {debate_memory}\n"
            f"Recent turns: {latest_turns or 'No turns yet.'}"
        )

    def _build_agent_prompt(
        self,
        *,
        agent: DebateAgent,
        topic: str,
        transcript: list[DebateTurn],
        debate_memory: str,
        phase: str,
        round_number: int,
        evidence: list[SearchResult],
    ) -> str:
        phase_guidance = {
            "opening": (
                "Offer an opening statement that frames the topic from your tradition, advances a positive case, "
                "and anticipates one serious challenge from another participant."
            ),
            "rebuttal": (
                "Respond directly to the strongest point made by at least one other participant, repair any weakness "
                "in your own side, and push the debate forward."
            ),
            "closing": (
                "Offer a closing statement. If a shared conclusion is possible, say what it is. If not, summarize "
                "your final position and acknowledge the remaining disagreement."
            ),
        }[phase]

        transcript_text = self._format_recent_turns(transcript, max_turns=len(self.agent_order) + 1)
        evidence_text = self._format_evidence(evidence)
        return (
            f'Topic: "{topic}"\n'
            f"Round: {round_number}\n"
            f"Phase: {phase}\n\n"
            f"Tradition profile:\n{agent.worldview}\n\n"
            f"Debate memory:\n{debate_memory}\n\n"
            f"Recent transcript:\n{transcript_text or 'This is the start of the debate.'}\n\n"
            f"Retrieved source material:\n{evidence_text}\n\n"
            f"Task:\n{phase_guidance}\n\n"
            "Write 2 to 3 substantial paragraphs in clear English. Build on prior turns instead of resetting the debate. "
            "Address at least one other agent by name when relevant. End with a line that starts with 'Current thesis:'. "
            "If this is a closing statement, finish with another line that starts with 'Consensus stance:' using either "
            "'shared conclusion' or 'respectful difference'."
        )

    def _format_recent_turns(self, transcript: list[DebateTurn], max_turns: int) -> str:
        if not transcript:
            return ""
        selected_turns = transcript[-max_turns:]
        return self._format_turns(selected_turns)

    @staticmethod
    def _format_turns(turns: list[DebateTurn]) -> str:
        if not turns:
            return ""
        return "\n\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)

    @staticmethod
    def _format_evidence(results: list[SearchResult]) -> str:
        if not results:
            return "No evidence retrieved."
        formatted = []
        for index, result in enumerate(results, start=1):
            excerpt = result.chunk.text[:240].strip()
            formatted.append(
                f"[{index}] {result.chunk.title} (score {result.score:.3f})\n"
                f"{excerpt}"
            )
        return "\n\n".join(formatted)

    @staticmethod
    def _citation_labels(results: list[SearchResult]) -> list[str]:
        seen: set[str] = set()
        labels: list[str] = []
        for result in results:
            label = result.chunk.title
            if label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    @staticmethod
    def _parse_scorecard(raw_scores: str, agent_names: list[str]) -> list[AgentScore]:
        default_comment = "Solid contribution, but the moderator model returned an incomplete justification."
        payload: object = {}
        json_text = DebateOrchestrator._extract_json_payload(raw_scores)
        if json_text:
            try:
                payload = json.loads(json_text)
            except json.JSONDecodeError:
                payload = {}

        if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
            items: list[object] = payload["scores"]
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        parsed_by_agent: dict[str, AgentScore] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent", "")).strip()
            if agent not in agent_names or agent in parsed_by_agent:
                continue
            coherence = DebateOrchestrator._clamp_score(item.get("coherence"))
            responsiveness = DebateOrchestrator._clamp_score(item.get("responsiveness"))
            evidence = DebateOrchestrator._clamp_score(item.get("evidence"))
            persuasion = DebateOrchestrator._clamp_score(item.get("persuasion"))
            total = coherence + responsiveness + evidence + persuasion
            comment = str(item.get("comment", "")).strip() or default_comment
            parsed_by_agent[agent] = AgentScore(
                agent=agent,
                coherence=coherence,
                responsiveness=responsiveness,
                evidence=evidence,
                persuasion=persuasion,
                total=total,
                comment=comment,
            )

        return [
            parsed_by_agent.get(
                agent_name,
                AgentScore(
                    agent=agent_name,
                    coherence=7,
                    responsiveness=7,
                    evidence=7,
                    persuasion=7,
                    total=28,
                    comment=default_comment,
                ),
            )
            for agent_name in agent_names
        ]

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            return object_match.group(0)
        list_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if list_match:
            return list_match.group(0)
        return ""

    @staticmethod
    def _clamp_score(value: object) -> int:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return 7
        return max(1, min(10, score))

    @staticmethod
    def _build_winner_text(scores: list[AgentScore]) -> str:
        if not scores:
            return "No winner could be determined."
        highest_total = max(score.total for score in scores)
        winners = [score.agent for score in scores if score.total == highest_total]
        if len(winners) == 1:
            return f"{winners[0]} delivered the strongest overall debate performance."
        return f"Tie between {', '.join(winners)} for the strongest overall debate performance."

    @staticmethod
    def _parse_labeled_sections(text: str, labels: list[str]) -> dict[str, str]:
        sections: dict[str, str] = {}
        for index, label in enumerate(labels):
            if index + 1 < len(labels):
                next_label = labels[index + 1]
                pattern = rf"{re.escape(label)}:\s*(.*?)(?=\n{re.escape(next_label)}:|\Z)"
            else:
                pattern = rf"{re.escape(label)}:\s*(.*)$"
            match = re.search(pattern, text, flags=re.DOTALL)
            sections[label] = match.group(1).strip() if match else ""

        if not sections["Outcome"]:
            sections["Outcome"] = "Respectful difference."
        if not sections["Moderator Summary"]:
            sections["Moderator Summary"] = "The agents developed distinct but serious answers to the topic."
        if not sections["Final Reflection"]:
            sections["Final Reflection"] = "The debate ends with a clearer map of agreement and disagreement."
        return sections
