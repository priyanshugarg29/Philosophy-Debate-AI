from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


class GroqTextGenerator:
    def __init__(self, model: str, temperature: float = 0.2) -> None:
        self.model = model
        self.client = ChatGroq(
            model=model,
            temperature=temperature,
        )

    def generate(
        self,
        instructions: str,
        prompt: str,
        *,
        max_output_tokens: int = 900,
    ) -> str:
        response = self.client.bind(max_tokens=max_output_tokens).invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=prompt),
            ]
        )
        text = self._extract_output_text(response.content)
        if not text:
            raise RuntimeError("The model returned an empty response.")
        return text.strip()

    @staticmethod
    def _extract_output_text(content: object) -> str:
        if isinstance(content, str):
            return content

        pieces: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
        return "\n".join(piece for piece in pieces if piece).strip()


class LocalEmbeddingService:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )