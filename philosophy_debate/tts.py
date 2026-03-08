from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

try:
    import edge_tts
except ImportError:  # pragma: no cover
    edge_tts = None

FALLBACK_VOICE = "en-US-GuyNeural"


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    voice: str
    rate: str = "-10%"
    pitch: str = "-10Hz"
    volume: str = "+0%"
    label: str = "Clear male voice."


VOICE_PROFILES = {
    "Stoic Agent": VoiceProfile(
        voice="en-GB-RyanNeural",
        rate="-12%",
        pitch="-12Hz",
        label="Calm, weighty male voice with classical gravitas.",
    ),
    "Vedantam Agent": VoiceProfile(
        voice="en-IN-PrabhatNeural",
        rate="-14%",
        pitch="-8Hz",
        label="Serene Indian English male voice with teacher-like warmth.",
    ),
    "Machiavellianism Agent": VoiceProfile(
        voice="en-US-ChristopherNeural",
        rate="-8%",
        pitch="-10Hz",
        label="Measured, strategic male voice with a firm edge.",
    ),
    "Narrator": VoiceProfile(
        voice="en-AU-WilliamNeural",
        rate="-8%",
        pitch="-10Hz",
        label="Reflective male narrator voice.",
    ),
    "Moderator": VoiceProfile(
        voice="en-US-GuyNeural",
        rate="-6%",
        pitch="-10Hz",
        label="Clear, composed male moderator voice.",
    ),
}


def get_voice_profile(speaker: str) -> VoiceProfile:
    return VOICE_PROFILES.get(speaker, VoiceProfile(voice=FALLBACK_VOICE))


def tts_is_available() -> bool:
    return edge_tts is not None


def clean_text_for_speech(text: str) -> str:
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"#+\s*", "", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+\|\s+", ", ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


async def _synthesize_async(text: str, profile: VoiceProfile) -> bytes:
    if edge_tts is None:
        raise RuntimeError(
            "Voice playback requires the optional 'edge-tts' package. Install it with: py -m pip install edge-tts"
        )
    communicate = edge_tts.Communicate(
        clean_text_for_speech(text),
        voice=profile.voice,
        rate=profile.rate,
        pitch=profile.pitch,
        volume=profile.volume,
    )
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
    if not audio_bytes:
        raise RuntimeError("No audio data was returned by the speech service.")
    return bytes(audio_bytes)


def _run_async(factory: Callable[[], Awaitable[bytes]]) -> bytes:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    result: dict[str, bytes] = {}
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result["audio"] = asyncio.run(factory())
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if errors:
        raise RuntimeError(str(errors[0])) from errors[0]
    if "audio" not in result:
        raise RuntimeError("Speech generation finished without audio output.")
    return result["audio"]


def synthesize_speech(text: str, speaker: str) -> bytes:
    profile = get_voice_profile(speaker)
    try:
        return _run_async(lambda: _synthesize_async(text, profile))
    except Exception:
        if profile.voice == FALLBACK_VOICE:
            raise
        fallback_profile = VoiceProfile(
            voice=FALLBACK_VOICE,
            rate=profile.rate,
            pitch=profile.pitch,
            volume=profile.volume,
            label=profile.label,
        )
        return _run_async(lambda: _synthesize_async(text, fallback_profile))
