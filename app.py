from __future__ import annotations

import hashlib
import html
import time
from queue import Empty, Queue
from threading import Thread
from typing import Any

import streamlit as st

from philosophy_debate.models import DebateTurn, IndexBuildReport
from philosophy_debate.runtime import RuntimeBundle, build_runtime
from philosophy_debate.tts import get_voice_profile, synthesize_speech, tts_is_available

st.set_page_config(
    page_title="Philosophy Debate AI",
    page_icon=":books:",
    layout="wide",
)


VOICE_ENABLED = tts_is_available()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(161, 74, 26, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(31, 91, 110, 0.16), transparent 30%),
                    linear-gradient(180deg, #fbf6ee 0%, #f4ede2 100%);
                color: #1f1a16;
            }
            h1, h2, h3 {
                font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
                letter-spacing: 0.02em;
            }
            .hero {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(85, 64, 44, 0.16);
                border-radius: 20px;
                padding: 1.5rem 1.75rem;
                box-shadow: 0 18px 40px rgba(38, 24, 13, 0.08);
                margin-bottom: 1rem;
            }
            .corpus-card {
                background: rgba(255, 255, 255, 0.72);
                border-radius: 18px;
                border: 1px solid rgba(85, 64, 44, 0.14);
                padding: 1rem 1.1rem;
                min-height: 145px;
                box-shadow: 0 10px 30px rgba(38, 24, 13, 0.06);
            }
            .corpus-kicker {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #7e5f46;
                margin-bottom: 0.35rem;
            }
            .corpus-count {
                font-size: 2rem;
                font-weight: 700;
                margin: 0.15rem 0 0.4rem;
            }
            .debate-card {
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.45rem;
                border: 1px solid rgba(69, 51, 34, 0.14);
                box-shadow: 0 10px 28px rgba(38, 24, 13, 0.05);
                background: rgba(255, 255, 255, 0.8);
            }
            .debate-card.stoic {
                border-left: 6px solid #a14a1a;
            }
            .debate-card.vedanta {
                border-left: 6px solid #1f5b6e;
            }
            .debate-card.machiavellian {
                border-left: 6px solid #3d4f2a;
            }
            .debate-card.narrator {
                border-left: 6px solid #6f665d;
                background: rgba(247, 244, 239, 0.88);
            }
            .debate-card.moderator {
                border-left: 6px solid #53443a;
                background: rgba(250, 247, 241, 0.9);
            }
            .card-topline {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                color: #6d5e51;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.5rem;
            }
            .card-title {
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }
            .card-body {
                line-height: 1.7;
                font-size: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_cached_runtime() -> RuntimeBundle:
    return build_runtime(force_rebuild=False)


@st.cache_data(show_spinner=False)
def get_audio_bytes(speaker: str, text: str) -> bytes:
    return synthesize_speech(text=text, speaker=speaker)


def load_runtime() -> RuntimeBundle:
    return get_cached_runtime()


def ensure_session_state() -> None:
    st.session_state.setdefault("debate_turns", [])
    st.session_state.setdefault("active_audio_turn", None)
    st.session_state.setdefault("debate_running", False)
    st.session_state.setdefault("debate_error", None)
    st.session_state.setdefault("debate_events", None)
    st.session_state.setdefault("debate_thread", None)
    st.session_state.setdefault("debate_topic", "")


def build_turn_key(turn: DebateTurn) -> str:
    digest = hashlib.sha1(f"{turn.speaker}|{turn.phase}|{turn.round_number}|{turn.text}".encode("utf-8")).hexdigest()
    return digest[:16]


def start_debate_worker(runtime: RuntimeBundle, *, topic: str, rounds: int, top_k: int) -> None:
    event_queue: Queue[dict[str, Any]] = Queue()

    def worker() -> None:
        try:
            for turn in runtime.debate_service.debate(topic=topic, rounds=rounds, top_k=top_k):
                event_queue.put({"type": "turn", "turn": turn})
        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put({"type": "done"})

    st.session_state["debate_turns"] = []
    st.session_state["active_audio_turn"] = None
    st.session_state["debate_error"] = None
    st.session_state["debate_events"] = event_queue
    st.session_state["debate_topic"] = topic
    st.session_state["debate_running"] = True

    thread = Thread(target=worker, daemon=True)
    st.session_state["debate_thread"] = thread
    thread.start()


def drain_debate_events() -> None:
    event_queue = st.session_state.get("debate_events")
    if event_queue is None:
        return

    while True:
        try:
            event = event_queue.get_nowait()
        except Empty:
            break

        event_type = event.get("type")
        if event_type == "turn":
            turn = event.get("turn")
            if isinstance(turn, DebateTurn):
                st.session_state["debate_turns"].append(turn)
        elif event_type == "error":
            st.session_state["debate_error"] = str(event.get("message", "Unknown debate error."))
        elif event_type == "done":
            st.session_state["debate_running"] = False

    thread = st.session_state.get("debate_thread")
    if thread is not None and not thread.is_alive() and event_queue.empty():
        st.session_state["debate_running"] = False


def render_report(report: IndexBuildReport, accent_color: str) -> None:
    cache_label = "Loaded from cache" if report.loaded_from_cache else "Fresh build"
    ocr_label = "OCR used on at least one PDF" if report.used_ocr else "Direct text extraction only"
    st.markdown(
        f"""
        <div class="corpus-card">
            <div class="corpus-kicker">{html.escape(report.display_name)}</div>
            <div class="corpus-count" style="color: {accent_color};">{report.source_count}</div>
            <div><strong>{report.chunk_count}</strong> searchable chunks</div>
            <div style="margin-top: 0.5rem; color: #66584b;">{html.escape(cache_label)}</div>
            <div style="margin-top: 0.25rem; color: #66584b;">{html.escape(ocr_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report.warnings:
        with st.expander(f"{report.display_name} warnings"):
            for warning in report.warnings:
                st.write(f"- {warning}")


def render_turn(turn: DebateTurn) -> None:
    card_class = {
        "Stoic Agent": "stoic",
        "Vedantam Agent": "vedanta",
        "Machiavellianism Agent": "machiavellian",
        "Narrator": "narrator",
        "Moderator": "moderator",
    }.get(turn.speaker, "moderator")
    round_label = "Opening" if turn.round_number == 0 else f"Round {turn.round_number}"
    body = html.escape(turn.text).replace("\n", "<br>")
    turn_key = build_turn_key(turn)
    st.markdown(
        f"""
        <div class="debate-card {card_class}">
            <div class="card-topline">
                <span>{html.escape(turn.phase.replace("_", " ").title())}</span>
                <span>{html.escape(round_label)}</span>
            </div>
            <div class="card-title">{html.escape(turn.speaker)}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if turn.citations:
        st.caption("Sources consulted: " + " | ".join(turn.citations))

    profile = get_voice_profile(turn.speaker)
    if not VOICE_ENABLED:
        st.caption(f"Voice: {profile.label} Audio is currently unavailable because 'edge-tts' is not installed.")
        return

    controls = st.columns([1.2, 5.8])
    with controls[0]:
        if st.button("Speak", key=f"speak-{turn_key}", use_container_width=True):
            st.session_state["active_audio_turn"] = turn_key
    with controls[1]:
        st.caption(f"Voice: {profile.label}")

    if st.session_state.get("active_audio_turn") == turn_key:
        try:
            with st.spinner(f"Generating {turn.speaker} voice..."):
                audio_bytes = get_audio_bytes(turn.speaker, turn.text)
            st.audio(audio_bytes, format="audio/mpeg")
        except Exception as exc:
            st.info(
                "Voice playback is unavailable right now. "
                f"Reason: {exc}"
            )


def render_transcript(turns: list[DebateTurn]) -> None:
    for turn in turns:
        render_turn(turn)


def render_debate_region() -> None:
    drain_debate_events()

    if st.session_state["debate_error"]:
        st.error(f"Debate failed: {st.session_state['debate_error']}")
    elif st.session_state["debate_running"]:
        st.info("Debate in progress. You can use Speak while new turns continue to arrive.")
    elif st.session_state["debate_turns"]:
        st.success("Debate complete.")

    if st.session_state["debate_topic"]:
        st.caption(f"Live topic: {st.session_state['debate_topic']}")

    render_transcript(st.session_state["debate_turns"])


if hasattr(st, "fragment"):
    @st.fragment(run_every=1)
    def render_live_debate() -> None:
        render_debate_region()
else:
    def render_live_debate() -> None:
        render_debate_region()
        if st.session_state.get("debate_running"):
            time.sleep(1)
            st.rerun()


def main() -> None:
    ensure_session_state()
    inject_styles()
    st.markdown(
        """
        <div class="hero">
            <h1>Philosophy Debate AI</h1>
            <p>
                A three-agent RAG debate between Stoicism, Vedanta, and Machiavellianism. Enter a topic,
                let each tradition reason from its own corpus, and follow the narrated exchange as it moves
                toward either shared ground or a careful agreement to disagree. The moderator closes with a
                compact scorecard on who debated best.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    debate_running = st.session_state["debate_running"]
    with st.sidebar:
        st.header("Debate Controls")
        rounds = st.slider("Debate rounds", min_value=2, max_value=6, value=3, disabled=debate_running)
        top_k = st.slider("Evidence chunks per turn", min_value=2, max_value=5, value=3, disabled=debate_running)
        st.caption("For Groq free tier, 3 rounds and 2 to 3 evidence chunks are the safest defaults.")
        st.caption("Knowledge bases are loaded from the prebuilt storage cache and repaired automatically only if a cached index is unusable.")
        if not VOICE_ENABLED:
            st.info("Voice playback is disabled in this environment because 'edge-tts' is not installed yet.")

    try:
        with st.spinner("Loading corpuses and indexes..."):
            runtime = load_runtime()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    report_columns = st.columns(len(runtime.config.corpora))
    for column, (corpus_key, corpus) in zip(report_columns, runtime.config.corpora.items(), strict=False):
        with column:
            render_report(runtime.reports[corpus_key], corpus.accent_color)

    default_topic = "Is a good life best achieved through virtue, realization of the deeper self, or strategic mastery of power?"
    topic = st.text_area(
        "Debate topic",
        value=default_topic,
        height=100,
        help="Write a question, thesis, or tension that all traditions should engage deeply.",
        disabled=debate_running,
    )
    start_debate = st.button("Start debate", type="primary", use_container_width=True, disabled=debate_running)

    if start_debate:
        if not topic.strip():
            st.warning("Enter a debate topic before starting the discussion.")
        else:
            start_debate_worker(runtime, topic=topic.strip(), rounds=rounds, top_k=top_k)

    render_live_debate()


if __name__ == "__main__":
    main()
