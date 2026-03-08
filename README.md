# Philosophy Debate AI

Author: Priyanshu Garg

This is a Streamlit project I built to stage long-form philosophy debates between multiple traditions using separate RAG pipelines.

Right now the app includes:

- Stoic Agent
- Vedantam Agent
- Machiavellianism Agent
- Narrator round summaries
- Moderator final verdict with a compact scorecard
- Per-turn voice playback with distinct male voices
- Prebuilt local indexes for faster startup

The main Streamlit entry file is `app.py`.

## What this project does

The app reads three philosophy corpuses, uses separate Chroma indexes for each one, and lets each agent debate from its own source material.

The flow is simple:

1. The user enters a debate topic.
2. Each agent retrieves from its own corpus.
3. The debate runs round by round.
4. A compressed debate memory is maintained so the discussion keeps context without blowing up prompt size.
5. A narrator summarizes each round.
6. A moderator closes the debate with an outcome, scorecard, winner, and final reflection.

## Stack

- UI: Streamlit
- Framework: LangChain
- LLM: Groq
- Embeddings: local sentence-transformers
- Vector database: Chroma
- OCR fallback: Tesseract + PyMuPDF + pytesseract
- Voice playback: edge-tts

## Project structure

```text
.
|-- app.py
|-- philosophy_debate/
|   |-- config.py
|   |-- debate.py
|   |-- document_processing.py
|   |-- llm.py
|   |-- models.py
|   |-- retrieval.py
|   |-- runtime.py
|   `-- tts.py
|-- scripts/
|   |-- build_indexes.py
|   `-- publish_to_github.ps1
|-- Stoicism Corpus/
|-- Vedanta corpus/
|-- Machiavellianism Corpus/
|-- storage/
|-- .streamlit/
|-- packages.txt
|-- requirements.txt
|-- WINDOWS_SETUP.md
`-- README.md
```

## Important deployment choice

This repository is set up to use prebuilt indexes stored in `storage/`.

That means:

- indexes are built locally once
- the generated `storage/` folder is committed to GitHub
- Streamlit Community Cloud loads those indexes instead of rebuilding them on every cold start

This makes deployment much faster and avoids repeated embedding work in the cloud.

## Local setup

If you want the full Windows walkthrough, use [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

Quick version:

```powershell
Set-Location "D:\Philosophy Debate AI"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Then put your Groq key in `.env`:

```env
GROQ_API_KEY=your_key_here
```

If OCR is needed and Tesseract is not on `PATH`, also set:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Build indexes locally

If the repo already contains a populated `storage/` folder, you usually do not need to rebuild indexes.

Rebuild only if you changed corpus files or changed the embedding model:

```powershell
py scripts\build_indexes.py --rebuild
```

## Run locally

```powershell
py -m streamlit run app.py
```

## Run again later

Once the environment and indexes already exist, the normal daily run is:

```powershell
Set-Location "D:\Philosophy Debate AI"
.\.venv\Scripts\Activate.ps1
py -m streamlit run app.py
```

## If you change the corpuses

If you add, remove, or edit files in any corpus folder, do this:

```powershell
Set-Location "D:\Philosophy Debate AI"
.\.venv\Scripts\Activate.ps1
py scripts\build_indexes.py --rebuild
```

Then commit the updated `storage/` folder to GitHub so Streamlit Cloud uses the new indexes.

## Streamlit Community Cloud deployment

This repo is already structured so it can be pushed to GitHub and connected to Streamlit Community Cloud.

Use this setup:

1. Push the full repository to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Select this repository.
4. Set the main file path to `app.py`.
5. In Advanced settings, choose Python `3.12`.
6. Add `GROQ_API_KEY` in the app secrets or environment settings.
7. Deploy.

If the app was previously created with Python `3.14`, delete it and redeploy it with Python `3.12`.

Notes:

- `packages.txt` is included for Tesseract on Streamlit Community Cloud.
- `storage/` is intentionally committed so the cloud app can load prebuilt indexes.
- The first cloud run is much faster when `storage/` is already present.
- Voice playback depends on outbound network access for the free TTS service.

## Debate design notes

- Each agent only sees its own corpus during retrieval.
- The app uses a compressed rolling debate memory instead of dragging the full transcript into every prompt.
- Recent turns are still kept so the conversation feels alive and responsive.
- The moderator scorecard is intentionally compact so all agents are always scored.

## Voice design notes

Each speaker has a distinct male voice profile with a wise or reflective tone.

The intent is:

- Stoic Agent: calm, weighty, classical gravitas
- Vedantam Agent: serene teacher-like warmth
- Machiavellianism Agent: strategic, firm, politically sharp
- Narrator: reflective documentary tone
- Moderator: composed, judicial tone

These are style choices, not direct impersonations of historical or sacred figures.

## Environment variables

```env
GROQ_API_KEY=
DEBATE_MODEL=llama-3.1-8b-instant
NARRATOR_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
DEFAULT_DEBATE_ROUNDS=4
TOP_K_RESULTS=4
MIN_DIRECT_TEXT_CHARS=1200
OCR_LANGUAGE=eng
TESSERACT_CMD=
```

## Practical tips

- For Groq free tier, 3 rounds and 2 to 3 evidence chunks per turn are the safest settings.
- If you want longer debates, increase rounds slowly.
- If a PDF already contains text, the app does not need OCR for it.
- If voice playback fails, the debate still works normally.
- If you rebuild indexes locally, commit the updated `storage/` folder before redeploying to Streamlit Cloud.

## Final note

This repository is intended to be simple to read, simple to run, and simple to publish.

If I push this to GitHub, the only things I still need to set outside the repo are:

- my Groq API key
- Streamlit Community Cloud app settings
- optional local Tesseract install path on Windows