# Windows Setup Guide

This guide walks through running the project on Windows from a clean machine. The app now includes Stoic, Vedantam, and Machiavellianism agents, with a moderator scorecard and per-turn voice playback at the end of each debate.

## 1. Install Python

1. Download Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/).
2. Run the installer.
3. Enable `Add python.exe to PATH` during installation.
4. Finish the install.

Verify in PowerShell:

```powershell
py --version
```

## 2. Open the Project Folder

Open PowerShell and move into the repo:

```powershell
Set-Location "D:\Philosophy Debate AI"
```

## 3. Create a Virtual Environment

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Install Project Dependencies

Upgrade `pip` first:

```powershell
py -m pip install --upgrade pip
```

Install the packages from `requirements.txt`:

```powershell
py -m pip install -r requirements.txt
```

Note:
The first actual project run may also download the local sentence-transformer embedding model.

## 5. Install Tesseract OCR on Windows

This project uses Tesseract only when direct PDF text extraction is too weak.

If you want to get the app running quickly, you can skip Tesseract at first and come back to it later. The app may still work fine for PDFs that already contain extractable text.

If the UB Mannheim download site shows `Forbidden`, use one of these workarounds.

### Recommended: install with winget

The easiest Windows path is:

```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```

Then verify:

```powershell
tesseract --version
```

If that works, Tesseract is on `PATH` and you can continue.

### Fallback: use the GitHub release installer

If `winget` is unavailable, try this installer URL in your browser:

[https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)

Install it, then verify:

```powershell
tesseract --version
```

Typical install directory:

```text
C:\Program Files\Tesseract-OCR
```

### If English language data is missing

If you see an error mentioning `eng.traineddata` or `Tesseract couldn't load any languages`, do this:

1. Open the official `tessdata` repository:
   [https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)
2. Download `eng.traineddata`.
3. Place it here:

```text
C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata
```

4. Test again:

```powershell
tesseract --list-langs
```

You should see `eng` in the output.

### If `tesseract` is still not recognized

You have two options.

Option A: add Tesseract to your Windows `PATH`

Typical folder:

```text
C:\Program Files\Tesseract-OCR
```

Option B: set `TESSERACT_CMD` in the project `.env` file

Typical value:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 6. Create the Environment File

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```env
GROQ_API_KEY=your_groq_api_key_here
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

If `tesseract --version` did not work, set `TESSERACT_CMD` explicitly:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 7. Build the Local Vector Indexes

Run:

```powershell
py scripts\build_indexes.py --rebuild
```

What this does:

- reads all configured PDF corpuses
- extracts text directly where possible
- falls back to OCR when needed
- creates local sentence-transformer embeddings
- stores persistent Chroma indexes under `storage/`

The first run can take a while.

## 8. Start the Streamlit App

Run:

```powershell
py -m streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that in your browser.

## 9. Use the App

1. Wait for the corpuses to load.
2. Enter a debate topic.
3. Click `Start debate`.
4. Watch the Stoic Agent, Vedantam Agent, Machiavellianism Agent, narrator, and final moderator verdict update turn by turn.
5. Use the `Speak` button under any turn if you want voice playback.

## 10. Future Runs

After the first setup, the normal flow is:

```powershell
Set-Location "D:\Philosophy Debate AI"
.\.venv\Scripts\Activate.ps1
py -m streamlit run app.py
```

If you change the PDFs, rebuild the indexes:

```powershell
py scripts\build_indexes.py --rebuild
```

## Troubleshooting

### `py` is not recognized

Reinstall Python and make sure the Python launcher is included.

### Activation is blocked

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### `ModuleNotFoundError`

Reinstall dependencies:

```powershell
py -m pip install -r requirements.txt
```

### `GROQ_API_KEY is not set`

Check `.env` and make sure it contains a valid Groq API key.

### `tesseract` is not recognized

Either:

- add `C:\Program Files\Tesseract-OCR` to your Windows `PATH`
- or set `TESSERACT_CMD` in `.env`

### OCR still fails

Confirm this file exists:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Then make sure `.env` contains the matching `TESSERACT_CMD`.

### Voice playback fails

The debate can still run normally. Voice playback depends on outbound network access for the free text-to-speech service.
