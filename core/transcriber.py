import os
from pathlib import Path

import requests
import whisper
from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_SST_MODEL", "saaras:v4")

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")

    return _model


def transcriber_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def transcriber_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing from .env")

    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(chunk_path, "rb") as audio_file:
        files = {
            "file": (
                os.path.basename(chunk_path),
                audio_file,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "mode": "translate"
        }

        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=300
        )

    response.raise_for_status()
    return response.json().get("transcript", "")








def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route English audio to Whisper and Hinglish audio to Sarvam.
    """

    if language.lower() == "hinglish":
        return transcriber_chunk_sarvam(chunk_path)

    return transcriber_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription completed")
    return full_transcript.strip()
