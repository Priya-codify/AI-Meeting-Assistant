import os
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and convert it to WAV."""
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    yt_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "retries": 3,
        "fragment_retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        wav_path = os.path.join(
            DOWNLOAD_DIR, f"{info['id']}.wav"
        )

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            f"Downloaded WAV file was not found: {wav_path}"
        )

    return wav_path


def convert_to_wav(input_path: str) -> str:
    """Convert an audio or video file to mono, 16 kHz WAV."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_seconds: int = 25) -> list[str]:
    """Split a WAV file into chunks for transcription."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")

    audio = AudioSegment.from_wav(wav_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    chunk_ms = chunk_seconds * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list[str]:
    """Download or convert an input file, then split it into chunks."""
    if source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Input file not found: {source}")

        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)

    if not chunks:
        raise ValueError("No audio chunks were created.")

    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
