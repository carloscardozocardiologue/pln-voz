"""Transcripción de audio mediante Whisper large-v3 vía HuggingFace Inference API."""

import io
import os
from pydub import AudioSegment
from huggingface_hub import InferenceClient


def _get_hf_token() -> str:
    try:
        import streamlit as st
        val = st.secrets.get("HF_TOKEN")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("HF_TOKEN", "")


def transcribir(audio_bytes: bytes) -> str:
    """
    Recibe los bytes de audio grabados en Streamlit y devuelve
    el texto transcrito en español usando Whisper large-v3.
    """
    # El navegador graba en WebM; pydub lo convierte a WAV
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    client = InferenceClient(token=_get_hf_token())
    resultado = client.automatic_speech_recognition(
        wav_buffer.read(),
        model="openai/whisper-large-v3",
    )
    return resultado.text
