"""Transcripción de audio mediante Whisper vía HuggingFace Inference API."""

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

MODELO_WHISPER = "openai/whisper-large-v3"


def transcribir(audio_bytes: bytes) -> str:
    """
    Recibe los bytes de audio grabados en Streamlit y devuelve
    el texto transcrito en español.
    """
    cliente = InferenceClient(token=os.getenv("HF_TOKEN"))
    resultado = cliente.automatic_speech_recognition(
        audio=audio_bytes,
        model=MODELO_WHISPER,
    )
    return resultado.text
