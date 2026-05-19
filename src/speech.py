"""Transcripción de audio mediante SpeechRecognition + Google Speech API."""

import io
import speech_recognition as sr
from pydub import AudioSegment


def transcribir(audio_bytes: bytes) -> str:
    """
    Recibe los bytes de audio grabados en Streamlit y devuelve
    el texto transcrito en español usando Google Speech Recognition.
    """
    # El navegador graba en WebM; pydub lo convierte a WAV
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio_data = recognizer.record(source)

    return recognizer.recognize_google(audio_data, language="es-ES")
