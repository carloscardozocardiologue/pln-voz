"""Transcripción de audio mediante Google Speech Recognition + corrección de términos médicos."""

import io
import speech_recognition as sr
from pydub import AudioSegment

# Correcciones fonéticas: lo que Google SR transcribe → término médico correcto
_CORRECCIONES = {
    # Anticoagulantes
    "david galan":       "dabigatrán",
    "david galán":       "dabigatrán",
    "david gatran":      "dabigatrán",
    "davigatran":        "dabigatrán",
    "rival oxaban":      "rivaroxabán",
    "rivaroxaban":       "rivaroxabán",
    "apixa van":         "apixabán",
    "apixa ban":         "apixabán",
    "apixaban":          "apixabán",
    "edoxaban":          "edoxabán",
    # Antiarrítmicos
    "amio darona":       "amiodarona",
    "amio de rona":      "amiodarona",
    "flecainida":        "flecainida",
    "propafenona":       "propafenona",
    # Insuficiencia cardíaca
    "sacubitrilo":       "sacubitrilo",
    "sacubitril":        "sacubitrilo",
    "ivabradina":        "ivabradina",
    "espironolactona":   "espironolactona",
    "eplerenona":        "eplerenona",
    "empagliflocina":    "empagliflozina",
    "empagliflozina":    "empagliflozina",
    "dapagliflozina":    "dapagliflozina",
    # IECA / ARA-II
    "ramipril":          "ramipril",
    "enalapril":         "enalapril",
    "losartan":          "losartán",
    "valsartan":         "valsartán",
    # Betabloqueantes
    "bisoprolol":        "bisoprolol",
    "carvedilol":        "carvedilol",
    "metoprolol":        "metoprolol",
    # Estatinas
    "atorvastatina":     "atorvastatina",
    "rosuvastatina":     "rosuvastatina",
    # Términos clínicos frecuentes
    "fibrilacion":       "fibrilación",
    "taquicardia supraventricular": "taquicardia supraventricular",
    "sindrome coronario": "síndrome coronario agudo",
    "infarto agudo":     "infarto agudo de miocardio",
    "insuficiencia cardiaca": "insuficiencia cardíaca",
}


def _corregir(texto: str) -> str:
    t = texto.lower()
    for erroneo, correcto in _CORRECCIONES.items():
        t = t.replace(erroneo, correcto)
    # Restaurar mayúscula inicial
    return t[0].upper() + t[1:] if t else t


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

    texto = recognizer.recognize_google(audio_data, language="es-ES")
    return _corregir(texto)
