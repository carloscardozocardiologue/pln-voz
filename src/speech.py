"""Transcripción de audio mediante Google Speech Recognition + corrección de términos médicos."""

import io
import speech_recognition as sr
from pydub import AudioSegment

# Correcciones fonéticas: lo que Google SR transcribe → término médico correcto
_CORRECCIONES = {
    # ── Anticoagulantes ───────────────────────────────────────────────
    "david galan":          "dabigatrán",
    "david galán":          "dabigatrán",
    "david gatran":         "dabigatrán",
    "davigatran":           "dabigatrán",
    "dabiga tran":          "dabigatrán",
    "rival oxaban":         "rivaroxabán",
    "ribaroxaban":          "rivaroxabán",
    "rivaroxaban":          "rivaroxabán",
    "apixa van":            "apixabán",
    "apixa ban":            "apixabán",
    "apixaban":             "apixabán",
    "edoxaban":             "edoxabán",
    # ── Antiagregantes ───────────────────────────────────────────────
    "tica grelor":          "ticagrelor",
    "ticker grelor":        "ticagrelor",
    "tika grelor":          "ticagrelor",
    "prazo grel":           "prasugrel",
    "clopi dogrel":         "clopidogrel",
    # ── Antiarrítmicos ───────────────────────────────────────────────
    "amio darona":          "amiodarona",
    "amio de rona":         "amiodarona",
    "drone darona":         "dronedarona",
    "dron darona":          "dronedarona",
    # ── Insuficiencia cardíaca ────────────────────────────────────────
    "sacubitril":           "sacubitrilo",
    "saku bitrilo":         "sacubitrilo",
    "epler enona":          "eplerenona",
    "espiro nolactona":     "espironolactona",
    "empagliflocina":       "empagliflozina",
    "empagliflozina":       "empagliflozina",
    "dapagliflocina":       "dapagliflozina",
    "canagliflocina":       "canagliflozina",
    # ── IECA / ARA-II ────────────────────────────────────────────────
    "perindo pril":         "perindopril",
    "cande sartan":         "candesartán",
    "candesartan":          "candesartán",
    "irbe sartan":          "irbesartán",
    "irbesartan":           "irbesartán",
    "olme sartan":          "olmesartán",
    "olmesartan":           "olmesartán",
    "telmi sartan":         "telmisartán",
    "telmisartan":          "telmisartán",
    "losartan":             "losartán",
    "valsartan":            "valsartán",
    # ── Betabloqueantes ──────────────────────────────────────────────
    "nebi bolol":           "nebivolol",
    "nebi volol":           "nebivolol",
    # ── Calcioantagonistas ───────────────────────────────────────────
    "amlo dipino":          "amlodipino",
    "vera pamilo":          "verapamilo",
    "dil tia sem":          "diltiazem",
    "diltia sem":           "diltiazem",
    # ── Estatinas ────────────────────────────────────────────────────
    "rosu vastatina":       "rosuvastatina",
    "pita vastatina":       "pitavastatina",
    # ── Términos clínicos con tildes ─────────────────────────────────
    "fibrilacion":          "fibrilación",
    "fibrilacion auricular":"fibrilación auricular",
    "ablacion":             "ablación",
    "cardioversion":        "cardioversión",
    "desfibrilacion":       "desfibrilación",
    "resincronizacion":     "resincronización",
    "resin cronizacion":    "resincronización",
    "trombo embolia":       "tromboembolia",
    "trombo embolismo":     "tromboembolismo",
    "sindrome coronario":   "síndrome coronario",
    "insuficiencia cardiaca": "insuficiencia cardíaca",
    "arritmia":             "arritmia",
    "hipertension":         "hipertensión",
    "hipertension arterial":"hipertensión arterial",
    "disfuncion":           "disfunción",
    "intervencion":         "intervención",
    "oclusion":             "oclusión",
    "estenosis":            "estenosis",
}


def _corregir(texto: str) -> str:
    """Aplica el diccionario de correcciones fonéticas al texto transcrito."""
    t = texto.lower()
    # Ordenamos por longitud descendente para que las frases largas tengan prioridad
    for erroneo, correcto in sorted(_CORRECCIONES.items(), key=lambda x: -len(x[0])):
        t = t.replace(erroneo, correcto)
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
