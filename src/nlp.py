"""
Pipeline NLP: TF-IDF retrieval + OpenAI RAG en español.

Etapa 1 — TF-IDF: encuentra los contextos más similares a la pregunta del médico.
Etapa 2 — OpenAI: genera una respuesta basándose ESTRICTAMENTE en esos contextos.
           Si el contexto no cubre la pregunta, devuelve el mensaje de fallback.
Fallback absoluto: si TF-IDF < umbral mínimo, se devuelve el mensaje de fallback
           sin llamar a OpenAI.
"""

import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

_DATA_PATH           = Path(__file__).parent.parent / "data" / "preguntas_cardiologia_esc_500.csv"
_TFIDF_UMBRAL        = 0.15   # mínimo para intentar responder (umbral absoluto)
_TFIDF_PREGUNTA_MIN  = 0.35   # si pregunta-sola supera esto, no hace falta buscar en contexto
_TOP_K               = 2


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y prefijos médicos normalizados al español."""
    texto = texto.lower()
    texto = re.sub(r'\bhypo', 'hipo', texto)
    texto = re.sub(r'\bhyper', 'hiper', texto)
    texto = re.sub(r'\bpneu', 'neu', texto)
    texto = re.sub(r'\bthrombo', 'trombo', texto)
    texto = re.sub(r'\barrhyth', 'arrit', texto)
    nfkd  = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _cargar():
    df = pd.read_csv(_DATA_PATH)

    # Matriz 1: solo pregunta — alta similitud para preguntas exactas o casi exactas
    vec_p = TfidfVectorizer(ngram_range=(1, 2))
    mat_p = vec_p.fit_transform(df["pregunta"].apply(_normalizar))

    # Matriz 2: pregunta + contexto — fallback para keywords que solo aparecen en el contexto
    vec_pc = TfidfVectorizer(ngram_range=(1, 2))
    mat_pc = vec_pc.fit_transform(
        (df["pregunta"] + " " + df["contexto"].fillna("")).apply(_normalizar)
    )

    return df, vec_p, mat_p, vec_pc, mat_pc


def _get_openai_key() -> str:
    try:
        import streamlit as st
        val = st.secrets.get("OPENAI_API_KEY")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


_df, _vec_p, _mat_p, _vec_pc, _mat_pc = _cargar()

FALLBACK = (
    "No tengo información suficiente en mi base de conocimiento para responder "
    "esa pregunta. Por favor, reformula la consulta o selecciona otra categoría."
)

_SYSTEM_PROMPT = (
    "Eres un asistente médico especializado en cardiología para uso hospitalario. "
    "REGLAS ESTRICTAS:\n"
    "1. Responde ÚNICAMENTE con información que aparezca explícitamente en los "
    "contextos clínicos proporcionados.\n"
    "2. No añadas información médica que no esté en los contextos.\n"
    "3. Si los contextos no contienen información suficiente para responder la "
    'pregunta, responde exactamente: "No tengo información suficiente en mi base '
    'de conocimiento para responder esa pregunta. Por favor, reformula la consulta '
    'o selecciona otra categoría."\n'
    "4. Responde en español, de forma concisa y clínica.\n"
    "5. No inventes datos, dosis ni recomendaciones que no estén en los contextos."
)


def recargar():
    """Reconstruye los índices TF-IDF después de añadir entradas al dataset."""
    global _df, _vec_p, _mat_p, _vec_pc, _mat_pc
    _df, _vec_p, _mat_p, _vec_pc, _mat_pc = _cargar()


def _openai_responder(pregunta: str, contextos: list) -> tuple:
    """
    Genera una respuesta grounded usando OpenAI.
    Devuelve (respuesta: str, estado: str).
    Estado: 'ok' | 'no_info' | 'error_api' | 'sin_clave'.
    """
    key = _get_openai_key()
    if not key:
        return "", "sin_clave"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        contexto_texto = "\n\n".join(
            f"Contexto {i + 1}:\n{ctx['contexto']}"
            for i, ctx in enumerate(contextos)
        )
        user_message = (
            f"Pregunta del médico: {pregunta}\n\n"
            f"Contextos clínicos disponibles:\n{contexto_texto}\n\n"
            "Responde basándote ÚNICAMENTE en los contextos proporcionados."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=300,
            temperature=0.1,
        )

        respuesta = response.choices[0].message.content.strip()
        estado = "no_info" if respuesta.startswith("No tengo información suficiente") else "ok"
        return respuesta, estado

    except Exception:
        return "", "error_api"


def responder(pregunta: str) -> dict:
    """
    Devuelve un dict con 'respuesta' (str) y 'confianza' (float 0-1).

    Flujo:
      1. TF-IDF selecciona los TOP_K contextos más relevantes del dataset.
      2. Si TF-IDF supera el umbral mínimo, OpenAI genera la respuesta basada
         estrictamente en esos contextos (RAG).
      3. Si OpenAI determina que el contexto no es suficiente, se devuelve
         el mensaje de fallback.
      4. Si TF-IDF no supera el umbral mínimo, se devuelve el fallback directamente.
    """
    # — Etapa 1: TF-IDF retrieval (dos matrices) —
    q_norm = _normalizar(pregunta)

    # Primero: solo pregunta — scores altos para coincidencias directas
    sim_p   = cosine_similarity(_vec_p.transform([q_norm]), _mat_p).flatten()
    top_p   = sim_p.argsort()[::-1][:_TOP_K]
    score_p = float(sim_p[top_p[0]])

    # Si pregunta-sola no supera el umbral, buscar también en pregunta+contexto
    if score_p < _TFIDF_PREGUNTA_MIN:
        sim_pc   = cosine_similarity(_vec_pc.transform([q_norm]), _mat_pc).flatten()
        top_pc   = sim_pc.argsort()[::-1][:_TOP_K]
        score_pc = float(sim_pc[top_pc[0]])
        if score_pc > score_p:
            top_indices     = top_pc
            sim_activa      = sim_pc
            confianza_tfidf = score_pc
        else:
            top_indices     = top_p
            sim_activa      = sim_p
            confianza_tfidf = score_p
    else:
        top_indices     = top_p
        sim_activa      = sim_p
        confianza_tfidf = score_p

    if confianza_tfidf < _TFIDF_UMBRAL:
        return {"respuesta": FALLBACK, "confianza": round(confianza_tfidf, 3), "categoria": "Sin categoría"}

    # — Etapa 2: OpenAI RAG —
    contextos = [
        {"contexto": _df.iloc[idx]["contexto"]}
        for idx in top_indices
        if sim_activa[idx] >= _TFIDF_UMBRAL
    ]

    respuesta_openai, openai_estado = _openai_responder(pregunta, contextos)

    mejor_fila = _df.iloc[top_indices[0]]

    if openai_estado == "ok":
        respuesta_final = respuesta_openai
        openai_usado    = True
    elif openai_estado in ("no_info",):
        respuesta_final = FALLBACK
        openai_usado    = False
    else:
        # sin_clave o error_api → usar respuesta pre-escrita del dataset
        respuesta_final = mejor_fila["respuesta"]
        openai_usado    = False

    return {
        "respuesta":        respuesta_final,
        "confianza":        round(confianza_tfidf, 3),
        "categoria":        mejor_fila["categoria"],
        "openai_usado":     openai_usado,
        "openai_estado":    openai_estado,
        "pregunta_dataset": mejor_fila["pregunta"],
        "total_entradas":   len(_df),
    }
