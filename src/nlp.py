"""
Pipeline NLP: TF-IDF retrieval + BERT QA en español.

Etapa 1 — TF-IDF: encuentra los contextos más similares a la pregunta del médico.
Etapa 2 — BERT QA: extrae la respuesta exacta del contexto usando el modelo
  mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es (vía HF API).
Fallback: si BERT no supera el umbral de confianza, devuelve la respuesta
  pre-escrita del dataset.
"""

import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

load_dotenv()

_DATA_PATH    = Path(__file__).parent.parent / "data" / "preguntas_cardiologia_esc_500.csv"
_QA_MODEL     = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
_TFIDF_UMBRAL        = 0.15   # mínimo para intentar responder
_TFIDF_BERT_UMBRAL   = 0.30   # mínimo para usar BERT
_BERT_UMBRAL         = 0.15   # confianza mínima del modelo BERT
_TFIDF_PREGUNTA_MIN  = 0.35   # si pregunta-sola supera esto, no hace falta buscar en contexto
_TOP_K        = 2


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y prefijos médicos normalizados al español."""
    texto = texto.lower()
    # Prefijos inglés/francés → español
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


def _get_hf_token() -> str:
    try:
        import streamlit as st
        val = st.secrets.get("HF_TOKEN")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("HF_TOKEN", "")


_df, _vec_p, _mat_p, _vec_pc, _mat_pc = _cargar()

FALLBACK = (
    "No tengo información suficiente en mi base de conocimiento para responder "
    "esa pregunta. Por favor, reformula la consulta o selecciona otra categoría."
)


def recargar():
    """Reconstruye los índices TF-IDF después de añadir entradas al dataset."""
    global _df, _vec_p, _mat_p, _vec_pc, _mat_pc
    _df, _vec_p, _mat_p, _vec_pc, _mat_pc = _cargar()


def responder(pregunta: str) -> dict:
    """
    Devuelve un dict con 'respuesta' (str) y 'confianza' (float 0-1).

    Flujo:
      1. TF-IDF selecciona los TOP_K contextos más relevantes del dataset.
      2. BERT QA extrae una respuesta de cada contexto.
      3. Se devuelve la respuesta con mayor confianza BERT.
      4. Si BERT no supera el umbral, se usa la respuesta pre-escrita del dataset.
      5. Si TF-IDF tampoco supera su umbral, se devuelve el mensaje de fallback.
    """
    # — Etapa 1: TF-IDF retrieval (dos matrices) —
    q_norm  = _normalizar(pregunta)

    # Primero: solo pregunta — da scores altos para coincidencias directas
    sim_p   = cosine_similarity(_vec_p.transform([q_norm]), _mat_p).flatten()
    top_p   = sim_p.argsort()[::-1][:_TOP_K]
    score_p = float(sim_p[top_p[0]])

    # Si la pregunta-sola no supera el umbral, buscar también en pregunta+contexto
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

    # — Etapa 2: BERT QA — solo si TF-IDF es suficientemente confiable —
    respuesta_bert = None
    score_bert     = 0.0
    bert_estado    = "no_aplicable"   # "no_aplicable" | "error_api" | "score_bajo" | "ok"

    if confianza_tfidf >= _TFIDF_BERT_UMBRAL:
        bert_estado = "error_api"
        for idx in top_indices:
            if sim_activa[idx] < _TFIDF_UMBRAL:
                break
            contexto = _df.iloc[idx]["contexto"]
            try:
                resultado = InferenceClient(token=_get_hf_token()).question_answering(
                    question=pregunta,
                    context=contexto,
                    model=_QA_MODEL,
                )
                if resultado.score > score_bert:
                    score_bert     = resultado.score
                    respuesta_bert = resultado.answer
                    bert_estado    = "score_bajo" if resultado.score < _BERT_UMBRAL else "ok"
            except Exception:
                continue

    # — Etapa 3: respuesta final —
    mejor_fila      = _df.iloc[top_indices[0]]
    bert_usado      = bert_estado == "ok"
    respuesta_final = respuesta_bert if bert_usado else mejor_fila["respuesta"]
    return {
        "respuesta":        respuesta_final,
        "confianza":        round(confianza_tfidf, 3),
        "categoria":        mejor_fila["categoria"],
        "bert_usado":       bert_usado,
        "bert_estado":      bert_estado,
        "score_bert":       round(score_bert, 3),
        "pregunta_dataset": mejor_fila["pregunta"],
        "total_entradas":   len(_df),
    }
