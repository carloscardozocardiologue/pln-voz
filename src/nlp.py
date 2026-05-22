"""
Pipeline NLP: TF-IDF retrieval + BERT QA en español.

Etapa 1 — TF-IDF: encuentra los contextos más similares a la pregunta del médico.
Etapa 2 — BERT QA: extrae la respuesta exacta del contexto usando el modelo
  mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es (vía HF API).
Fallback: si BERT no supera el umbral de confianza, devuelve la respuesta
  pre-escrita del dataset.
"""

import os
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
_TFIDF_UMBRAL      = 0.15   # mínimo para intentar responder
_TFIDF_BERT_UMBRAL = 0.30   # mínimo para usar BERT (retrieval suficientemente confiable)
_BERT_UMBRAL       = 0.15   # confianza mínima del modelo BERT
_TOP_K        = 2


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes para que TF-IDF no distinga síntomas/sintomas."""
    texto = texto.lower()
    nfkd  = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _cargar():
    df  = pd.read_csv(_DATA_PATH)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    mat = vec.fit_transform(df["pregunta"].apply(_normalizar))
    return df, vec, mat


def _get_hf_token() -> str:
    try:
        import streamlit as st
        val = st.secrets.get("HF_TOKEN")
        if val:
            return val
    except Exception:
        pass
    return os.getenv("HF_TOKEN", "")


_df, _vectorizador, _matriz = _cargar()
_client = InferenceClient(token=_get_hf_token())

FALLBACK = (
    "No tengo información suficiente en mi base de conocimiento para responder "
    "esa pregunta. Por favor, reformula la consulta o selecciona otra categoría."
)


def recargar():
    """Reconstruye el índice TF-IDF después de añadir entradas al dataset."""
    global _df, _vectorizador, _matriz
    _df, _vectorizador, _matriz = _cargar()


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
    # — Etapa 1: TF-IDF retrieval —
    vec_pregunta = _vectorizador.transform([_normalizar(pregunta)])
    similitudes  = cosine_similarity(vec_pregunta, _matriz).flatten()
    top_indices  = similitudes.argsort()[::-1][:_TOP_K]
    confianza_tfidf = float(similitudes[top_indices[0]])

    if confianza_tfidf < _TFIDF_UMBRAL:
        return {"respuesta": FALLBACK, "confianza": round(confianza_tfidf, 3), "categoria": "Sin categoría"}

    # — Etapa 2: BERT QA — solo si TF-IDF es suficientemente confiable —
    respuesta_bert = None
    score_bert     = 0.0
    bert_estado    = "no_aplicable"   # "no_aplicable" | "error_api" | "score_bajo" | "ok"

    if confianza_tfidf >= _TFIDF_BERT_UMBRAL:
        bert_estado = "error_api"
        for idx in top_indices:
            if similitudes[idx] < _TFIDF_UMBRAL:
                break
            contexto = _df.iloc[idx]["contexto"]
            try:
                resultado = _client.question_answering(
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
    }
