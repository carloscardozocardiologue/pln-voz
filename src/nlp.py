"""Procesamiento de lenguaje natural: QA médico vía HuggingFace Inference API."""

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from src.contexto_cardio import CONTEXTOS

load_dotenv()

MODELO_QA = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"


def responder(pregunta: str, categoria: str) -> dict:
    """
    Recibe la pregunta del médico y la categoría seleccionada.
    Devuelve un dict con 'respuesta' (str) y 'confianza' (float 0-1).
    """
    cliente = InferenceClient(token=os.getenv("HF_TOKEN"))
    contexto = CONTEXTOS.get(categoria, CONTEXTOS["Síntomas"])

    resultado = cliente.question_answering(
        question=pregunta,
        context=contexto,
        model=MODELO_QA,
    )
    return {
        "respuesta": resultado.answer,
        "confianza": round(resultado.score, 3),
    }
