"""Módulo de persistencia: guarda y recupera consultas en MySQL."""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def _conectar():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def guardar(categoria: str, transcripcion: str, respuesta: str,
            confianza: float = None, duracion_ms: int = None):
    conn = _conectar()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO consultas (categoria, transcripcion, respuesta, confianza, duracion_ms) "
        "VALUES (%s, %s, %s, %s, %s)",
        (categoria, transcripcion, respuesta, confianza, duracion_ms),
    )
    conn.commit()
    cursor.close()
    conn.close()


def obtener_historial(limite: int = 20) -> list[dict]:
    conn = _conectar()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT fecha_hora, categoria, transcripcion, respuesta, confianza "
        "FROM consultas ORDER BY fecha_hora DESC LIMIT %s",
        (limite,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
