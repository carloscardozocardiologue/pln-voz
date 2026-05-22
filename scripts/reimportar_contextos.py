"""
Reimporta contextos corregidos desde data/dosis_para_revisar.csv
al dataset principal data/preguntas_cardiologia_esc_500.csv.

Uso:
    python3 scripts/reimportar_contextos.py

Solo actualiza las columnas 'respuesta' y 'contexto' de las filas cuyo
'id' aparezca en el CSV de revisión. El resto del dataset no se toca.
"""

import pandas as pd
from pathlib import Path

_DATASET   = Path(__file__).parent.parent / "data" / "preguntas_cardiologia_esc_500.csv"
_REVISADOS = Path(__file__).parent.parent / "data" / "dosis_para_revisar.csv"

df_main = pd.read_csv(_DATASET)
df_rev  = pd.read_csv(_REVISADOS)

actualizados = 0
for _, fila in df_rev.iterrows():
    mask = df_main["id"] == fila["id"]
    if not mask.any():
        print(f"  AVISO: id {fila['id']} no encontrado en el dataset principal.")
        continue
    df_main.loc[mask, "pregunta"]  = fila["pregunta"]
    df_main.loc[mask, "respuesta"] = fila["respuesta"]
    df_main.loc[mask, "contexto"]  = fila["contexto"]
    actualizados += 1

df_main.to_csv(_DATASET, index=False)
print(f"Listo. {actualizados} entradas actualizadas en el dataset principal.")
print("Reinicia la app (o usa el botón de la pestaña Conocimiento) para reconstruir el índice TF-IDF.")
