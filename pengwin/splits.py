"""Splits fijos train/val/test POR PACIENTE (nunca por corte).

Los cortes vecinos de un mismo CT son casi idénticos; si se reparten cortes
del mismo paciente entre train y test, las métricas salen infladas. Por eso
la unidad del split es el caso completo y el archivo se congela en
`splits/splits.json` para que todo el equipo use exactamente la misma partición.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def estratificar(df: pd.DataFrame, columna: str = "estrato",
                 proporciones=(0.70, 0.15, 0.15), seed: int = SEED) -> dict:
    """Reparte cada estrato con las mismas proporciones (redondeo al más cercano)."""
    rng = np.random.default_rng(seed)
    salida = {"train": [], "val": [], "test": []}
    for _, grupo in df.groupby(columna, sort=True):
        ids = sorted(grupo["caso"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * proporciones[1]))
        n_test = int(round(n * proporciones[2]))
        salida["test"] += ids[:n_test]
        salida["val"] += ids[n_test:n_test + n_val]
        salida["train"] += ids[n_test + n_val:]
    return {k: sorted(v) for k, v in salida.items()}


def estrato_por_complejidad(df_casos: pd.DataFrame) -> pd.Series:
    """Estrato = cuántos huesos están fracturados (0..3), con 0 y 1 fusionados si son pocos.

    Garantiza que val y test tengan casos con varios fragmentos (los que
    realmente ponen a prueba la cabeza de instancia y la medición en mm).
    """
    e = df_casos["regiones_fracturadas"].clip(upper=3)
    conteo = e.value_counts()
    for valor in sorted(conteo.index):
        if conteo[valor] < 7 and valor + 1 in conteo.index:   # < 7 no alcanza para 1 en val y 1 en test
            e = e.replace(valor, valor + 1)
            conteo = e.value_counts()
    return e


def guardar(splits: dict, ruta: Path, meta: dict | None = None) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"seed": SEED, **(meta or {}), **splits}, indent=2, ensure_ascii=False))


def cargar(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text())
