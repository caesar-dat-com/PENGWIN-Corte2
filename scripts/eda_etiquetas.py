"""EDA de fragmentos sobre las 100 máscaras de referencia.

Uso:  python scripts/eda_etiquetas.py --etiquetas data/raw/labels --salida salidas
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pengwin import eda  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiquetas", default="data/raw/labels")
    ap.add_argument("--salida", default="salidas")
    ap.add_argument("--procesos", type=int, default=3)
    a = ap.parse_args()

    rutas = sorted(Path(a.etiquetas).glob("*.mha"))
    with Pool(a.procesos) as pool:
        res = pool.map(eda.resumen_caso, rutas, chunksize=1)
    casos = pd.DataFrame([c for c, _ in res])
    frags = pd.DataFrame([f for _, fs in res for f in fs])
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    casos.to_csv(out / "eda_casos.csv", index=False)
    frags.to_csv(out / "eda_fragmentos.csv", index=False)
    print(f"{len(casos)} casos, {len(frags)} fragmentos -> {out}")


if __name__ == "__main__":
    main()
