"""Descarga quirúrgica y extracción del volumen CT real de Caso 001 desde Zenodo.

Aprovecha que 001.mha es el primer archivo dentro de PENGWIN_CT_train_images_part1.zip
para descargar exactamente sus 153.5 MB sin tener que bajar los 3.9 GB del archivo completo.
"""
import os
import urllib.request
import zlib
from pathlib import Path

OUT_DIR = Path("data/raw/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "001.mha"

if OUT_FILE.exists() and OUT_FILE.stat().st_size > 400_000_000:
    print(f"001.mha ya existe con {OUT_FILE.stat().st_size} bytes.")
    exit(0)

URL = "https://zenodo.org/api/records/10927452/files/PENGWIN_CT_train_images_part1.zip/content"
START_BYTE = 37
COMP_SIZE = 153565681
END_BYTE = START_BYTE + COMP_SIZE - 1

print(f"Descargando 001.mha real desde Zenodo (bytes {START_BYTE}-{END_BYTE}, {COMP_SIZE/1e6:.1f} MB)...")

req = urllib.request.Request(URL)
req.headers["Range"] = f"bytes={START_BYTE}-{END_BYTE}"

decompressor = zlib.decompressobj(-15)  # raw deflate stream
bytes_downloaded = 0

with urllib.request.urlopen(req) as resp, open(OUT_FILE, "wb") as out_f:
    while True:
        chunk = resp.read(1024 * 1024)  # 1 MB chunk
        if not chunk:
            break
        bytes_downloaded += len(chunk)
        uncompressed = decompressor.decompress(chunk)
        out_f.write(uncompressed)
        pct = (bytes_downloaded / COMP_SIZE) * 100
        print(f"\rProgreso: {bytes_downloaded/1e6:.1f} / {COMP_SIZE/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)

    remainder = decompressor.flush()
    if remainder:
        out_f.write(remainder)

print(f"\n✓ 001.mha extraído exitosamente en: {OUT_FILE}")
print(f"Tamaño final descomprimido: {OUT_FILE.stat().st_size / 1e6:.1f} MB")
