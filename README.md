# PENGWIN — Proyecto Integrador Corte 2

Detección, segmentación y clasificación de fragmentos pélvicos en CT (dataset
[PENGWIN Task 1](https://pengwin.grand-challenge.org/), Zenodo
[10927452](https://zenodo.org/records/10927452)).
Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro.

> ⚠️ **Uso exclusivamente académico.** No es un dispositivo médico, no está validado
> clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Estado

| Semana | Entregable | Estado |
|---|---|---|
| 8 (sem. 1) | Splits fijos, carga + ventaneo HU, EDA, visualizador 1 (MIP) | ✅ |
| 9 | Backbone FundidoraPC + CBAM + cabeza de detección, overfit de un batch | — |
| 10 | Pipeline completo, distancia en mm, SAM, métricas §5 | — |
| 11 | Visualizadores 2 y 3, dashboard, túnel Cloudflare, model card, pitch | — |

## Datos

- **CT** (`.mha`, 100 casos): carpeta del equipo en Drive
  (`PENGWIN_CT_train_images part1` = 001–050, `Part2` = 051–100).
- **Máscaras**: `PENGWIN_CT_train_labels.zip` de Zenodo (33 MB; 13 GB descomprimido).
  El notebook la descarga sola.
- Los `.mha` son conversiones de NIfTI y conservan su header (`pixdim`, `scl_slope`);
  el espaciado en mm sale de ahí. Etiquetas: 1–10 sacro, 11–20 coxal izq., 21–30 coxal der.;
  x1 = fragmento principal.

## Reproducir la semana 1

**Colab**: subir esta carpeta (sin `data/`) al Drive del equipo y abrir
`notebooks/Semana1_Datos_EDA_MIP.ipynb`. El notebook ubica solo las carpetas de imágenes.

**Local**:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# máscaras
mkdir -p data/raw && curl -L -o data/raw/labels.zip \
  https://zenodo.org/api/records/10927452/files/PENGWIN_CT_train_labels.zip/content
unzip -q data/raw/labels.zip -d data/raw/labels
# CT: copiar los .mha del Drive a data/raw/images/
python scripts/eda_etiquetas.py            # EDA de las 100 máscaras -> salidas/
python scripts/construir_notebook.py       # regenera el .ipynb desde su fuente
jupyter nbconvert --execute --to notebook --inplace notebooks/Semana1_Datos_EDA_MIP.ipynb
```

Semilla global 42. `splits/splits.json` está **congelado**: el notebook lo reutiliza si existe.

## Estructura

```
pengwin/io.py        carga .mha, reorientación LPS, spacing (z,y,x), ventaneo HU, taxonomía
pengwin/eda.py       resumen por caso/fragmento, distancia EDT en mm, HU del hueso
pengwin/splits.py    split estratificado por paciente
pengwin/viz_mip.py   visualizador 1: MIP rotatorio del hueso por umbral HU (Plotly)
scripts/             EDA por lotes y generador del notebook
notebooks/           notebook de la semana
salidas/             CSV, figuras y HTML del visualizador 1
splits/splits.json   partición fija train/val/test
```
