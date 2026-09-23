"""Genera notebooks/Semana1_Datos_EDA_MIP.ipynb (fuente única: este archivo)."""
from pathlib import Path

import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Proyecto Integrador Corte 2 — PENGWIN · Semana 1 (semana 8 del curso)

**Sistema de detección, segmentación y clasificación de fragmentos pélvicos en CT**
Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro

> ⚠️ **Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido validado
> clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Entregable verificable de esta semana (sección 6 del enunciado):

| # | Entregable | Sección |
|---|---|---|
| 1 | Pipeline de carga del volumen + espaciado físico del header | §1 |
| 2 | Dataset curado: control de calidad CT ↔ máscara | §2 |
| 3 | Análisis exploratorio (EDA) de fragmentos por caso | §3 |
| 4 | Splits fijos train/val/test por paciente | §4 |
| 5 | Ventaneo HU funcionando | §5 |
| 6 | Visualizador 1: MIP del hueso por umbral HU | §6 |
""")

md(r"""
## 0. Entorno

Funciona en **Colab** (monta Drive) y en local. En Colab, la carpeta del proyecto
(`PENGWIN-Corte2/`, con `pengwin/` adentro) debe estar en el Drive del equipo, junto a las
carpetas `PENGWIN_CT_train_images part1` y `Part2`. Si la carpeta está *compartida contigo*,
primero: clic derecho → *Organizar → Agregar acceso directo* a Mi unidad.
""")

code(r"""
import os, sys, glob, json, zipfile, urllib.request
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    !pip -q install SimpleITK plotly

def buscar(patron, raices):
    for r in raices:
        hits = sorted(glob.glob(os.path.join(r, patron), recursive=True))
        if hits:
            return Path(hits[0])
    return None

if EN_COLAB:
    RAICES = ["/content/drive/MyDrive", "/content/drive/Shareddrives"]
    DIR_IMG_1 = buscar("**/PENGWIN_CT_train_images part1", RAICES)
    DIR_IMG_2 = buscar("**/PENGWIN_CT_train_images Part2", RAICES)
    RAIZ = buscar("**/PENGWIN-Corte2", RAICES) or Path("/content/PENGWIN-Corte2")
    DIRS_IMAGENES = [d for d in (DIR_IMG_1, DIR_IMG_2) if d]
    DIR_ETIQUETAS = Path("/content/labels")          # disco local de Colab (13 GB descomprimido)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    DIRS_IMAGENES = [RAIZ / "data/raw/images"]
    DIR_ETIQUETAS = RAIZ / "data/raw/labels"

sys.path.insert(0, str(RAIZ))
SALIDAS = RAIZ / "salidas"; SALIDAS.mkdir(exist_ok=True)
print("Proyecto :", RAIZ)
print("Imágenes :", DIRS_IMAGENES)
print("Máscaras :", DIR_ETIQUETAS)
""")

code(r"""
# Las máscaras NO están en el Drive: se bajan de Zenodo (registro 10927452, 33 MB comprimidas).
URL_LABELS = "https://zenodo.org/api/records/10927452/files/PENGWIN_CT_train_labels.zip/content"
if not any(DIR_ETIQUETAS.glob("*.mha")):
    zip_local = DIR_ETIQUETAS.parent / "PENGWIN_CT_train_labels.zip"
    if not zip_local.exists():
        urllib.request.urlretrieve(URL_LABELS, zip_local)
    zipfile.ZipFile(zip_local).extractall(DIR_ETIQUETAS)

RUTA_CT = {p.stem: p for d in DIRS_IMAGENES for p in d.glob("*.mha")}
RUTA_LB = {p.stem: p for p in DIR_ETIQUETAS.glob("*.mha")}
print(f"CT disponibles: {len(RUTA_CT)} · máscaras: {len(RUTA_LB)} · con ambas: {len(RUTA_CT.keys() & RUTA_LB.keys())}")
""")

code(r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import SimpleITK as sitk
from pengwin import io, eda, splits, viz_mip

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 160)
""")

md(r"""
## 1. Carga del volumen y espaciado físico

Los archivos del challenge están en **MetaImage `.mha`**, convertidos desde NIfTI: el header
conserva los campos NIfTI originales (`pixdim[1..3]`, `scl_slope`, `qform/sform`). SimpleITK lee el
espaciado (mm/vóxel) de ese header; es el que se usa en **todas** las medidas en mm.

Dos decisiones de carga:
- **Reorientación a LPS** (`sitk.DICOMOrient`). Los casos no comparten dirección (p. ej. el 001 es
  RAI y el 002 es LPI); sin reorientar, la mitad de los volúmenes queda espejada y un modelo 2D
  aprendería la lateralidad (coxal izq./der.) al revés.
- El array numpy queda en orden **(z, y, x)** y el espaciado se devuelve en ese mismo orden.
""")

code(r"""
CASO_DEMO = sorted(RUTA_CT.keys() & RUTA_LB.keys())[0]
caso = io.cargar_caso(CASO_DEMO, RUTA_CT[CASO_DEMO].parent, DIR_ETIQUETAS)
print(f"Caso {caso.id}: forma (z,y,x) = {caso.ct.shape}")
print(f"Espaciado (z,y,x) = {tuple(round(s, 4) for s in caso.spacing)} mm")
print(f"Dirección original = {caso.direccion_original[::4]} (diagonal) -> reorientado a {io.ORIENTACION}")
print(f"HU: min {caso.ct.min()}, max {caso.ct.max()}")
print("Etiquetas presentes:", {int(v): io.REGIONES[io.region_de_etiqueta(v)]['sigla'] + f"-{io.fragmento_de_etiqueta(v)}"
                               for v in np.unique(caso.etiqueta) if v})
""")

md(r"""
## 2. Curación: control de calidad CT ↔ máscara

Para cada caso, solo con los headers (sin cargar vóxeles): tamaño, espaciado, origen y dirección
deben coincidir entre CT y máscara; el `scl_slope` debe ser 1 (los valores ya son HU) y el
`ElementSpacing` debe coincidir con el `pixdim` NIfTI original.
""")

code(r"""
def qc_caso(cid):
    fila = {"caso": cid}
    lb = io.leer_header(RUTA_LB[cid])
    fila.update({k: lb[k] for k in ("nx", "ny", "nz", "sx_mm", "sy_mm", "sz_mm")})
    fila["dir_diag"] = str(lb["direccion"])
    if cid in RUTA_CT:
        ct = io.leer_header(RUTA_CT[cid])
        fila["tam_ok"] = (ct["nx"], ct["ny"], ct["nz"]) == (lb["nx"], lb["ny"], lb["nz"])
        fila["spacing_ok"] = np.allclose([ct["sx_mm"], ct["sy_mm"], ct["sz_mm"]],
                                         [lb["sx_mm"], lb["sy_mm"], lb["sz_mm"]], atol=1e-3)
        fila["dir_ok"] = ct["direccion"] == lb["direccion"]
        fila["slope_1"] = ct["scl_slope"] == 1 and ct["scl_inter"] == 0
        r = sitk.ImageFileReader(); r.SetFileName(str(RUTA_CT[cid])); r.ReadImageInformation()
        pix = [float(r.GetMetaData(f"pixdim[{i}]")) for i in (1, 2, 3)] if r.HasMetaDataKey("pixdim[1]") else None
        fila["pixdim_ok"] = pix is None or np.allclose(pix, r.GetSpacing(), atol=1e-4)
    return fila

qc = pd.DataFrame([qc_caso(c) for c in sorted(RUTA_LB)])
cols_ok = [c for c in ("tam_ok", "spacing_ok", "dir_ok", "slope_1", "pixdim_ok") if c in qc]
qc["apto"] = qc[cols_ok].all(axis=1) if cols_ok else True
qc.to_csv(SALIDAS / "qc_headers.csv", index=False)
print(f"Casos revisados: {len(qc)} · con CT: {qc[cols_ok[0]].notna().sum() if cols_ok else 0} · aptos: {int(qc['apto'].sum())}")
print("\nDirecciones (diagonal) encontradas:\n", qc["dir_diag"].value_counts().to_string())
print("\nEspaciado (mm):\n", qc[["sx_mm", "sy_mm", "sz_mm"]].describe().round(3).to_string())
qc[~qc["apto"]]
""")

md(r"""
## 3. EDA de fragmentos por caso

Taxonomía del dataset (sección 2.1): **1–10 sacro (SA)**, **11–20 coxal izquierdo (LI)**,
**21–30 coxal derecho (RI)**. El valor x1 es el fragmento principal; un hueso sin fractura es un
único fragmento. Por fragmento se calcula volumen (mL, con el espaciado del header), número de
componentes conexas y la **distancia mínima al principal en mm**
(`scipy.ndimage.distance_transform_edt` con `sampling=spacing`) sobre el ground truth: es la
referencia contra la que se comparará la distancia predicha en la semana 10.

Corre sobre las 100 máscaras (≈ 10 min en Colab). Si ya existe `salidas/eda_casos.csv`, se reutiliza.
""")

code(r"""
if (SALIDAS / "eda_casos.csv").exists() and (SALIDAS / "eda_fragmentos.csv").exists():
    casos = pd.read_csv(SALIDAS / "eda_casos.csv", dtype={"caso": str})
    frags = pd.read_csv(SALIDAS / "eda_fragmentos.csv", dtype={"caso": str})
else:
    res = [eda.resumen_caso(RUTA_LB[c]) for c in sorted(RUTA_LB)]
    casos = pd.DataFrame([c for c, _ in res]); frags = pd.DataFrame([f for _, fs in res for f in fs])
    casos.to_csv(SALIDAS / "eda_casos.csv", index=False); frags.to_csv(SALIDAS / "eda_fragmentos.csv", index=False)
casos["caso"] = casos["caso"].str.zfill(3); frags["caso"] = frags["caso"].str.zfill(3)
print(f"{len(casos)} casos · {len(frags)} fragmentos · {int((~frags['es_principal']).sum())} fragmentos no principales")
casos.head()
""")

code(r"""
SIG = [i["sigla"] for i in io.REGIONES.values()]
NOMBRE = {i["sigla"]: i["nombre"] for i in io.REGIONES.values()}
COLOR = {i["sigla"]: i["color"] for i in io.REGIONES.values()}

tabla = pd.DataFrame({
    "casos con la región": [(casos[f"n_frag_{s}"] > 0).sum() for s in SIG],
    "casos fracturados": [(casos[f"n_frag_{s}"] > 1).sum() for s in SIG],
    "fragmentos (media)": [casos[f"n_frag_{s}"].mean().round(2) for s in SIG],
    "fragmentos (máx)": [casos[f"n_frag_{s}"].max() for s in SIG],
    "volumen mL (mediana)": [casos[f"vol_ml_{s}"].median().round(1) for s in SIG],
    "principal = el mayor": [f"{casos[f'principal_es_mayor_{s}'].mean():.0%}" for s in SIG],
}, index=[NOMBRE[s] for s in SIG])
tabla
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
maxf = int(casos[[f"n_frag_{s}" for s in SIG]].max().max())
bins = np.arange(0.5, maxf + 1.5)
for s in SIG:
    ax[0].hist(casos[f"n_frag_{s}"], bins=bins, alpha=.6, label=NOMBRE[s], color=COLOR[s])
ax[0].set(title="Fragmentos por región y caso", xlabel="n.º de fragmentos", ylabel="casos"); ax[0].legend()

ax[1].hist(casos["regiones_fracturadas"], bins=np.arange(-.5, 4), rwidth=.8, color="gray")
ax[1].set(title="Huesos fracturados por caso", xlabel="regiones con >1 fragmento", ylabel="casos", xticks=range(4))

nop = frags[~frags["es_principal"]]
for s in SIG:
    v = nop.loc[nop["region"] == s, "vol_ml"]
    if len(v): ax[2].hist(np.log10(v), bins=25, alpha=.6, color=COLOR[s], label=NOMBRE[s])
ax[2].set(title="Volumen de fragmentos no principales", xlabel="log10(mL)", ylabel="fragmentos"); ax[2].legend()
fig.tight_layout(); fig.savefig(SALIDAS / "eda_fragmentos.png", dpi=120)
""")

md(r"""
### 3.1 Separación al fragmento principal (ground truth)

`dist_mm_gt` es la distancia mínima entre vóxeles del fragmento y del principal. Si se tocan,
vale ≈ un vóxel (0.7–1 mm): `en_contacto` marca los que están a una diagonal de vóxel o menos.
""")

code(r"""
nop = frags[~frags["es_principal"]].copy()
nop["en_contacto"] = nop["en_contacto"].astype(str).eq("True")   # el CSV lo lee como texto
print(f"Fragmentos no principales: {len(nop)}")
print(f"En contacto con el principal: {nop['en_contacto'].mean():.1%}")
print(f"Con más de una componente conexa: {(nop['n_componentes'] > 1).mean():.1%}")
print("\nDistancia al principal (mm) por región:")
print(nop.groupby("region")["dist_mm_gt"].describe().round(2).rename(index=NOMBRE).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sep = nop[~nop["en_contacto"]]
for s in SIG:
    d = sep.loc[sep["region"] == s, "dist_mm_gt"]
    if len(d): ax[0].hist(d, bins=30, alpha=.6, color=COLOR[s], label=f"{NOMBRE[s]} (n={len(d)})")
ax[0].set(title="Separación de fragmentos que NO tocan al principal", xlabel="mm", ylabel="fragmentos"); ax[0].legend()
ax[1].scatter(nop["vol_ml"], nop["dist_mm_gt"], c=nop["region"].map(COLOR), s=14, alpha=.7)
ax[1].set(xscale="log", title="Volumen vs. separación", xlabel="volumen (mL, log)", ylabel="distancia (mm)")
fig.tight_layout(); fig.savefig(SALIDAS / "eda_separacion.png", dpi=120)
""")

md(r"""
### 3.2 Cortes axiales por región (desbalance 2D)

El modelo trabaja corte a corte: lo que importa para la cabeza de clasificación es cuántos
**cortes** contienen cada región, no cuántos casos.
""")

code(r"""
cortes = pd.DataFrame({NOMBRE[s]: casos[f"cortes_{s}"] for s in SIG})
cortes["Con algún hueso"] = casos["cortes_con_hueso"]; cortes["Total del volumen"] = casos["nz"]
print(cortes.sum().to_string())
print(f"\nFracción de cortes sin hueso anotado: {1 - casos['cortes_con_hueso'].sum() / casos['nz'].sum():.1%}")
cortes.describe().round(1)
""")

md(r"""
### 3.3 Intensidad del hueso anotado (justifica ventana y umbral)

Percentiles de HU dentro de la máscara de referencia en una muestra de casos. Si una fracción
importante del hueso anotado está por debajo del umbral clásico (200 HU), ese umbral sirve para el
MIP (que muestra el máximo, la cortical) pero **no** como segmentación: el hueso trabecular del sacro
queda por debajo. Por eso la red recibe una ventana ancha y no una máscara umbralizada.
""")

code(r"""
N_CASOS_HU = 12
disponibles = sorted(RUTA_CT.keys() & RUTA_LB.keys())
muestra = list(np.random.default_rng(SEED).choice(disponibles, min(N_CASOS_HU, len(disponibles)), replace=False))
hu = pd.DataFrame([eda.estadisticas_hu(RUTA_CT[c], RUTA_LB[c]) for c in sorted(muestra)])
hu.to_csv(SALIDAS / "eda_hu_hueso.csv", index=False)
print(f"Hueso anotado bajo 200 HU (media de {len(hu)} casos): {hu['frac_hueso_bajo_200HU'].mean():.1%}")
hu.round(1)
""")

md(r"""
## 4. Splits fijos por paciente

La unidad del split es el **caso completo**: cortes vecinos de un mismo CT son casi idénticos y
repartirlos entre train y test inflaría las métricas. Estratificado por cuántos huesos están
fracturados (70/15/15, semilla 42) para que val y test tengan casos multifragmento.
Solo se incluyen los casos que pasaron el QC. El archivo `splits/splits.json` queda congelado en el
repositorio; nadie vuelve a sortearlo.
""")

code(r"""
aptos = set(qc.loc[qc["apto"], "caso"]) if "tam_ok" in qc and qc["tam_ok"].notna().all() else set(casos["caso"])
base = casos[casos["caso"].isin(aptos)].copy()
base["estrato"] = splits.estrato_por_complejidad(base)
ruta_splits = RAIZ / "splits/splits.json"
if ruta_splits.exists():
    sp = splits.cargar(ruta_splits); print("splits.json existente (congelado) -> se reutiliza")
else:
    sp = splits.estratificar(base, "estrato")
    splits.guardar(sp, ruta_splits, {"proporciones": [0.70, 0.15, 0.15], "unidad": "caso",
                                     "estrato": "regiones_fracturadas (0-3, fusionando estratos < 7 casos)"})
assert not (set(sp["train"]) & set(sp["val"]) or set(sp["train"]) & set(sp["test"]) or set(sp["val"]) & set(sp["test"]))

resumen = []
for nombre in ("train", "val", "test"):
    d = base[base["caso"].isin(sp[nombre])]
    resumen.append({"split": nombre, "casos": len(d), "cortes": int(d["nz"].sum()),
                    "fragmentos": int(d["n_frag_total"].sum()),
                    **{f"fract. {s}": int((d[f"n_frag_{s}"] > 1).sum()) for s in SIG},
                    "frag/caso": round(d["n_frag_total"].mean(), 2)})
pd.DataFrame(resumen).set_index("split")
""")

md(r"""
## 5. Ventaneo HU

`io.ventana_hu(ct, nivel, ancho)` recorta a `[L − W/2, L + W/2]` y normaliza a [0, 1]: es la
entrada de la red. Ventana de hueso L = 400, W = 1800 → [−500, 1300] HU.
""")

code(r"""
# corte con más vóxeles de fragmentos no principales (si no hay fractura, el de más hueso)
no_principal = (caso.etiqueta > 0) & ((caso.etiqueta.astype(int) - 1) % 10 > 0)
z = int(np.argmax(no_principal.sum(axis=(1, 2)) if no_principal.any() else (caso.etiqueta > 0).sum(axis=(1, 2))))
reg = np.vectorize(io.region_de_etiqueta)(caso.etiqueta[z])
from matplotlib.colors import ListedColormap
cmap_reg = ListedColormap(["none"] + [i["color"] for i in io.REGIONES.values()])

fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
ax[0].imshow(caso.ct[z], cmap="gray"); ax[0].set_title("HU crudo")
ax[1].imshow(io.ventana_hu(caso.ct[z], *io.VENTANAS["tejido_blando"]), cmap="gray"); ax[1].set_title("Ventana tejido blando (40/400)")
ax[2].imshow(io.ventana_hu(caso.ct[z]), cmap="gray"); ax[2].set_title("Ventana hueso (400/1800)")
ax[3].imshow(io.ventana_hu(caso.ct[z]), cmap="gray"); ax[3].imshow(reg, cmap=cmap_reg, vmin=0, vmax=3, alpha=.45, interpolation="nearest")
ax[3].set_title("Ventana hueso + GT por región")
for a in ax: a.axis("off")
fig.suptitle(f"Caso {caso.id} · corte axial z={z} · uso académico, no clínico"); fig.tight_layout()
fig.savefig(SALIDAS / f"ventaneo_{caso.id}.png", dpi=110)
""")

md(r"""
## 6. Visualizador 1 — MIP del volumen crudo (solo hueso)

Preprocesamiento clásico, **sin modelo**: umbral de hueso en HU sobre el volumen sin procesar,
remuestreo a vóxel isotrópico de 2 mm (solo para que la proyección no se deforme) y proyección de
máxima intensidad. Girando el volumen alrededor del eje cráneo-caudal se obtiene la vista 3D; el
slider recorre los ángulos y ▶ lo anima. Se exporta como HTML autocontenido que el dashboard de la
semana 11 incrusta tal cual.
""")

code(r"""
UMBRAL_HU = 200
viz_mip.figura_tres_vistas(caso.ct, caso.spacing, caso.id, UMBRAL_HU).savefig(SALIDAS / f"mip3_{caso.id}.png", dpi=110)
fig = viz_mip.figura_mip(caso.ct, caso.spacing, caso.id, umbral_hu=UMBRAL_HU)
fig.write_html(SALIDAS / f"visualizador1_mip_{caso.id}.html", include_plotlyjs="cdn")
fig.show()
""")

md(r"""
## Resumen de la semana

Archivos generados en `salidas/` y `splits/`:
`qc_headers.csv`, `eda_casos.csv`, `eda_fragmentos.csv`, `eda_hu_hueso.csv`, `eda_*.png`,
`ventaneo_*.png`, `mip3_*.png`, `visualizador1_mip_*.html`, `splits/splits.json`.

Siguiente semana (9): backbone (extensión de `FundidoraPC`) + CBAM + cabeza de detección con grid y
NMS propios; overfit intencional sobre un batch pequeño.
""")

nb = nbf.v4.new_notebook(cells=C, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "colab": {"provenance": []}, "accelerator": "GPU"})
out = Path(__file__).resolve().parents[1] / "notebooks/Semana1_Datos_EDA_MIP.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(out)
