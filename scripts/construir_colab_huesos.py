"""Genera notebooks/Colab_Correccion_Huesos.ipynb (editar aquí, no el .ipynb)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# PENGWIN · Corrección de la selección de huesos + control de anomalías (Colab GPU)

**Rama:** `fix-huesos-colab` · Entorno: *Entorno de ejecución → Cambiar tipo → GPU (T4 o mejor)*.

### Por qué fallaba la selección de huesos (diagnóstico del repo)
| # | Falla | Evidencia | Corrección en esta rama |
|---|---|---|---|
| 1 | `main` no corre: faltaba fusionar `mejoras-yesenia` (`rutas_dataset`, `normalizar_lps`, `balancear_obj`) | `ImportError` en `preparar_entrenamiento.py` | merge de `93cf1ad` |
| 2 | Muy pocos datos: 8 cortes por paciente (560 train), 6 épocas en CPU | sacro en solo 52/120 cortes de val | `--modo hueso`: todos los cortes con hueso + 15 % vacíos, GPU, 30 épocas |
| 3 | Coxal izquierdo con Dice 0 % = el decoder no separaba los dos coxales simétricos | matriz de confusión: columna «pred LI» en 0 | flip izquierda-derecha **con intercambio de etiquetas**, más datos y control de lateralidad |
| 4 | La máscara se recortaba con la caja predicha (IoU de caja ≈ 0,5) | `gate_semantic` estricto | `--gate-margin 0.1` |
| 5 | Tejido sobrante: el fondo pesaba 0,2 en la CE | ~81 mil px de fondo predichos como hueso | `--bg-weight 0.5` |
| 6 | Sobresegmentación: 13–26 instancias frente a 5–7 GT; los fragmentos pequeños se borraban | `instances.py` | suavizado en z, sin islas, **fusionar** los fragmentos pequeños con su vecino |
| 7 | `raise` cuando dos regiones caen en la misma celda de 16 px | `loss.py` | se conserva la caja mayor y se cuenta la colisión |

### Control de anomalías (`pengwin/anomalias.py`)
- **Entrada CT:** cizalla u oblicuidad, cortes > 5 mm, pocos cortes, rango HU imposible, metal.
- **Salida del modelo:**
  - lateralidad invertida respecto a la línea media; se corrige por componente;
  - hueso sobre aire (HU < −500);
  - islas de menos de 3 cortes;
  - región ausente;
  - volumen fuera del rango de train;
  - hueso hipodenso;
  - más de 10 fragmentos por hueso.

Ninguna regla usa el GT. La referencia de volúmenes se ajusta solo con train."""),
code("""#@title 1 · GPU, repo y dependencias
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os
if not os.path.exists('/content/PENGWIN-Corte2'):
    !git clone -q -b fix-huesos-colab https://github.com/caesar-dat-com/PENGWIN-Corte2.git /content/PENGWIN-Corte2
else:
    !git -C /content/PENGWIN-Corte2 pull -q
%cd /content/PENGWIN-Corte2
!pip -q install SimpleITK scikit-image opencv-python-headless nbformat
!git log --oneline -3"""),
code("""#@title 2 · (Opcional) Drive para guardar checkpoints entre sesiones
USAR_DRIVE = True  #@param {type:"boolean"}
RUN = '/content/runs'
if USAR_DRIVE:
    from google.colab import drive; drive.mount('/content/drive')
    RUN = '/content/drive/MyDrive/PENGWIN_runs'
os.makedirs(RUN, exist_ok=True); print('Salidas en', RUN)"""),
code("""#@title 3 · Datos PENGWIN desde Zenodo (≈7,6 GB, 5–15 min)
!apt-get -qq install -y aria2 > /dev/null
import json, pathlib
D = pathlib.Path('/content/data'); D.mkdir(exist_ok=True)
Z = 'https://zenodo.org/records/10927452/files/'
for f in ['PENGWIN_CT_train_labels.zip', 'PENGWIN_CT_train_images_part1.zip', 'PENGWIN_CT_train_images_part2.zip']:
    destino = D / ('labels' if 'labels' in f else 'images')
    if destino.exists() and len(list(destino.rglob('*.mha'))) >= (100 if 'labels' in f else 50 * (1 + ('part2' in f))):
        print('ya está', f); continue
    !aria2c -q -x 8 -s 8 -c -d {D} -o {f} "{Z}{f}?download=1"
    !unzip -q -o {D}/{f} -d {destino} && rm {D}/{f}
print('CT:', len(list((D/'images').rglob('*.mha'))), '· máscaras:', len(list((D/'labels').rglob('*.mha'))))
json.dump({'images': str(D/'images'), 'labels': str(D/'labels')}, open('config_datos.local.json', 'w'))"""),
code("""#@title 4 · Control de anomalías sobre los datos: referencia de volúmenes (train) y auditoría de entrada (100 CT)
import numpy as np, SimpleITK as sitk, pandas as pd
from pengwin.io import normalizar_lps, rutas_dataset, resolver_volumen
from pengwin.instances import semantic_labels
from pengwin import anomalias
splits = json.load(open('splits/splits.json')); im, lb = rutas_dataset()
vols, filas = [], []
for split in ('train', 'val', 'test'):
    for cid in splits[split]:
        img = normalizar_lps(sitk.ReadImage(str(resolver_volumen(im, cid)))); hu = sitk.GetArrayFromImage(img)
        for a in anomalias.anomalias_entrada(img, hu): filas.append({'case': cid, 'split': split, **a})
        if split == 'train':
            sem = semantic_labels(sitk.GetArrayFromImage(normalizar_lps(sitk.ReadImage(str(resolver_volumen(lb, cid))))))
            vols.append(anomalias.volumen_regiones(sem, img.GetSpacing()[::-1]))
REF = anomalias.referencia_desde_gt(vols); json.dump(REF, open(f'{RUN}/referencia_volumenes.json', 'w'), indent=2)
print('Referencia (cm³, p2–p98 train):', {anomalias.NOMBRES[r]: (round(v['p02']/1000), round(v['p98']/1000)) for r, v in REF.items()})
pd.DataFrame(filas) if filas else 'Sin anomalías de entrada'"""),
code("""#@title 5 · Preparar cortes: todos los que tienen hueso (cada 2) + 15 % vacíos
DATOS = '/content/datos_hueso'
!python scripts/preparar_entrenamiento.py --modo hueso --paso 2 --vacios 0.15 --salida {DATOS} | tail -2
m = json.load(open(f'{DATOS}/manifest.json')); print(pd.DataFrame(m).groupby('split').size())"""),
code("""#@title 6 · Entrenar en GPU (reanuda solo si se cortó la sesión)
EPOCAS = 30  #@param {type:"integer"}
SALIDA = f'{RUN}/fix_huesos_v1'
reanudar = '--resume' if os.path.exists(f'{SALIDA}/last.pth') else ''
!python scripts/entrenar_sesion2.py --data {DATOS} --output {SALIDA} --epochs {EPOCAS} --batch 32 --lr 6e-4 --cosine \\
    --bg-weight 0.5 --gate-margin 0.1 --flip --workers 2 --threads 2 {reanudar}"""),
code("""#@title 7 · Curvas y comparación contra el avance oct-08 (val)
import matplotlib.pyplot as plt
h = json.load(open(f'{SALIDA}/historial.json'))
ep = [r['epoch'] for r in h]; dice = np.array([r['val']['semantic']['dice'] for r in h])
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(ep, [r['loss']['total'] for r in h]); ax[0].set_title('pérdida train')
for i, n in enumerate(['sacro', 'coxal izq', 'coxal der']): ax[1].plot(ep, dice[:, i], label=n)
ax[1].axhline(0.0, ls=':', c='k'); ax[1].set_title('Dice val por hueso'); ax[1].legend()
ax[2].plot(ep, [r['val']['detection']['mAP50'] for r in h]); ax[2].set_title('mAP50 val'); plt.show()
best = max(h, key=lambda r: r['selection_score'])
print('Mejor época', best['epoch'])
print(pd.DataFrame({'oct-08 (CPU, 8 cortes)': [38.82, 0.00, 42.59, 44.82], 'fix_huesos_v1': [*(100*np.array(best['val']['semantic']['dice'])), 100*best['val']['detection']['mAP50']]},
                   index=['Dice sacro', 'Dice coxal izq', 'Dice coxal der', 'mAP50']).round(2))
cm = np.array(best['val']['semantic']['confusion']); print('Confusión (filas GT, columnas pred: fondo/SA/LI/RI)'); print(cm)"""),
code("""#@title 8 · Volúmenes completos de val: modelo crudo vs con control de anomalías
CASOS = splits['val'][:3]
!python scripts/colab_evaluar_volumen.py --checkpoint {SALIDA}/best.pth --cases {' '.join(CASOS)} --output {SALIDA}/volumenes \\
    --gate-margin 0.1 --referencia {RUN}/referencia_volumenes.json
t = pd.DataFrame(json.load(open(f'{SALIDA}/volumenes/tabla.json'))).set_index('case')
t.T"""),
code("""#@title 9 · Qué anomalías se detectaron y qué se corrigió
for cid in CASOS:
    inf = json.load(open(f'{SALIDA}/volumenes/{cid}/anomalias.json'))
    print(f'— {cid}: estado={inf["estado"]} acciones={inf["acciones"]}')
    for a in inf['anomalias'] + inf['anomalias_post']: print('   ', a['severidad'].upper(), a['tipo'], a['region'] or '', '·', a['detalle'])"""),
code("""#@title 10 · Visual: GT vs crudo vs corregido
CASO = CASOS[0]  #@param {type:"string"}
ld = lambda n: sitk.GetArrayFromImage(sitk.ReadImage(f'{SALIDA}/volumenes/{CASO}/{n}.mha'))
gt, cr, co = semantic_labels(ld('gt')), ld('semantica_cruda'), ld('semantica_corregida')
zs = np.flatnonzero(gt.reshape(len(gt), -1).any(1)); zs = zs[np.linspace(0, len(zs)-1, 4).astype(int)]
cmap = plt.matplotlib.colors.ListedColormap(['black', '#E4572E', '#29335C', '#17BEBB'])
fig, ax = plt.subplots(3, 4, figsize=(16, 12))
for j, z in enumerate(zs):
    for i, (n, a) in enumerate([('GT', gt), ('crudo', cr), ('corregido', co)]):
        ax[i, j].imshow(a[z], cmap=cmap, vmin=0, vmax=3, interpolation='nearest'); ax[i, j].set_title(f'{n} z={z}'); ax[i, j].axis('off')
plt.suptitle('rojo sacro · azul oscuro coxal izq · turquesa coxal der'); plt.show()"""),
md("""### Siguiente
- Si el coxal izquierdo sigue bajo, probar `--canales 10` (tope del decoder) y comparar en la misma tabla.
- Barrer la política de instancias (`--policy`) **solo en val**; luego congelar y correr test una vez.
- `--min-iou 0.3` muestra cuántos pares de distancia aparecen si el emparejamiento se relaja; reportar ambos."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                               'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
out = ROOT / 'notebooks/Colab_Correccion_Huesos.ipynb'
nbf.write(nb, out); print(out)
