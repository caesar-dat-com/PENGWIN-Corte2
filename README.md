# PENGWIN — Proyecto Integrador Corte 2

Detección, segmentación y clasificación de fragmentos pélvicos en CT (dataset
[PENGWIN Task 1](https://pengwin.grand-challenge.org/), Zenodo
[10927452](https://zenodo.org/records/10927452)).
Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro.


## 1. Integrantes

> Equipo de tres integrantes autorizado por el profesor, según confirmación del usuario en esta revisión. Los tres integrantes se relacionan abajo.

| Nombre completo | Código | Correo institucional |
|---|---|---|
| César Armando Reyes Oliveros | 2236379 | cesar_armando.reyes@uao.edu.co |
| Yesenia Díaz | 2231783 |yesenia.diaz@uao.edu.co |
| Juan Pablo Maya | 2236377 |juan_pablo.maya@uao.edu.co |

> ⚠️ **Uso exclusivamente académico.** No es un dispositivo médico, no está validado
> clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Estado

| Semana | Entregable | Estado |
|---|---|---|
| 8 (sem. 1) | Splits fijos, carga + ventaneo HU, EDA, visualizador 1 (MIP) | ✅ |
| 9 (sem. 2) | Backbone FundidoraPC + CBAM + cabeza detección grid + NMS propio + overfit batch | ✅ |
| 10 | Pipeline completo, distancia en mm, SAM, métricas §5 | Integración y evaluación local; calidad y cobertura detalladas en el informe de cierre |
| 11 | Visualizadores 2 y 3, dashboard, túnel Cloudflare, model card, pitch | Dashboard, model cards, guion y respaldo local; despliegue y GPU según registro de cierre |

## Datos

- **CT** (`.mha`, 100 casos): carpeta del equipo en Drive
  (`PENGWIN_CT_train_images part1` = 001–050, `Part2` = 051–100).
- **Máscaras**: `PENGWIN_CT_train_labels.zip` de Zenodo (33 MB; 13 GB descomprimido).
  El notebook la descarga sola.
- Los archivos locales son MetaImage (`.mha`). No inferimos su historial de conversión.
  El espaciado, origen y dirección se leen mediante SimpleITK. Etiquetas: 1–10 sacro, 11–20 coxal izq., 21–30 coxal der.;
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


## Revisión local del 8 de octubre de 2026

Consultar [informe y resultados](doc/REVISION_OCT08.md). Se conserva FundidoraPC, CBAM espacial de 9 × 9 y decodificador de ocho canales internos. Se añade una salida auxiliar de interfaces entre fragmentos. La salida semántica mantiene fondo, sacro y ambos coxales; no se convierten los identificadores de fragmento en clases anatómicas.

La auditoría real cuenta 255 cajas excluidas por el umbral 15 de 62.738 apariciones, no 1.506. El umbral de entrenamiento ahora es cero. La figura antigua `salidas/inventario_ruido_componentes.png` queda obsoleta; usar `salidas/revision_oct08/auditoria_areas.png`.

### Ejecución local en Windows

`config_datos.local.json` apunta a los volúmenes extraídos y al intérprete instalado en este equipo. Los ZIP originales permanecen en `C:/Users/Asus/OneDrive/Documentos/Data_Proyecto_2`. Si se mueve el proyecto a otro equipo, actualizar esa configuración; no contiene los datos.

Desde esta carpeta en PowerShell:

```powershell
$cfgPengwin = Get-Content config_datos.local.json -Raw | ConvertFrom-Json
& $cfgPengwin.python scripts/auditar_anotaciones.py
& $cfgPengwin.python scripts/generar_inventario_ruido.py
& $cfgPengwin.python scripts/preparar_entrenamiento.py
& $cfgPengwin.python scripts/entrenar_pacientes.py --epochs 6
# Si existen checkpoints, reanudar explícitamente:
& $cfgPengwin.python scripts/entrenar_pacientes.py --epochs 12 --resume
& $cfgPengwin.python scripts/inferir_fragmentos_3d.py --case 002 --compare-gt
# Solo después de congelar modelo y protocolo, sin volver a ajustar con test:
& $cfgPengwin.python scripts/comparar_sam_avance3.py --include-test
& $cfgPengwin.python -m unittest discover -s tests -v
```

El entrenamiento usa ocho cortes uniformes por paciente: 560 cortes de 70 pacientes train y 120 de 15 val. Es una ejecución inicial con muestreo; no es entrenamiento sobre todos los cortes de los volúmenes. Cada inferencia 3D sí recorre todos los cortes contiguos del caso elegido. Las métricas 2D, semánticas y de fragmentos 3D se reportan por separado.

Los scripts de evaluación rechazan carpetas de salida existentes para preservar evidencias. Elegir otra `--output` cuando se necesite una ejecución independiente. El cuaderno corregido de semana 9 no se ha reejecutado completamente; su versión histórica y salidas se preservan en `notebooks/historico/` y contienen afirmaciones corregidas por este informe.

SAM es una dependencia opcional (`segment_anything`, implementación oficial de Meta); requiere sus pesos ViT-B. En este equipo ya está disponible mediante las rutas de configuración. No interviene en el entrenamiento del modelo propio.

Los checkpoints del experimento nuevo incluyen la salida auxiliar de bordes. No son intercambiables sin adaptación con checkpoints anteriores de solo detección o de segmentación sin esa salida.


## Adaptación del notebook del profesor — Sesión 2

Entrada recomendada: `notebooks/Semana3_Sesion2_Fragmentos.ipynb` y `doc/ADAPTACION_SESION2.md`.

- `pengwin/models/sesion2.py`: skips aditivas proyectadas a ocho canales y supervisión profunda. Adaptación propia; no U-Net++ completa.
- `pengwin/fragmentos_sesion2.py`: máscaras individuales e interiores por fragmento; reconstrucción mediante bordes e interiores aprendidos.
- `scripts/entrenar_sesion2.py`: experimento separado, desde cero, sin sustituir checkpoints previos.
- `scripts/evaluar_sesion2.py`: comparación de ambas variantes en los mismos pacientes val.
- `scripts/inferir_sesion2_3d.py`: volumen contiguo, instancias y distancias físicas.

Los resultados de la adaptación están en `salidas/sesion2/`; los de la revisión previa están en `salidas/revision_oct08/`. No atribuir unos a otros. El notebook nuevo sirve para inspección y reproducción; sus celdas de inspección se ejecutan con el entorno local y muestran resultados guardados. El entrenamiento se ejecuta mediante los scripts indicados.


## Máscara macro, Sobel y refinamiento local

Consultar [cambios, comparación controlada y límites](doc/MEJORAS_MACRO_SOBEL.md) y el notebook `notebooks/Semana3_MascaraMacro_Refinamiento.ipynb`. Los experimentos nuevos están en `salidas/macro_sobel` y `salidas/macro_control`; no sustituyen resultados históricos. `scripts/inferir_macro_3d.py` carga el modelo y el refinamiento elegidos en `salidas/macro_sobel/seleccion.json`. Usar este script para la variante seleccionada; el notebook de Sesión 2 sigue mostrando el experimento previo.

Se añadió supervisión de la unión de huesos y sus contornos mediante Sobel, refinamiento bilateral de probabilidades y alertas de calidad. LPS ya estaba implementado; ahora se valida explícitamente su geometría. Las alertas no son diagnósticos ni eliminan fragmentos automáticamente. La selección se realiza en validación, sin nueva evaluación test.


## Integración actual: cierre de semana 10

Entrada actual: [informe con resultados y límites](doc/INFORME_CIERRE.md), [modelo seleccionado](doc/MODEL_CARD_FINAL.md) y [guion de 25 minutos](doc/GUION_CIERRE.md). Los resultados actuales están separados en `salidas/cierre/`; los avances anteriores se conservan como históricos.

### Explorar resultados locales

Desde la carpeta del proyecto, en PowerShell:

```powershell
$cfgPengwin = Get-Content config_datos.local.json -Raw | ConvertFrom-Json
& $cfgPengwin.python scripts/servir_dashboard.py --port 8765
```

Abrir `http://127.0.0.1:8765`. El explorador contiene seis casos completos ya calculados, con MIP del CT, cortes con cajas/máscaras/distancias y reconstrucción 3D. No hace inferencia de archivos nuevos al mover el deslizador. Video de respaldo: `salidas/cierre/demo_respaldo.webm`.

### Reproducción y trazabilidad

`requirements-reproducible.txt` fija el entorno CPU usado. Para otro equipo, crear un entorno nuevo, instalar esas dependencias y actualizar `config_datos.local.json` con rutas a imágenes, máscaras, intérprete y pesos de SAM. La configuración es local; no distribuye el dataset. ResNet18 utiliza el archivo oficial `pesos_externos/resnet18-f37072fd.pth`, cuyo hash está en la configuración de ablaciones. SAM es solo una comparación zero-shot con cajas predichas.

Secuencia implementada:

1. `scripts/ajustar_segmentacion.py`: tres variantes, selección en validación.
2. `scripts/ajustar_cajas.py`: ajuste de cajas sujeto a no degradar mAP50.
3. `scripts/estudio_ablacion.py`: comparaciones con/sin CBAM y con/sin transferencia, tres épocas por variante.
4. `scripts/predecir_volumen_cierre.py --cases 002 012 028` y `scripts/evaluar_instancias_cierre.py`: selección de reconstrucción en validación y congelación del protocolo.
5. `scripts/predecir_volumen_cierre.py --cases 004 006 010 --include-test`, `scripts/evaluar_instancias_cierre.py --test` y `scripts/comparar_sam_cierre.py --include-test`: evaluación sin reajuste posterior.
6. `scripts/medir_latencia.py`, `scripts/documentar_cierre.py`, `scripts/construir_dashboard.py --cases 002 012 028 004 006 010` y `scripts/video_respaldo.py`: evidencia, visualizadores y respaldo.

Los scripts experimentales preservan salidas y pueden rechazar carpetas existentes. Para repetir desde cero, usar una copia separada del proyecto y una carpeta nueva de resultados; no borrar la evidencia de esta entrega. El supervisor `scripts/ejecutar_cierre.py` fue creado para la secuencia inicial y **no es un comando para sobrescribir resultados ya terminados**. El entrenamiento inicial que proporciona el checkpoint de partida se documenta en `doc/MEJORAS_MACRO_SOBEL.md` y sus configuraciones.

Pruebas: `& $cfgPengwin.python -m unittest discover -s tests -v`. Las imágenes sintéticas de los tests comprueban funciones; no se incorporaron como datos de entrenamiento.

La evaluación volumétrica cubre tres pacientes val y tres test; no 100 inferencias completas. Los 100 casos sí están inventariados y divididos por paciente. Las medidas se calculan con geometría física y solo se comparan cuando principal y fragmento están correctamente emparejados. GPU queda pendiente si `torch.cuda.is_available()` es falso. El entrenamiento original soporta AMP en CUDA; esta ejecución y los ajustes adicionales se realizaron en CPU. Git y los trámites de colaboración remota se mantienen pendientes por indicación del usuario.
