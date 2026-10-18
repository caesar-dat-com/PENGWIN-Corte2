---
title: "PENGWIN — Corrección de la selección de huesos y control de anomalías"
subtitle: "Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro"
author: "Yesenia Díaz · Juan Pablo Maya · César Reyes"
date: "10 de octubre de 2026"
---


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

Ninguna regla usa el GT. La referencia de volúmenes se ajusta solo con train.

### Validación realizada (CPU, antes de entrenar en GPU)
- Inversión artificial de coxales izquierdo/derecho sobre el GT de los casos 001–003: el control de lateralidad recupera **99,3 %–99,97 %** de los vóxeles de coxal.
- El control aplicado al GT correcto deja intacto **≥ 99,4 %** del hueso anotado (no destruye anotación válida).
- Umbral de «hueso en aire» fijado en −500 HU: en el caso 001 hay hueso anotado real hasta −400 HU (médula grasa).
- Augmentación espejo verificada: tras el flip el coxal izquierdo permanece del lado izquierdo en LPS, con cajas y clases coherentes.
- Entrenamiento de 2 épocas y evaluación volumétrica completa corren de punta a punta.

### Resultados del entrenamiento completo en GPU (30 épocas, Tesla T4)
El entrenamiento se ejecutó sobre `9,889` cortes de entrenamiento (70 pacientes) y `2,186` cortes de validación (15 pacientes) bajo `--modo hueso --paso 2 --vacios 0.15`, alcanzando su mejor punto de selección en la **época 21** (`selection_score = 0.8968`, Macro-Dice `94.15 %`, `mAP@0.50 = 85.21 %`):

| Métrica (Validación, 15 pacientes) | Avance oct-08 (CPU, 8 cortes) | Época 3 (GPU parcial) | **Final 30 épocas (Mejor: Época 21)** | Mejora total |
|---|---:|---:|---:|---:|
| **Dice Sacro (SA)** | 38,82 % | 55,27 % | **91,91 %** | **+53,09 pts** |
| **Dice Coxal Izquierdo (LI)** | 0,00 % | 82,28 % | **95,09 %** | **+95,09 pts** |
| **Dice Coxal Derecho (RI)** | 42,59 % | 83,39 % | **95,45 %** | **+52,86 pts** |
| **Macro-Dice Semántico** | 27,14 % | 73,65 % | **94,15 %** | **+67,01 pts** |
| **mAP@0.50 (Detección)** | 44,82 % | 51,51 % | **85,21 %** | **+40,39 pts** |

#### Matriz de confusión en validación (filas = GT, columnas = predicción: Fondo / SA / LI / RI)
```text
[[139092540     78075     67998     77534]
 [    74248    918421      2487      2608]
 [    70538      1839   1399811      1527]
 [    51284      2463        35   1420288]]
```
- La confusión cruzada entre coxal izquierdo (`LI`) y coxal derecho (`RI`) quedó prácticamente eliminada (`1,527 / 1,473,715 = 0,10 %` en LI y `35 / 1,474,070 = 0,002 %` en RI).

#### Evaluación sobre volúmenes 3D completos de validación (`002`, `012`, `028`)
1. **Política conservadora de baja sobresegmentación** (`{"method": "interfaces", "threshold": 0.7, "seed_mm3": 500.0, "min_volume_mm3": 2500.0, "fusionar": true}`):

| Métrica 3D | Caso `002` (GT=6) | Caso `012` (GT=5) | Caso `028` (GT=7) |
|---|---:|---:|---:|
| **Dice 3D Sacro (crudo → corregido)** | 94,45 % → **94,56 %** | 68,77 % → **68,21 %** | 92,16 % → **92,39 %** |
| **Dice 3D Coxal Izq (crudo → corregido)** | 97,04 % → **96,97 %** | 81,59 % → **82,20 %** | 96,34 % → **96,36 %** |
| **Dice 3D Coxal Der (crudo → corregido)** | 96,63 % → **96,77 %** | 89,35 % → **89,67 %** | 95,58 % → **95,68 %** |
| **Instancias predichas (crudo → corregido)** | 3 → **3** (`extra=0`) | 9 → **8** (`extra=4`) | 3 → **3** (`extra=0`) |
| **Dice simétrico por instancia (`dice_sim`)** | 46,58 % → **46,60 %** | 26,51 % → **29,62 %** | 38,14 % → **38,19 %** |
| **Vóxeles corregidos (lateralidad / islas z / suavizado z)** | 749 / 253 / 1874 | 865 / 1488 / 2778 | 92 / 380 / 4487 |
| **Anomalías post-corrección (lateralidad / exceso frag.)** | **0 / 0** | **0 / 0** | **0 / 0** |

2. **Política sensible a fracturas y pares de distancia (`VAL_FRAG`)** (`{"method": "interiores", "threshold": 0.35, "seed_mm3": 80.0, "min_volume_mm3": 250.0, "fusionar": true}`, `--min-iou 0.10`):
   - **Caso `002` (`GT=6`):** `corr_missed = 0` (6/6 fragmentos detectados), `corr_pares = 2`, **`corr_MAE = 0.00 mm`**, `corr_dice_sim = 28.96 %`.
   - **Caso `012` (`GT=5`):** `corr_missed = 1` (4/5 fragmentos detectados), `corr_pares = 1`, **`corr_MAE = 0.00 mm`**, `corr_dice_sim = 8.18 %` (mejora frente a `6.74 %` crudo).
   - **Caso `028` (`GT=7`):** `corr_missed = 0` (7/7 fragmentos detectados), `corr_pares = 3`, **`corr_MAE = 0.00 mm`**, `corr_dice_sim = 20.20 %`.

#### Evaluación final congelada sobre volúmenes 3D de prueba (`TEST_FINAL`: `004`, `006`, `010`)
Evaluación única sobre los tres pacientes reservados de `test` con el modelo congelado (`best.pth`, época 21):

| Métrica 3D (`TEST_FINAL`) | Caso `004` (GT=6) | Caso `006` (GT=7) | Caso `010` (GT=5) | Promedio Test 3D |
|---|---:|---:|---:|---:|
| **Estado control de anomalías** | `revisar` (2 anom.) | **`ok` (1 anom.)** | **`ok` (1 anom.)** | 2/3 casos `ok` |
| **Dice 3D Sacro (crudo → corregido)** | 92,17 % → **92,30 %** | 89,92 % → **89,96 %** | 90,16 % → **90,28 %** | **90,85 %** |
| **Dice 3D Coxal Izq (crudo → corregido)** | 96,56 % → **94,76 %** | 96,81 % → **96,81 %** | 91,92 % → **91,95 %** | **94,51 %** (95,10 % crudo) |
| **Dice 3D Coxal Der (crudo → corregido)** | 96,73 % → **94,83 %** | 96,73 % → **96,75 %** | 94,47 % → **94,46 %** | **95,35 %** (95,98 % crudo) |
| **Fragmentos omitidos (`corr_missed`)** | 1 (5/6 detectados) | 1 (6/7 detectados) | **0 (5/5 detectados)** | **0,67** (89 % recall frag.) |
| **Dice simétrico (`crudo → corr`)** | 37,01 % → 30,73 % | **45,80 % → 42,06 %** | 17,16 % → **17,16 %** | **33,33 % crudo / 29,98 % corr** |
| **Pares válidos de distancia (`pares`)** | **2** | **3** | 0 | **5 pares totales** |
| **Error físico de separación (`MAE` mm)** | **0,00 mm** → **0,00 mm** | 0,41 mm → **0,24 mm** (−41 %) | No evaluable | **0,15 mm (corregido)** |

### Enlaces
- Repositorio: https://github.com/caesar-dat-com/PENGWIN-Corte2
- Notebook Colab: [`notebooks/Colab_Correccion_Huesos.ipynb`](../notebooks/Colab_Correccion_Huesos.ipynb)
- Módulo de anomalías: `pengwin/anomalias.py` · Evaluación: `scripts/colab_evaluar_volumen.py`

