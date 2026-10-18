---
title: "PENGWIN — Corrección de la selección de huesos y control de anomalías"
subtitle: "Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro"
author: "Yesenia Díaz · Juan Pablo Maya · César Reyes"
date: "17 de octubre de 2026"
---

**Rama:** `fix-huesos-colab` · Entorno: *Google Colab GPU (Tesla T4, 15,360 MiB)*.

### Identificación del modelo entrenado en GPU
- **Checkpoint:** `salidas/fix_huesos_v1/best.pth` (mejor época = `21` de `30`).
- **SHA256:** `3d68d4668e6b550a171cf112d0bd22dfddebcdbad013db55044148b562ed72c7` (`salidas/fix_huesos_v1/best_pth_sha256.json`).
- **Artefactos verificables en el repositorio:** `salidas/fix_huesos_v1/config.json`, `salidas/fix_huesos_v1/historial.json` (30 épocas completas), `salidas/fix_huesos_v1/metricas_val.json`, `salidas/fix_huesos_v1/volumenes/tabla.json`, `salidas/fix_huesos_v1/val_frag/tabla.json` y `salidas/fix_huesos_v1/test_final/tabla.json`.

### Por qué fallaba la selección de huesos (diagnóstico del repo)
| # | Falla | Evidencia | Corrección en esta rama |
|---|---|---|---|
| 1 | `main` no corre: faltaba fusionar `mejoras-yesenia` (`rutas_dataset`, `normalizar_lps`, `balancear_obj`) | `ImportError` en `preparar_entrenamiento.py` | merge de `93cf1ad` |
| 2 | Muy pocos datos: 8 cortes por paciente (560 train), 6 épocas en CPU | sacro en solo 52/120 cortes de val | `--modo hueso`: todos los cortes con hueso + 15 % vacíos (`9,889` train / `2,186` val), GPU, 30 épocas |
| 3 | Coxal izquierdo con Dice 0 % = el decoder no separaba los dos coxales simétricos | matriz de confusión: columna «pred LI» en 0 | flip izquierda-derecha **con intercambio de etiquetas**, más datos y control de lateralidad |
| 4 | La máscara se recortaba con la caja predicha (IoU de caja ≈ 0,5) | `gate_semantic` estricto | `--gate-margin 0.1` |
| 5 | Tejido sobrante: el fondo pesaba 0,2 en la CE | ~81 mil px de fondo predichos como hueso | `--bg-weight 0.5` |
| 6 | Sobresegmentación: 13–26 instancias frente a 5–7 GT; `reconstruir` en modo `interiores` no pasaba `threshold` ni `fusionar` | `instances.py` y `instancias_estables.py` | suavizado en z, sin islas, transmisión explícita de `threshold` y `fusionar` en `instancias_estables.py` |
| 7 | `raise` cuando dos regiones caen en la misma celda de 16 px y desbordamiento FP16 en focal loss | `loss.py` y `entrenar_sesion2.py` | se conserva la caja mayor, evaluación de pérdida en `float32` con `.clamp(min=1e-6)` bajo `torch.amp.autocast` |

### Control de anomalías (`pengwin/anomalias.py`)
- **Entrada CT:** cizalla u oblicuidad, cortes > 5 mm, pocos cortes, rango HU imposible, metal (0 anomalías en los 100 CT de PENGWIN).
- **Salida del modelo:**
  - lateralidad invertida respecto a la línea media; se corrige por componente;
  - hueso sobre aire (HU < −500);
  - islas de menos de 3 cortes;
  - región ausente;
  - volumen fuera del rango de train (`Sacro: 118–296 cm³`, `Coxal izq: 233–574 cm³`, `Coxal der: 229–588 cm³`);
  - hueso hipodenso;
  - más de 10 fragmentos por hueso.

Ninguna regla usa el GT. La referencia de volúmenes se ajusta solo con train.

### Resultados en Validación 2D por corte (`15` pacientes, `2,186` cortes, época 21)

| Métrica | Avance oct-08 (CPU, 8 cortes) | `fix_huesos_v1` (GPU T4, 30 épocas) | Mejora |
|---|---:|---:|---:|
| **Dice Sacro (`SA`)** | 38.82 % | **91.91 %** | +53.09 pts |
| **Dice Coxal Izquierdo (`LI`)** | 0.00 % | **95.09 %** | +95.09 pts |
| **Dice Coxal Derecho (`RI`)** | 42.59 % | **95.45 %** | +52.86 pts |
| **Macro-Dice Semántico** | 27.14 % | **94.15 %** | +67.01 pts |
| **Detección `mAP@0.50`** | 44.82 % | **85.21 %** | +40.39 pts |

Matriz de confusión en validación (filas GT, columnas pred: `fondo / SA / LI / RI`):
```text
[[139092540     78075     67998     77534]
 [    74248    918421      2487      2608]
 [    70538      1839   1399811      1527]
 [    51284      2463        35   1420288]]
```

### Resultados en Volúmenes 3D Completos (Validación y Prueba Final)

#### 1. Validación 3D — Política conservadora (`interfaces`, `threshold=0.7`, `seed_mm3=500`, `min_volume_mm3=2500`, `fusionar=True`, `min_iou=0.35`)
| Caso (`val`) | GT | Dice SA (corr) | Dice LI (corr) | Dice RI (corr) | Pred (crudo→corr) | Extra (crudo→corr) | Omitidos | Dice simétrico (corr) | Pares | MAE (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **002** | 6 | 94.56 % | 96.97 % | 96.77 % | 3 → **3** | 0 → **0** | 3 | **46.60 %** | 0 | No evaluable |
| **012** | 5 | 68.21 % | 82.20 % | 89.67 % | 9 → **8** | 5 → **4** | 1 | **29.62 %** | 0 | No evaluable |
| **028** | 7 | 92.39 % | 96.36 % | 95.68 % | 3 → **3** | 0 → **0** | 4 | **38.19 %** | 0 | No evaluable |

#### 2. Validación 3D — Política sensible a fracturas (`interiores`, `threshold=0.35`, `seed_mm3=80`, `min_volume_mm3=250`, `fusionar=True`, `min_iou=0.10`)
| Caso (`val`) | GT | Pred (crudo→corr) | Extra (crudo→corr) | Omitidos | Dice simétrico (crudo→corr) | Pares válidos | MAE (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **002** | 6 | 11 → **11** | 6 → **6** | **1** | 33.86 % → **33.88 %** | **2** | **5.14 mm** |
| **012** | 5 | 30 → **24** | 26 → **20** | **1** | 8.06 % → **10.00 %** | **1** | **5.00 mm** |
| **028** | 7 | 14 → **15** | 8 → **9** | **1** | 32.14 % → **30.07 %** | **3** | **12.59 mm** |

#### 3. Prueba Volumétrica 3D Final (`test`: casos `004`, `006`, `010`, política `interiores` congelada)
| Caso (`test`) | Estado QC | GT | Dice SA (corr) | Dice LI (corr) | Dice RI (corr) | Pred (crudo→corr) | Extra (crudo→corr) | Omitidos | Dice sim (crudo→corr) | Pares | MAE (crudo → corr) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **004** | `revisar` | 6 | **92.30 %** | **94.76 %** | **94.83 %** | 14 → **14** | 8 → **8** | **0** | 33.31 % → **31.99 %** | **3** | **0.00 mm → 0.00 mm** |
| **006** | `ok` | 7 | **89.96 %** | **96.81 %** | **96.75 %** | 13 → **14** | 8 → **9** | **2** | 25.15 % → **23.86 %** | **2** | **0.62 mm → 0.36 mm** |
| **010** | `ok` | 5 | **90.28 %** | **91.95 %** | **94.46 %** | 16 → **16** | 11 → **11** | **0** | 17.70 % → **17.70 %** | **2** | **0.00 mm → 0.00 mm** |

### Figuras del entrenamiento y validación en GPU
- Curvas de entrenamiento (30 épocas): `salidas/cierre/curvas_gpu_30epocas.png`
- Comparación axial en validación (Caso `002`, GT vs crudo vs corregido): `salidas/cierre/visual_val_002_gpu.png`
