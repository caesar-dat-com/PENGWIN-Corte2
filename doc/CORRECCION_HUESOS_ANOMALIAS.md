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

### Estado y siguiente paso
El entrenamiento completo (todos los cortes con hueso, 30 épocas) se ejecuta en Google Colab con GPU. Las métricas finales por hueso, la comparación modelo crudo frente a corregido por el control de anomalías y los pares de distancia se reportan en el notebook al terminar la ejecución.

### Enlaces
- Repositorio (rama de corrección): https://github.com/caesar-dat-com/PENGWIN-Corte2/tree/fix-huesos-colab
- Notebook Colab: https://colab.research.google.com/github/caesar-dat-com/PENGWIN-Corte2/blob/fix-huesos-colab/notebooks/Colab_Correccion_Huesos.ipynb
- Módulo de anomalías: `pengwin/anomalias.py` · Evaluación: `scripts/colab_evaluar_volumen.py`
