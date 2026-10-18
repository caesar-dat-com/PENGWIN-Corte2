# Model card del modelo final seleccionado

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Arquitectura: FundidoraPC + CBAM, tres cabezas propias, decoder con skips y ocho canales. Backbone inicialmente desde cero, sin pesos de SAM ni de detectores externos. Entrenamiento y selección en 70/15 pacientes; resultados de prueba limitados a los cortes y volúmenes descritos en el informe.

Checkpoint: `salidas\cierre\ajuste\contorno\best.pth`. SHA256: `2a5e9f1daeec2328da2b78f263aa8edc1d10aa62593bfe5f18f56f494fc7297d`.

Clasificación test (línea base CPU): F1 91.23%, AUC 97.98%. Con entrenamiento completo en GPU Tesla T4 (30 épocas, `--modo hueso --paso 2 --vacios 0.15`, `9,889` cortes train / `2,186` val, mejor época = 21 en `fix_huesos_v1/best.pth`):
- **Validación (15 pacientes, 2,186 cortes):** `mAP@0.50 = 85.21%`, Macro-Dice semántico `94.15%` (Sacro `91.91%`, Coxal Izq `95.09%`, Coxal Der `95.45%`).
- **Prueba volumétrica 3D (`test`: casos `004`, `006`, `010`):** Macro-Dice 3D `93.57%` corregido / `94.02%` crudo (Sacro `90.85%`, Coxal Izq `94.51%`, Coxal Der `95.35%`), `dice_symmetric` de hasta `45.80%` (caso `006`), `5` pares válidos de distancia inter-fragmento y error físico `MAE = 0.00 mm` (caso `004`) y `0.24 mm` corregido vs `0.41 mm` crudo (caso `006`).

Políticas de instancias evaluadas:
1. Conservadora (`interfaces`, `threshold=0.7`, `seed_mm3=500`, `min_volume_mm3=2500`, `fusionar=True`): elimina la sobresegmentación (`extra=0` en casos `002` y `028`) y alcanza `dice_symmetric = 46.60%`.
2. Sensible a fracturas (`interiores`, `threshold=0.35`, `seed_mm3=80`, `min_volume_mm3=250`, `fusionar=True`): recupera el `89%–100%` de los fragmentos GT y habilita la medición física de separación (`distance_transform_edt` con `spacing` físico en mm).

No usar para decisiones clínicas. Uso exclusivamente académico. Las alertas de `pengwin/anomalias.py` son control de calidad geométrico/anatómico y no diagnóstico médico. La reproducción usa la configuración y los hashes guardados; no modificar el protocolo tras observar test.
