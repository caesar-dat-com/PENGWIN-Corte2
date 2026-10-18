# Model card del modelo final seleccionado

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Identificación inequívoca del modelo

- **Arquitectura:** `PelvisSesion2` (`sesion2_skips8_interiores_ds_v1`): `FundidoraPC + CBAM9`, tres cabezas propias (clasificación multietiqueta, detección por grid $16\times 16$ con NMS propio y segmentación con supervisión profunda), decoder con skips aditivas de 8 canales y cabezas auxiliares de bordes e interiores. Backbone entrenado desde cero, sin pesos de SAM ni de detectores externos.
- **Checkpoint final en GPU (30 épocas):** `salidas/fix_huesos_v1/best.pth` (en Google Drive: `/content/drive/MyDrive/PENGWIN_runs/fix_huesos_v1/best.pth`, época seleccionada = `21` de `30`).
- **SHA256 (`fix_huesos_v1/best.pth`):** `3d68d4668e6b550a171cf112d0bd22dfddebcdbad013db55044148b562ed72c7` (registrado en `salidas/fix_huesos_v1/best_pth_sha256.json`, con configuración completa en `salidas/fix_huesos_v1/config.json` e historial de 30 épocas en `salidas/fix_huesos_v1/historial.json`).
- **Cómo reproducir / obtener el checkpoint GPU:** ejecutar `notebooks/Colab_Correccion_Huesos.ipynb` (o `python scripts/preparar_entrenamiento.py --modo hueso --paso 2 --vacios 0.15 --salida /content/datos_hueso` seguido de `python scripts/entrenar_sesion2.py --data /content/datos_hueso --output salidas/fix_huesos_v1 --epochs 30 --batch 32 --lr 3e-4 --cosine --bg-weight 0.5 --gate-margin 0.1 --flip`).
- **Referencia histórica en CPU (6 épocas, 8 cortes/paciente):** `salidas/cierre/ajuste/contorno/best.pth` (`SHA256: 2a5e9f1daeec2328da2b78f263aa8edc1d10aa62593bfe5f18f56f494fc7297d`), conservada únicamente como línea base comparativa previa al entrenamiento en GPU.

## Desempeño verificado del modelo GPU (`3d68d466...`, época 21)

- **Validación 2D por corte (`15` pacientes, `2,186` cortes, `salidas/fix_huesos_v1/metricas_val.json`):**
  - Detección: `mAP@0.50 = 85.21%`, `mAP@0.50:0.95 = 65.43%`, `IoU medio GT = 80.59%`.
  - Clasificación por corte: `Macro-F1 = 96.46%`, `Macro-AUC = 99.64%`.
  - Segmentación anatómica: `Macro-Dice = 94.15%` (Sacro `91.91%`, Coxal Izq `95.09%`, Coxal Der `95.45%`).
- **Validación volumétrica 3D (`002`, `012`, `028`):**
  1. Política conservadora (`interfaces`, `threshold=0.7`, `seed_mm3=500`, `min_volume_mm3=2500`, `fusionar=True`, `min_iou=0.35`, `salidas/fix_huesos_v1/volumenes/tabla.json`):
     - Caso `002` (`GT=6`): `pred=3`, `extra=0`, `missed=3`, `dice_sim=46.60%`, `pares=0`, `MAE=No evaluable`.
     - Caso `012` (`GT=5`): `pred=8`, `extra=4`, `missed=1`, `dice_sim=29.62%`, `pares=0`, `MAE=No evaluable`.
     - Caso `028` (`GT=7`): `pred=3`, `extra=0`, `missed=4`, `dice_sim=38.19%`, `pares=0`, `MAE=No evaluable`.
  2. Política sensible a fracturas (`interiores`, `threshold=0.35`, `seed_mm3=80`, `min_volume_mm3=250`, `fusionar=True`, `min_iou=0.10`, `salidas/fix_huesos_v1/val_frag/tabla.json`):
     - Caso `002` (`GT=6`): `pred=11`, `extra=6`, `missed=1`, `dice_sim=33.88%`, `pares=2`, `MAE=5.14 mm`.
     - Caso `012` (`GT=5`): `pred=24`, `extra=20`, `missed=1`, `dice_sim=10.00%`, `pares=1`, `MAE=5.00 mm`.
     - Caso `028` (`GT=7`): `pred=15`, `extra=9`, `missed=1`, `dice_sim=30.07%`, `pares=3`, `MAE=12.59 mm`.
- **Prueba volumétrica 3D (`test`: casos `004`, `006`, `010`, política `interiores` congelada, `salidas/fix_huesos_v1/test_final/tabla.json`):**
  - Caso `004` (`GT=6`): Dice 3D corregido SA `92.30%`, LI `94.76%`, RI `94.83%`, `pred=14`, `extra=8`, `missed=0`, `dice_sim=31.99%` (`33.31%` crudo), `pares=3`, `MAE=0.00 mm`.
  - Caso `006` (`GT=7`): Dice 3D corregido SA `89.96%`, LI `96.81%`, RI `96.75%`, `pred=14`, `extra=9`, `missed=2`, `dice_sim=23.86%` (`25.15%` crudo), `pares=2`, `MAE=0.36 mm` corregido vs `0.62 mm` crudo.
  - Caso `010` (`GT=5`): Dice 3D corregido SA `90.28%`, LI `91.95%`, RI `94.46%`, `pred=16`, `extra=11`, `missed=0`, `dice_sim=17.70%`, `pares=2`, `MAE=0.00 mm`.

## Limitaciones y uso responsable

No usar para decisiones clínicas. Uso exclusivamente académico. Las alertas de `pengwin/anomalias.py` son control de calidad geométrico/anatómico y no diagnóstico médico. La separación de fragmentos por `interiores` incrementa la sensibilidad a fracturas y permite evaluar distancias EDT con `spacing` físico, pero mantiene sobresegmentación (`extra` de `6` a `20` instancias), por lo que las distancias deben interpretarse siempre junto con el conteo de instancias sobrantes y omitidas.
