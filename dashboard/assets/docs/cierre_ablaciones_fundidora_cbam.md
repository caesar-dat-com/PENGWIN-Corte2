# Model card: cierre_ablaciones_fundidora_cbam

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{
  "name": "fundidora_cbam",
  "epochs": 3,
  "seed": 42,
  "train_slices": 560,
  "val_slices": 120,
  "train_patients": 70,
  "val_patients": 15,
  "lr": 0.0003,
  "batch": 8,
  "heads_initial_sha256": "a7cf690a92c867ed10c2da9457b2cdf166fce66f0695c30211a6bcc0c799f413",
  "pretrained_backbone_sha256": null,
  "test_used": false,
  "status": "completed"
}
```

## Pesos

- salidas/cierre/ablaciones/fundidora_cbam/final.pth — SHA256 f1ad362ba593c7464e07e7948f7f78d5a8936951b73575112a76e0a3e95e3009

## Métricas de esta variante

```json
{
  "detection": {
    "mAP50": 0.39919198307048703,
    "mAP50_95": 0.11028882847071216,
    "per_class": {
      "0": {
        "gt": 52,
        "ap50": 0.1479450155482566,
        "ap50_95": 0.024441678083224644
      },
      "1": {
        "gt": 77,
        "ap50": 0.6242936340398624,
        "ap50_95": 0.19256836119937973
      },
      "2": {
        "gt": 77,
        "ap50": 0.425337299623342,
        "ap50_95": 0.11385644612953205
      }
    },
    "mean_gt_iou_at_conf025": 0.4972293027929093,
    "definition": "AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área"
  },
  "classification": {
    "per_class": [
      {
        "class": 0,
        "f1": 0.8360655737704918,
        "auc": 0.9055429864253394,
        "tp": 51,
        "fp": 19,
        "fn": 1,
        "tn": 49
      },
      {
        "class": 1,
        "f1": 0.9554140127388535,
        "auc": 0.9924494110540623,
        "tp": 75,
        "fp": 5,
        "fn": 2,
        "tn": 38
      },
      {
        "class": 2,
        "f1": 0.9487179487179487,
        "auc": 0.9867109634551495,
        "tp": 74,
        "fp": 5,
        "fn": 3,
        "tn": 38
      }
    ],
    "macro_f1": 0.9133991784090979,
    "macro_auc": 0.9615677869781837
  },
  "semantic": {
    "dice": [
      0.12201953826204529,
      0.1600496792739271,
      0.1442210546337535
    ],
    "iou": [
      0.06497380603689762,
      0.08698586992868863,
      0.07771456562424293
    ],
    "macro_dice": 0.14209675738990862,
    "confusion": [
      [
        6089755,
        359565,
        568191,
        677641
      ],
      [
        228,
        26442,
        9006,
        7882
      ],
      [
        659,
        3284,
        55928,
        3092
      ],
      [
        911,
        557,
        2795,
        58384
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
