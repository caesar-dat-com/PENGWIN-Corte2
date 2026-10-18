# Model card: cierre_ablaciones_resnet_sin_transfer

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{
  "name": "resnet_sin_transfer",
  "epochs": 3,
  "seed": 42,
  "train_slices": 560,
  "val_slices": 120,
  "train_patients": 70,
  "val_patients": 15,
  "lr": 0.0003,
  "batch": 8,
  "heads_initial_sha256": "a55e9adb3cffa0466d9af779ff0a5f65cdf4ab8c4a31a2f09500690c9bd41dcc",
  "pretrained_backbone_sha256": null,
  "test_used": false,
  "status": "completed"
}
```

## Pesos

- salidas/cierre/ablaciones/resnet_sin_transfer/final.pth — SHA256 2f960991f8e09fb4db81880d12b16ffb4ecdd8b3d5308c28145f45bb57d55c8c

## Métricas de esta variante

```json
{
  "detection": {
    "mAP50": 0.4573881061946478,
    "mAP50_95": 0.119248961465808,
    "per_class": {
      "0": {
        "gt": 52,
        "ap50": 0.22198696289505504,
        "ap50_95": 0.047521853682220325
      },
      "1": {
        "gt": 77,
        "ap50": 0.5392174689580341,
        "ap50_95": 0.13595474178381267
      },
      "2": {
        "gt": 77,
        "ap50": 0.6109598867308541,
        "ap50_95": 0.17427028893139102
      }
    },
    "mean_gt_iou_at_conf025": 0.5008592608682885,
    "definition": "AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área"
  },
  "classification": {
    "per_class": [
      {
        "class": 0,
        "f1": 0.864406779661017,
        "auc": 0.9827488687782805,
        "tp": 51,
        "fp": 15,
        "fn": 1,
        "tn": 53
      },
      {
        "class": 1,
        "f1": 0.9487179487179487,
        "auc": 0.9897311990335246,
        "tp": 74,
        "fp": 5,
        "fn": 3,
        "tn": 38
      },
      {
        "class": 2,
        "f1": 0.9615384615384616,
        "auc": 0.9930534581697372,
        "tp": 75,
        "fp": 4,
        "fn": 2,
        "tn": 39
      }
    ],
    "macro_f1": 0.9248877299724758,
    "macro_auc": 0.9885111753271808
  },
  "semantic": {
    "dice": [
      0.2368156008699587,
      0.6137594454426137,
      0.6217197328677917
    ],
    "iou": [
      0.1343113068529894,
      0.4427510387175056,
      0.4510836784752101
    ],
    "macro_dice": 0.49076492639345465,
    "confusion": [
      [
        7603902,
        14913,
        42823,
        33514
      ],
      [
        31396,
        8112,
        1787,
        2263
      ],
      [
        13048,
        697,
        48166,
        1052
      ],
      [
        15331,
        1229,
        1215,
        44872
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
