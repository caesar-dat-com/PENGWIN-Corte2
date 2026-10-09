# Model card: cierre_ablaciones_resnet_transfer

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{
  "name": "resnet_transfer",
  "epochs": 3,
  "seed": 42,
  "train_slices": 560,
  "val_slices": 120,
  "train_patients": 70,
  "val_patients": 15,
  "lr": 0.0003,
  "batch": 8,
  "heads_initial_sha256": "a55e9adb3cffa0466d9af779ff0a5f65cdf4ab8c4a31a2f09500690c9bd41dcc",
  "pretrained_backbone_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
  "test_used": false,
  "status": "completed"
}
```

## Pesos

- salidas/cierre/ablaciones/resnet_transfer/final.pth — SHA256 f385cc677c68112fd6927e6a026aa3beb6d700e724982266748df88844cb9933

## Métricas de esta variante

```json
{
  "detection": {
    "mAP50": 0.28187856841373565,
    "mAP50_95": 0.06196968452134606,
    "per_class": {
      "0": {
        "gt": 52,
        "ap50": 0.1742228959953834,
        "ap50_95": 0.041753644996809676
      },
      "1": {
        "gt": 77,
        "ap50": 0.2853384370958246,
        "ap50_95": 0.06819081986803044
      },
      "2": {
        "gt": 77,
        "ap50": 0.38607437214999896,
        "ap50_95": 0.07596458869919806
      }
    },
    "mean_gt_iou_at_conf025": 0.4666752093153323,
    "definition": "AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área"
  },
  "classification": {
    "per_class": [
      {
        "class": 0,
        "f1": 0.9423076923076923,
        "auc": 0.9932126696832579,
        "tp": 49,
        "fp": 3,
        "fn": 3,
        "tn": 65
      },
      {
        "class": 1,
        "f1": 0.9681528662420382,
        "auc": 0.9891271519178496,
        "tp": 76,
        "fp": 4,
        "fn": 1,
        "tn": 39
      },
      {
        "class": 2,
        "f1": 0.961038961038961,
        "auc": 0.9921473874962247,
        "tp": 74,
        "fp": 3,
        "fn": 3,
        "tn": 40
      }
    ],
    "macro_f1": 0.9571665065295639,
    "macro_auc": 0.9914957363657774
  },
  "semantic": {
    "dice": [
      0.23263860947935946,
      0.3279657218040298,
      0.4541264178925007
    ],
    "iou": [
      0.13163046942585258,
      0.19614772620444487,
      0.2937668533499274
    ],
    "macro_dice": 0.33824358305862995,
    "confusion": [
      [
        7349149,
        42944,
        215736,
        87323
      ],
      [
        23114,
        11564,
        6219,
        2661
      ],
      [
        4875,
        112,
        56335,
        1641
      ],
      [
        13800,
        1238,
        2289,
        45320
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
