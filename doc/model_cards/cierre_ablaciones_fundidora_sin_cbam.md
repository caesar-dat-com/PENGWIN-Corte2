# Model card: cierre_ablaciones_fundidora_sin_cbam

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{
  "name": "fundidora_sin_cbam",
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

- salidas/cierre/ablaciones/fundidora_sin_cbam/final.pth — SHA256 031ffe384ba19c43acf26e54d7defeefd05ec22b4398ee31b66c098f83887247

## Métricas de esta variante

```json
{
  "detection": {
    "mAP50": 0.3173157766509505,
    "mAP50_95": 0.09846987684506806,
    "per_class": {
      "0": {
        "gt": 52,
        "ap50": 0.07317164934169985,
        "ap50_95": 0.019852699432508374
      },
      "1": {
        "gt": 77,
        "ap50": 0.6317215422198236,
        "ap50_95": 0.21341202498178843
      },
      "2": {
        "gt": 77,
        "ap50": 0.24705413839132823,
        "ap50_95": 0.06214490612090735
      }
    },
    "mean_gt_iou_at_conf025": 0.4727408933188209,
    "definition": "AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área"
  },
  "classification": {
    "per_class": [
      {
        "class": 0,
        "f1": 0.7938931297709924,
        "auc": 0.8959276018099548,
        "tp": 52,
        "fp": 27,
        "fn": 0,
        "tn": 41
      },
      {
        "class": 1,
        "f1": 0.8901734104046243,
        "auc": 0.9830866807610994,
        "tp": 77,
        "fp": 19,
        "fn": 0,
        "tn": 24
      },
      {
        "class": 2,
        "f1": 0.8636363636363636,
        "auc": 0.9646632437330112,
        "tp": 76,
        "fp": 23,
        "fn": 1,
        "tn": 20
      }
    ],
    "macro_f1": 0.8492343012706601,
    "macro_auc": 0.9478925087680219
  },
  "semantic": {
    "dice": [
      0.10720003904189683,
      0.17406406567242663,
      0.15017360663265042
    ],
    "iou": [
      0.056635693815015715,
      0.0953286818009462,
      0.0811825407892902
    ],
    "macro_dice": 0.1438125704489913,
    "confusion": [
      [
        6036501,
        458464,
        552350,
        647837
      ],
      [
        943,
        28556,
        8507,
        5552
      ],
      [
        1556,
        760,
        59583,
        1064
      ],
      [
        1801,
        1423,
        1207,
        58216
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
