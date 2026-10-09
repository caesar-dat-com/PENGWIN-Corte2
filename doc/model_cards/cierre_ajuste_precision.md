# Model card: cierre_ajuste_precision

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{
  "status": "completed",
  "architecture": "sesion2_skips8_interiores_ds_v1",
  "initial_sha256": "ae2e052a8310305e6fb498c0a98960f5fd1f4ebdc354668d4126f5cd3ad268b9",
  "decoder_only": true,
  "seed": 42,
  "epochs": 4,
  "lr": 0.0002,
  "batch": 8,
  "ce_background": 1.0,
  "macro_weight": 0.25,
  "contour_weight": 0.5,
  "selection": "0.7 Dice semantico + 0.3 F1 contorno tolerancia2px",
  "test_used": false
}
```

## Pesos

- salidas/cierre/ajuste/precision/best.pth — SHA256 54c3f1776664663696b1ee4a777802e1d906034cc8f13d8f563deb63050b1e52

## Métricas de esta variante

```json
{
  "detection": {
    "mAP50": 0.638178058187376,
    "mAP50_95": 0.1826584366288429,
    "per_class": {
      "0": {
        "gt": 52,
        "ap50": 0.3790549493904876,
        "ap50_95": 0.06339771214949197
      },
      "1": {
        "gt": 77,
        "ap50": 0.7249449800873928,
        "ap50_95": 0.22177126801922017
      },
      "2": {
        "gt": 77,
        "ap50": 0.8105342450842478,
        "ap50_95": 0.2628063297178166
      }
    },
    "mean_gt_iou_at_conf025": 0.5875883654645921,
    "definition": "AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área"
  },
  "classification": {
    "per_class": [
      {
        "class": 0,
        "f1": 0.9306930693069307,
        "auc": 0.9850113122171946,
        "tp": 47,
        "fp": 2,
        "fn": 5,
        "tn": 66
      },
      {
        "class": 1,
        "f1": 0.9673202614379085,
        "auc": 0.9924494110540623,
        "tp": 74,
        "fp": 2,
        "fn": 3,
        "tn": 41
      },
      {
        "class": 2,
        "f1": 0.9473684210526315,
        "auc": 0.987012987012987,
        "tp": 72,
        "fp": 3,
        "fn": 5,
        "tn": 40
      }
    ],
    "macro_f1": 0.9484605839324902,
    "macro_auc": 0.9881579034280813
  },
  "semantic": {
    "dice": [
      0.5544085671902623,
      0.6955789043649223,
      0.6983840273521917
    ],
    "iou": [
      0.3835167770140148,
      0.533247205747059,
      0.5365515190563513
    ],
    "macro_dice": 0.6494571663024588,
    "confusion": [
      [
        7599426,
        26238,
        34916,
        34572
      ],
      [
        9672,
        27283,
        3125,
        3478
      ],
      [
        7650,
        641,
        54484,
        188
      ],
      [
        6645,
        702,
        1170,
        54130
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
