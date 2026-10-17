# Model card: macro_sobel_entrenamiento

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{
  "seed": 42,
  "train_patients": 70,
  "val_patients": 15,
  "train_slices": 560,
  "val_slices": 120,
  "device": "cpu",
  "seg_weight": 1.2,
  "batch": 8,
  "threads": 4,
  "data_protocol": {
    "cortes_por_paciente": 8,
    "split_sha256": "109860dd5a57289258e4993a53aaf81307e16758dbfbc88c4a07489abd92b745",
    "resolucion": 256,
    "muestreo": "uniforme 0..100%, independiente de anotaciones, incluye negativos",
    "images": "C:\\Users\\Asus\\.codex\\.chatgpt-projects\\g-p-6a7924f5f0c881918894fd3001414af6\\.avance2-datos\\images",
    "labels": "C:\\Users\\Asus\\.codex\\.chatgpt-projects\\g-p-6a7924f5f0c881918894fd3001414af6\\.avance2-datos\\labels"
  },
  "architecture": "sesion2_skips8_interiores_ds_v1",
  "mode": "macro_sobel",
  "initial_sha256": "ec2778e3e880d9ae1e3b29d5ccd12a23f0e42526db42d1d6a597ed471b55f4e3",
  "initial_epoch": 6,
  "lr": 0.0001,
  "macro_weight": 0.5,
  "contour_weight": 0.2,
  "status": "completed",
  "selection": "0.75 Dice macro + 0.25 peor region; mismo criterio para control y candidato",
  "boundary_threshold": 0.5,
  "epochs": 6
}
```

## Pesos

- salidas/macro_sobel/entrenamiento/best.pth — SHA256 ae2e052a8310305e6fb498c0a98960f5fd1f4ebdc354668d4126f5cd3ad268b9

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
      0.5268489009137388,
      0.6300548370170047,
      0.6565202977607667
    ],
    "iou": [
      0.35763398692810455,
      0.45991245054297497,
      0.4886715420162412
    ],
    "macro_dice": 0.6044746785638367,
    "confusion": [
      [
        7566238,
        31449,
        52167,
        45298
      ],
      [
        10004,
        27359,
        2939,
        3256
      ],
      [
        7371,
        715,
        54633,
        244
      ],
      [
        6688,
        778,
        721,
        54460
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
