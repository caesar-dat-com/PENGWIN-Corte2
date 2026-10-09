# Model card: macro_control_entrenamiento

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
  "mode": "control",
  "initial_sha256": "ec2778e3e880d9ae1e3b29d5ccd12a23f0e42526db42d1d6a597ed471b55f4e3",
  "initial_epoch": 6,
  "lr": 0.0001,
  "macro_weight": 0.0,
  "contour_weight": 0.0,
  "status": "completed",
  "selection": "0.75 Dice macro + 0.25 peor region; mismo criterio para control y candidato",
  "boundary_threshold": 0.5,
  "epochs": 6
}
```

## Pesos

- salidas/macro_control/entrenamiento/best.pth — SHA256 02eeec07a967bff7b16579c6f21a9d1fe51502cca707e3ec92c4c7436d4f701b

## Métricas de esta variante

```json
{
  "detection": {
    "mAP50": 0.5878315805220481,
    "mAP50_95": 0.18165252018404304,
    "per_class": {
      "0": {
        "gt": 52,
        "ap50": 0.3670369027118157,
        "ap50_95": 0.08243533098116984
      },
      "1": {
        "gt": 77,
        "ap50": 0.6207313564972785,
        "ap50_95": 0.20610139061242103
      },
      "2": {
        "gt": 77,
        "ap50": 0.7757264823570501,
        "ap50_95": 0.2564208389585382
      }
    },
    "mean_gt_iou_at_conf025": 0.5888519416881529,
    "definition": "AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área"
  },
  "classification": {
    "per_class": [
      {
        "class": 0,
        "f1": 0.9423076923076923,
        "auc": 0.9841628959276018,
        "tp": 49,
        "fp": 3,
        "fn": 3,
        "tn": 65
      },
      {
        "class": 1,
        "f1": 0.9605263157894737,
        "auc": 0.9954696466324373,
        "tp": 73,
        "fp": 2,
        "fn": 4,
        "tn": 41
      },
      {
        "class": 2,
        "f1": 0.9536423841059603,
        "auc": 0.9876170341286621,
        "tp": 72,
        "fp": 2,
        "fn": 5,
        "tn": 41
      }
    ],
    "macro_f1": 0.952158797401042,
    "macro_auc": 0.989083192229567
  },
  "semantic": {
    "dice": [
      0.4887390336519576,
      0.6301946841927794,
      0.6645512594507776
    ],
    "iou": [
      0.32339817181475544,
      0.46006149700288484,
      0.49762393663831034
    ],
    "macro_dice": 0.5944949924318382,
    "confusion": [
      [
        7583431,
        23815,
        50228,
        37678
      ],
      [
        17429,
        22395,
        1920,
        1814
      ],
      [
        9151,
        416,
        53265,
        131
      ],
      [
        9628,
        1460,
        667,
        50892
      ]
    ]
  },
  "scope": "Validación por paciente; Dice semántico no es Dice por fragmento"
}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
