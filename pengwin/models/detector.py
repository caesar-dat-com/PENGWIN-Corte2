"""Cabeza de detección basada en Grid propio y arquitectura completa del detector.

Requisitos de la sección 3.1 y 4.1 del documento de proyecto:
- "Detección de bounding boxes por región mediante grid propio y NMS propio."
- "Tres cabezas sobre un mismo backbone: clasificación y detección (y luego segmentación)."
- Prohibición explícita de frameworks como YOLO o Detectron2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import PelvisBackbone
from .nms import nms_por_clase


class GridDetectionHead(nn.Module):
    """Cabeza de detección basada en cuadrícula (Grid) S x S.

    Divide el mapa espacial en un grid S x S.
    Por cada celda (i, j), predice:
    - 1 valor de objetidad (objectness / confianza de presencia ósea)
    - 4 valores de coordenadas (tx, ty, tw, th) para la caja delimitadora
    - num_clases (3) logits para la región anatómica (Sacro, Coxal Izq, Coxal Der)
    Total de canales de salida: 1 + 4 + num_clases = 8 canales.
    """

    def __init__(self, in_channels: int, num_clases: int = 3, grid_size: int = 16):
        super().__init__()
        self.num_clases = num_clases
        self.grid_size = grid_size
        self.canales_salida = 1 + 4 + num_clases  # 8

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, self.canales_salida, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, S) -> salida: (B, 8, S, S)
        return self.head(x)


class PelvisDetector(nn.Module):
    """Modelo completo para detección y clasificación de regiones pélvicas.

    Integra:
    1. Backbone compartido (FundidoraPC o ResNet18)
    2. Bloque de atención CBAM
    3. Cabeza de Detección por Grid (Bounding Boxes)
    4. Cabeza de Clasificación multietiqueta a nivel de corte
    """

    def __init__(
        self,
        backbone_tipo: str = "fundidora",
        in_channels: int = 3,
        num_clases: int = 3,
        usar_cbam: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_clases = num_clases
        self.backbone = PelvisBackbone(
            tipo=backbone_tipo,
            in_channels=in_channels,
            usar_cbam=usar_cbam,
            pretrained=pretrained,
        )

        feat_channels = self.backbone.out_channels

        # Cabeza 1: Detección con Grid propio
        self.detection_head = GridDetectionHead(feat_channels, num_clases=num_clases)

        # Cabeza 2: Clasificación multietiqueta del corte completo
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_clases),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pred_grid = self.detection_head(features)
        pred_cls = self.classification_head(features)

        return {
            "grid": pred_grid,     # (B, 8, S, S)
            "clases": pred_cls,    # (B, 3) logits
            "features": features,  # (B, C, S, S) útil para la futura cabeza de segmentación
        }

    def inferir_boxes(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> list[dict]:
        """Realiza inferencia completa (Forward -> Decodificación de Grid -> NMS propio)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            grid = out["grid"]
            b, _, s_h, s_w = grid.shape

            resultados = []
            for i in range(b):
                cajas, scores, clases = decodificar_grid(grid[i], conf_threshold=conf_threshold)
                if len(cajas) > 0:
                    cajas_f, scores_f, clases_f = nms_por_clase(
                        cajas, scores, clases,
                        iou_threshold=iou_threshold,
                        score_threshold=conf_threshold,
                    )
                else:
                    cajas_f = torch.empty((0, 4))
                    scores_f = torch.empty(0)
                    clases_f = torch.empty(0, dtype=torch.long)

                probs_corte = torch.sigmoid(out["clases"][i])

                resultados.append({
                    "boxes": cajas_f,
                    "scores": scores_f,
                    "clases": clases_f,
                    "prob_corte": probs_corte,
                })

            return resultados


def decodificar_grid(
    grid: torch.Tensor,
    conf_threshold: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decodifica el mapa (8, S, S) de una imagen a cajas normalizadas [xmin, ymin, xmax, ymax]."""
    # grid: (8, S, S)
    _, s_h, s_w = grid.shape
    obj_logits = grid[0]                   # (S, S)
    box_raw = grid[1:5]                    # (4, S, S)
    cls_logits = grid[5:]                  # (3, S, S)

    obj_prob = torch.sigmoid(obj_logits)
    cls_probs = torch.softmax(cls_logits, dim=0)
    cls_max_prob, cls_idx = torch.max(cls_probs, dim=0)

    # Score combinado de detección: P(obj) * P(clase|obj)
    scores = obj_prob * cls_max_prob

    # Malla de coordenadas de celda
    grid_y, grid_x = torch.meshgrid(
        torch.arange(s_h, device=grid.device),
        torch.arange(s_w, device=grid.device),
        indexing="ij",
    )

    # Decodificación de centro y tamaño (en [0, 1])
    cx = (grid_x.float() + torch.sigmoid(box_raw[0])) / s_w
    cy = (grid_y.float() + torch.sigmoid(box_raw[1])) / s_h
    w = torch.sigmoid(box_raw[2])
    h = torch.sigmoid(box_raw[3])

    xmin = torch.clamp(cx - w / 2.0, min=0.0, max=1.0)
    ymin = torch.clamp(cy - h / 2.0, min=0.0, max=1.0)
    xmax = torch.clamp(cx + w / 2.0, min=0.0, max=1.0)
    ymax = torch.clamp(cy + h / 2.0, min=0.0, max=1.0)

    boxes_todas = torch.stack([xmin, ymin, xmax, ymax], dim=-1).reshape(-1, 4)
    scores_todos = scores.reshape(-1)
    clases_todas = cls_idx.reshape(-1)

    # Filtrar por umbral de confianza preliminar
    mask = scores_todos >= conf_threshold
    return boxes_todas[mask], scores_todos[mask], clases_todas[mask]
