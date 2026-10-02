"""Algoritmo de Supresión de No Máximos (NMS - Non-Maximum Suppression) implementado desde cero.

Requerimiento oficial de la sección 3.1 y 4.1 del documento de proyecto:
"La supresión de no máximos (NMS) deberá ser implementada por el equipo."
"Está explícitamente prohibido utilizar frameworks de alto nivel... la implementación final debe ser propia."
"""
from __future__ import annotations

import torch


def calcular_iou_boxes(caja_a: torch.Tensor, cajas_b: torch.Tensor) -> torch.Tensor:
    """Calcula el IoU (Intersection over Union) entre una caja y un conjunto de cajas.

    Args:
        caja_a: Tensor (4,) con [xmin, ymin, xmax, ymax]
        cajas_b: Tensor (M, 4) con [xmin, ymin, xmax, ymax]

    Returns:
        Tensor (M,) con el IoU para cada caja en cajas_b
    """
    # Intersección
    x1 = torch.maximum(caja_a[0], cajas_b[:, 0])
    y1 = torch.maximum(caja_a[1], cajas_b[:, 1])
    x2 = torch.minimum(caja_a[2], cajas_b[:, 2])
    y2 = torch.minimum(caja_a[3], cajas_b[:, 3])

    ancho_inter = torch.clamp(x2 - x1, min=0.0)
    alto_inter = torch.clamp(y2 - y1, min=0.0)
    area_inter = ancho_inter * alto_inter

    # Áreas individuales
    area_a = torch.clamp(caja_a[2] - caja_a[0], min=0.0) * torch.clamp(caja_a[3] - caja_a[1], min=0.0)
    area_b = torch.clamp(cajas_b[:, 2] - cajas_b[:, 0], min=0.0) * torch.clamp(cajas_b[:, 3] - cajas_b[:, 1], min=0.0)

    # Unión
    union = area_a + area_b - area_inter
    iou = area_inter / (union + 1e-7)
    return iou


def nms_propio(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
) -> torch.Tensor:
    """Aplica Supresión de No Máximos (NMS) estándar desde cero.

    Args:
        boxes: Tensor (N, 4) con [xmin, ymin, xmax, ymax]
        scores: Tensor (N,) con puntajes de confianza/probabilidad
        iou_threshold: Umbral de solapamiento para suprimir cajas redundantes
        score_threshold: Umbral mínimo para considerar una caja válida

    Returns:
        Tensor con los índices (LongTensor) de las cajas seleccionadas
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    # 1. Filtrar cajas con score inferior al umbral
    valid_mask = scores >= score_threshold
    if not valid_mask.any():
        return torch.empty(0, dtype=torch.long)

    indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    boxes = boxes[indices]
    scores = scores[indices]

    # 2. Ordenar las cajas por score descendente
    orden = torch.argsort(scores, descending=True)
    seleccionados = []

    while orden.numel() > 0:
        actual_idx = orden[0].item()
        seleccionados.append(indices[actual_idx])

        if orden.numel() == 1:
            break

        # Caja actual con mayor score
        caja_actual = boxes[orden[0]]
        otras_cajas = boxes[orden[1:]]

        # Calcular IoU con el resto de cajas candidatas
        ious = calcular_iou_boxes(caja_actual, otras_cajas)

        # Quedarse con las cajas cuyo solapamiento sea MENOR al umbral
        orden = orden[1:][ious < iou_threshold]

    if len(seleccionados) == 0:
        return torch.empty(0, dtype=torch.long)

    return torch.tensor(seleccionados, dtype=torch.long)


def nms_por_clase(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aplica NMS de forma independiente para cada clase anatómica (Sacro, Coxal Izq, Coxal Der).

    Retorna: (cajas_filtradas, scores_filtrados, clases_filtradas)
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    cajas_finales = []
    scores_finales = []
    labels_finales = []

    for c in torch.unique(labels):
        c_mask = labels == c
        c_boxes = boxes[c_mask]
        c_scores = scores[c_mask]

        keep = nms_propio(c_boxes, c_scores, iou_threshold=iou_threshold, score_threshold=score_threshold)
        if keep.numel() > 0:
            cajas_finales.append(c_boxes[keep])
            scores_finales.append(c_scores[keep])
            labels_finales.append(torch.full((len(keep),), c.item(), dtype=labels.dtype))

    if len(cajas_finales) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0, dtype=torch.long)

    return (
        torch.cat(cajas_finales, dim=0),
        torch.cat(scores_finales, dim=0),
        torch.cat(labels_finales, dim=0),
    )
