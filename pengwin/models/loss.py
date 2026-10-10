"""Función de pérdida compuesta multitarea para detección y clasificación de regiones pélvicas.

Requerimiento oficial de la sección 4.1:
"Función de pérdida compuesta multitarea: los pesos (lambda) de cada término deberán ser calibrados
y justificados explícitamente en el informe."
Incluye modulación Focal con gamma (por ejemplo gamma=0.5 o 2.0) para combatir el desbalance de fondo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_binaria(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 0.5, alpha: float = 0.25, balanced: bool = False) -> torch.Tensor:
    """Focal Loss binaria para manejar el severo desbalance de celdas de fondo vs. regiones óseas."""
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    if balanced:
        pos=targets.bool()
        return (loss[pos].mean() if pos.any() else logits.sum()*0) + (loss[~pos].mean() if (~pos).any() else logits.sum()*0)
    return loss.mean()


class PelvisDetectionLoss(nn.Module):
    """Pérdida multitarea compuesta para la cabeza de detección y clasificación.

    L_total = lambda_obj * L_obj + lambda_box * L_box + lambda_cls * L_cls + lambda_slice * L_slice
    """

    def __init__(
        self,
        lambda_obj: float = 2.0,
        lambda_box: float = 5.0,
        lambda_cls: float = 1.0,
        lambda_slice: float = 1.0,
        gamma: float = 0.5,
        balancear_obj: bool = False,
    ):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_slice = lambda_slice
        self.gamma = gamma
        self.balancear_obj = balancear_obj

        self.box_loss_fn = nn.SmoothL1Loss(reduction="mean")
        self.slice_loss_fn = nn.BCEWithLogitsLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        predicciones: dict[str, torch.Tensor],
        boxes_gt: torch.Tensor,
        clases_slice_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predicciones: Salida de PelvisDetector ('grid': (B, 8, S, S), 'clases': (B, 3))
            boxes_gt: Tensor (N, 6) con [batch_idx, clase_idx, xmin, ymin, xmax, ymax]
            clases_slice_gt: Tensor (B, 3) vector multietiqueta del corte
        """
        grid = predicciones["grid"]
        b, _, s_h, s_w = grid.shape
        device = grid.device

        # Matrices de target en el grid
        target_obj = torch.zeros((b, s_h, s_w), device=device)
        target_box = torch.zeros((b, 4, s_h, s_w), device=device)
        target_cls = torch.zeros((b, s_h, s_w), dtype=torch.long, device=device)
        mask_pos = torch.zeros((b, s_h, s_w), dtype=torch.bool, device=device)

        # Asignar cada caja Ground Truth a su celda correspondiente en el grid
        if boxes_gt.numel() > 0:
            for caja in boxes_gt:
                b_idx = int(caja[0].item())
                c_idx = int(caja[1].item())
                xmin, ymin, xmax, ymax = caja[2:].tolist()

                # Centro de la caja en coordenadas normalizadas
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                w = max(xmax - xmin, 1e-4)
                h = max(ymax - ymin, 1e-4)

                # Celda en el grid
                gi = min(int(cx * s_w), s_w - 1)
                gj = min(int(cy * s_h), s_h - 1)

                if mask_pos[b_idx, gj, gi]:
                    raise ValueError("Dos regiones comparten una celda; no sobrescribir el objetivo silenciosamente")
                target_obj[b_idx, gj, gi] = 1.0
                mask_pos[b_idx, gj, gi] = True

                # Targets de regresión relativos a la celda
                # cx = (gi + sigmoid(tx)) / s_w => sigmoid(tx) = cx * s_w - gi
                tx_target = cx * s_w - gi
                ty_target = cy * s_h - gj
                target_box[b_idx, 0, gj, gi] = tx_target
                target_box[b_idx, 1, gj, gi] = ty_target
                target_box[b_idx, 2, gj, gi] = w
                target_box[b_idx, 3, gj, gi] = h
                target_cls[b_idx, gj, gi] = c_idx

        # 1. Pérdida de objetidad (Focal Loss)
        pred_obj_logits = grid[:, 0]  # (B, S, S)
        loss_obj = focal_loss_binaria(pred_obj_logits, target_obj, gamma=self.gamma, balanced=self.balancear_obj)

        # 2. Pérdida de cajas y de clasificación en celdas positivas
        if mask_pos.any():
            pred_box = grid[:, 1:5]  # (B, 4, S, S)
            # Predicciones decodificadas a la misma escala que los targets
            pred_tx = torch.sigmoid(pred_box[:, 0][mask_pos])
            pred_ty = torch.sigmoid(pred_box[:, 1][mask_pos])
            pred_w = torch.sigmoid(pred_box[:, 2][mask_pos])
            pred_h = torch.sigmoid(pred_box[:, 3][mask_pos])

            pred_boxes_pos = torch.stack([pred_tx, pred_ty, pred_w, pred_h], dim=1)
            target_boxes_pos = target_box[:, :, mask_pos.squeeze(1) if mask_pos.ndim == 4 else mask_pos].permute(1, 0) if False else torch.stack([
                target_box[:, 0][mask_pos],
                target_box[:, 1][mask_pos],
                target_box[:, 2][mask_pos],
                target_box[:, 3][mask_pos],
            ], dim=1)

            loss_box = self.box_loss_fn(pred_boxes_pos, target_boxes_pos)

            # Clasificación de la región en la celda
            pred_cls_logits = grid[:, 5:]  # (B, 3, S, S)
            pred_cls_pos = pred_cls_logits.permute(0, 2, 3, 1)[mask_pos]  # (N_pos, 3)
            target_cls_pos = target_cls[mask_pos]
            loss_cls = self.cls_loss_fn(pred_cls_pos, target_cls_pos)
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_cls = torch.tensor(0.0, device=device)

        # 3. Pérdida de clasificación a nivel de corte
        loss_slice = self.slice_loss_fn(predicciones["clases"], clases_slice_gt)

        # Pérdida total ponderada
        loss_total = (
            self.lambda_obj * loss_obj
            + self.lambda_box * loss_box
            + self.lambda_cls * loss_cls
            + self.lambda_slice * loss_slice
        )

        return {
            "loss_total": loss_total,
            "loss_obj": loss_obj,
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "loss_slice": loss_slice,
        }
