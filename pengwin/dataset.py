"""Dataset PyTorch para cortes axiales 2D y extracción de Bounding Boxes.

Requisitos de la sección 3 y 4 del proyecto:
- Cada muestra corresponde a un corte axial 2D con ventaneo HU calibrado.
- Genera anotaciones de detección (bounding boxes por región anatómica: Sacro, Coxal Izq, Coxal Der).
- Genera etiquetas de clasificación multietiqueta (presencia de las 3 regiones).
- Genera máscaras de segmentación por región.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .io import REGIONES, Caso, cargar_caso, region_de_etiqueta, ventana_hu

# Clases de detección: 0 = Sacro (SA), 1 = Coxal Izquierdo (LI), 2 = Coxal Derecho (RI)
NOMBRE_CLASES = ["Sacro", "Coxal izquierdo", "Coxal derecho"]
SIGLAS_CLASES = ["SA", "LI", "RI"]


def extraer_bboxes_region(etiqueta_2d: np.ndarray, normalizado: bool = True) -> list[dict]:
    """Extrae bounding boxes por región anatómica (1..3) presentes en el corte 2D.

    Retorna una lista de diccionarios con:
    - 'clase_idx': 0 (Sacro), 1 (Coxal Izq), 2 (Coxal Der)
    - 'sigla': 'SA', 'LI', 'RI'
    - 'bbox': [xmin, ymin, xmax, ymax]
    """
    h, w = etiqueta_2d.shape
    cajas = []

    for r_id, info in REGIONES.items():
        lo, hi = info["rango"]
        # Máscara binaria de la región macro
        mask_region = (etiqueta_2d >= lo) & (etiqueta_2d <= hi)
        if not mask_region.any():
            continue

        ys, xs = np.nonzero(mask_region)
        ymin, ymax = float(ys.min()), float(ys.max() + 1)
        xmin, xmax = float(xs.min()), float(xs.max() + 1)

        if normalizado:
            bbox = [xmin / w, ymin / h, xmax / w, ymax / h]
        else:
            bbox = [xmin, ymin, xmax, ymax]

        cajas.append({
            "clase_idx": r_id - 1,  # 0, 1, 2
            "sigla": info["sigla"],
            "bbox": bbox,
        })

    return cajas


class PelvisSliceDataset(Dataset):
    """Dataset 2D a nivel de corte axial a partir de los volúmenes 3D.

    Soporta precarga en memoria para batches rápidos y ajuste de resolución a 256x256.
    """

    def __init__(
        self,
        casos: Sequence[Caso],
        target_size: tuple[int, int] = (256, 256),
        solo_con_hueso: bool = True,
        canales: int = 3,
        transform: Callable | None = None,
    ):
        """
        Args:
            casos: Lista de objetos Caso (ya cargados en memoria)
            target_size: (H, W) de salida recomendada (por defecto 256x256)
            solo_con_hueso: Si True, filtra cortes sin ninguna región anotada
            canales: 1 o 3 (3 repite el canal en escala de grises para backbones preentrenados)
            transform: Transformaciones opcionales de aumento de datos
        """
        self.target_size = target_size
        self.canales = canales
        self.transform = transform
        self.muestras = []

        # Construir índice plano de cortes axiales válidos
        for caso in casos:
            nz = caso.ct.shape[0]
            for z in range(nz):
                mask_z = caso.etiqueta[z] if caso.etiqueta is not None else None
                tiene_hueso = (mask_z > 0).any() if mask_z is not None else False

                if solo_con_hueso and not tiene_hueso:
                    continue

                self.muestras.append({
                    "caso": caso,
                    "z": z,
                    "tiene_hueso": tiene_hueso,
                })

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, idx: int) -> dict:
        info = self.muestras[idx]
        caso: Caso = info["caso"]
        z: int = info["z"]

        # 1. Ventaneo HU óseo calibrado [400, 1800] -> [0, 1]
        corte_hu = ventana_hu(caso.ct[z])  # (H_orig, W_orig) float32
        h_orig, w_orig = corte_hu.shape

        # 2. Redimensionar imagen a target_size
        th, tw = self.target_size
        corte_resized = cv2.resize(corte_hu, (tw, th), interpolation=cv2.INTER_LINEAR)

        if self.canales == 3:
            img_tensor = torch.from_numpy(corte_resized).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        else:
            img_tensor = torch.from_numpy(corte_resized).unsqueeze(0)  # (1, H, W)

        # 3. Etiquetas de regiones y bounding boxes
        clases_vector = torch.zeros(3, dtype=torch.float32)  # [SA, LI, RI]
        boxes_list = []

        if caso.etiqueta is not None:
            mask_z = caso.etiqueta[z]
            cajas_dict = extraer_bboxes_region(mask_z, normalizado=True)

            for item in cajas_dict:
                c_idx = item["clase_idx"]
                clases_vector[c_idx] = 1.0
                # Guardamos [clase_idx, xmin, ymin, xmax, ymax]
                boxes_list.append([float(c_idx)] + item["bbox"])

            # Redimensionar máscara de segmentación categórica a target_size
            # Convertir etiquetas originales (1..30) a regiones (0: fondo, 1: SA, 2: LI, 3: RI)
            reg_mask = np.zeros_like(mask_z, dtype=np.uint8)
            for v in np.unique(mask_z):
                if v > 0:
                    reg_mask[mask_z == v] = region_de_etiqueta(int(v))
            mask_resized = cv2.resize(reg_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_resized).long()
        else:
            mask_tensor = torch.zeros(self.target_size, dtype=torch.long)

        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        else:
            boxes_tensor = torch.empty((0, 5), dtype=torch.float32)

        return {
            "imagen": img_tensor,                       # Tensor (C, H, W) float32 [0, 1]
            "boxes": boxes_tensor,                      # Tensor (N, 5) [clase_idx, xmin, ymin, xmax, ymax]
            "clases": clases_vector,                    # Tensor (3,) [SA, LI, RI] multilabel
            "mascara": mask_tensor,                     # Tensor (H, W) long [0, 1, 2, 3]
            "caso_id": caso.id,
            "z": z,
            "shape_original": (h_orig, w_orig),
        }


def collate_deteccion(batch: list[dict]) -> dict:
    """Collate function para DataLoader que agrupa imágenes y maneja número variable de cajas."""
    imagenes = torch.stack([b["imagen"] for b in batch])
    clases = torch.stack([b["clases"] for b in batch])
    mascaras = torch.stack([b["mascara"] for b in batch])
    caso_ids = [b["caso_id"] for b in batch]
    zs = [b["z"] for b in batch]

    # Para bboxes, añadimos un batch_idx al inicio: [batch_idx, clase_idx, xmin, ymin, xmax, ymax]
    boxes_con_batch = []
    for b_idx, item in enumerate(batch):
        bxs = item["boxes"]
        if len(bxs) > 0:
            indices = torch.full((len(bxs), 1), b_idx, dtype=torch.float32)
            boxes_con_batch.append(torch.cat([indices, bxs], dim=1))

    if len(boxes_con_batch) > 0:
        boxes_totales = torch.cat(boxes_con_batch, dim=0)
    else:
        boxes_totales = torch.empty((0, 6), dtype=torch.float32)

    return {
        "imagenes": imagenes,
        "boxes": boxes_totales,
        "clases": clases,
        "mascaras": mascaras,
        "caso_ids": caso_ids,
        "zs": zs,
    }
