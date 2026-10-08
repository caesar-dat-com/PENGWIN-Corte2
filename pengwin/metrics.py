"""Módulo de Métricas y Análisis Clínico Cuantitativo (Semana 10 - Fases 2 y 3).

Implementa:
1. Métricas de segmentación multi-región: Dice Similarity Coefficient (DSC) e IoU (Jaccard).
2. Cálculo métrico de separación física entre fragmentos y regiones en MILÍMETROS (mm).
   - Usa scipy.ndimage.distance_transform_edt con espaciado real del header SimpleITK (sz, sy, sx).
   - Prohíbe estrictamente reportar distancias en píxeles.
3. Evaluación comparativa entre predicción del modelo y Ground Truth.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch


def calcular_dice_iou_multiclase(
    pred_mask: np.ndarray | torch.Tensor,
    target_mask: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    nombres_clases: list[str] | None = None,
    smooth: float = 1e-5,
) -> dict[str, dict[str, float] | float]:
    """Calcula Dice Similarity Coefficient (DSC) e Intersection over Union (IoU) por clase.

    Parámetros:
    - pred_mask: Array o Tensor de etiquetas discretas predichas (H, W) o (B, H, W).
    - target_mask: Array o Tensor de etiquetas discretas reales (H, W) o (B, H, W).
    - num_classes: Número de clases (por defecto 4: 0: Fondo, 1: Sacro, 2: Coxal Izq, 3: Coxal Der).
    - nombres_clases: Lista opcional con los nombres legibles de cada clase.

    Retorna:
    - Dict con métricas por clase y promedios globales (mDice, mIoU) sobre las regiones anatómicas.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.detach().cpu().numpy()

    pred_mask = pred_mask.astype(np.int64)
    target_mask = target_mask.astype(np.int64)

    if nombres_clases is None:
        nombres_clases = ["Fondo", "Sacro (SA)", "Coxal Izq (LI)", "Coxal Der (RI)"]

    resultados: dict[str, dict[str, float] | float] = {"por_clase": {}}

    dice_anatomicos = []
    iou_anatomicos = []

    for c in range(num_classes):
        p_c = pred_mask == c
        t_c = target_mask == c

        interseccion = np.sum(p_c & t_c)
        area_p = np.sum(p_c)
        area_t = np.sum(t_c)
        union = np.sum(p_c | t_c)

        if area_p == 0 and area_t == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = float((2.0 * interseccion + smooth) / (area_p + area_t + smooth))
            iou = float((interseccion + smooth) / (union + smooth))

        nombre = nombres_clases[c] if c < len(nombres_clases) else f"Clase_{c}"
        resultados["por_clase"][nombre] = {
            "dice": dice,
            "iou": iou,
            "area_pred": int(area_p),
            "area_gt": int(area_t),
        }

        # Promedio macro sobre las estructuras anatómicas de interés (excluyendo fondo c=0)
        if c > 0:
            dice_anatomicos.append(dice)
            iou_anatomicos.append(iou)

    resultados["mDice_anatomico"] = float(np.mean(dice_anatomicos)) if dice_anatomicos else 0.0
    resultados["mIoU_anatomico"] = float(np.mean(iou_anatomicos)) if iou_anatomicos else 0.0

    return resultados


def calcular_mapa_distancia_edt_mm(
    mascara_binaria: np.ndarray,
    spacing_yx: tuple[float, float],
) -> np.ndarray:
    """Calcula la Transformada de Distancia Euclidiana Exacta (EDT) en milímetros.

    Utiliza el espaciado físico real (sy_mm, sx_mm) proveniente del header DICOM/MHA.
    Cada píxel del mapa resultante contiene la distancia geodésica euclidiana más cercana
    en MILÍMETROS hacia el borde de la estructura dada.

    Parámetros:
    - mascara_binaria: Array 2D booleano o binario de la estructura ósea (True donde hay hueso).
    - spacing_yx: Tupla (sy_mm, sx_mm) con la calibración física de cada píxel en mm.

    Retorna:
    - mapa_distancia_mm: Array 2D float con la distancia física en mm en el espacio exterior.
    """
    assert mascara_binaria.ndim == 2, f"Se requiere máscara 2D, se recibió shape {mascara_binaria.shape}"
    # Si la máscara está vacía, no hay estructura de referencia
    if not np.any(mascara_binaria):
        return np.full_like(mascara_binaria, fill_value=np.nan, dtype=np.float32)

    # La distancia exterior se calcula desde el fondo (~mascara) hacia el borde del objeto
    fondo = ~mascara_binaria
    dist_map_mm = ndi.distance_transform_edt(fondo, sampling=spacing_yx)
    return dist_map_mm.astype(np.float32)


def medir_distancia_minima_entre_regiones_mm(
    mascara_a: np.ndarray,
    mascara_b: np.ndarray,
    spacing_yx: tuple[float, float],
) -> tuple[float, tuple[int, int] | None, tuple[int, int] | None]:
    """Mide con precisión sub-milimétrica la distancia mínima de separación entre dos regiones.

    Calcula la Transformada de Distancia Euclidiana (EDT) de la máscara A con sampling=(sy, sx)
    y evalúa el valor mínimo alcanzado sobre el soporte espacial de la máscara B.

    Parámetros:
    - mascara_a: Estructura ósea A (e.g. Coxal Derecho).
    - mascara_b: Estructura ósea B o fragmento separado (e.g. Sacro o Fragmento desplazado).
    - spacing_yx: (sy_mm, sx_mm) en milímetros.

    Retorna:
    - distancia_min_mm: Distancia mínima física en mm (0.0 si contactan).
    - coord_a: Punto (y, x) en la frontera de A más cercano a B.
    - coord_b: Punto (y, x) en la frontera de B más cercano a A.
    """
    mask_a = mascara_a.astype(bool)
    mask_b = mascara_b.astype(bool)

    if not np.any(mask_a) or not np.any(mask_b):
        return float("inf"), None, None

    # Si hay solapamiento o contacto directo
    if np.any(mask_a & mask_b):
        return 0.0, None, None

    # Mapa EDT físico desde la frontera de A hacia todo el espacio
    dist_edt_desde_a = ndi.distance_transform_edt(~mask_a, sampling=spacing_yx)

    # Distancias sobre los píxeles de B
    distancias_en_b = dist_edt_desde_a[mask_b]
    dist_min_mm = float(np.min(distancias_en_b))

    # Coordenadas óptimas de contacto mínimo
    coords_b = np.argwhere(mask_b)
    idx_min = np.argmin(distancias_en_b)
    pt_b = tuple(coords_b[idx_min])

    # Encontrar el punto más cercano en A usando EDT inverso desde el punto pt_b
    mapa_pt = np.zeros_like(mask_a)
    mapa_pt[pt_b[0], pt_b[1]] = True
    dist_desde_pt = ndi.distance_transform_edt(~mapa_pt, sampling=spacing_yx)
    dist_en_a = dist_desde_pt[mask_a]
    coords_a = np.argwhere(mask_a)
    pt_a = tuple(coords_a[np.argmin(dist_en_a)])

    return dist_min_mm, pt_a, pt_b
