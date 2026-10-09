"""Demostración y Validación Ejecutable de Semana 10 (Fases 1, 2 y 3).

Demostración histórica parcial, no validación del avance completo.
Mide distancias 2D entre regiones distintas, no fragmento-principal 3D.
Conserva la arquitectura del equipo (Prof. Carlos A. Ferro):
1. FASE 1: Cabeza de Segmentación densa de regiones y fragmentos pélvicos
   - Reconstrucción en exactamente 8 canales latentes (restricción: NO más de 10 canales).
   - Bloques de doble convolución (DoubleConv) en cada etapa de subida (16 -> 32 -> 64 -> 128 -> 256).
   - Sin frameworks prohibidos (sin YOLO, sin Detectron2, sin segmentadores pre-enlatados).
2. FASE 2: Entrenamiento Multitarea y Evaluación con Métricas Oficiales
   - Pérdida combinada: Detección (Focal + SmoothL1 + CE) + Segmentación (CE Ponderada + Dice Loss).
   - Evaluación cuantitativa con Dice Similarity Coefficient (DSC) e Intersection over Union (IoU)
     por cada región anatómica (Sacro, Coxal Izquierdo, Coxal Derecho).
3. FASE 3: Medición Métrica de Separación en Milímetros (mm)
   - Uso obligatorio de la Transformada de Distancia Euclidiana Exacta (scipy.ndimage.distance_transform_edt).
   - Calibración física estricta con el espaciado real del header SimpleITK (sz, sy, sx en mm).
   - Prohibición de reportar distancias en píxeles.
   - Comparación cuantitativa: Distancia de separación predicha vs. Ground Truth real.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Añadir raíz del repositorio al path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from pengwin import io, dataset
from pengwin.metrics import (
    calcular_dice_iou_multiclase,
    calcular_mapa_distancia_edt_mm,
    medir_distancia_minima_entre_regiones_mm,
)
from pengwin.models import (
    PelvisDetector,
    PelvisDetectionLoss,
    CombinedSegmentationLoss,
)

# Fijar semilla de reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SALIDAS = ROOT_DIR / "salidas"
SALIDAS.mkdir(exist_ok=True)

print("=" * 80)
print("INICIANDO EJECUCIÓN: SEMANA 10 - FASES 1, 2 Y 3")
print("=" * 80)

# ==============================================================================
# PASO 1: CARGA DE CORTES CLÍNICOS REALES DEL CASO 001
# ==============================================================================
DIR_IMG, DIR_LBL = io.rutas_dataset()

print("\n[Paso 1] Cargando Caso 001 real desde rutas configuradas...")
caso = io.cargar_caso("001", DIR_IMG, DIR_LBL)
assert caso.etiqueta is not None, "El caso 001 debe contener las máscaras anotadas oficiales"

sz_mm, sy_mm, sx_mm = caso.spacing
print(f"-> Volumen cargado: {caso.ct.shape} vóxeles")
print(f"-> Espaciado físico real del header: sz={sz_mm:.4f} mm, sy={sy_mm:.4f} mm, sx={sx_mm:.4f} mm")

# Cortes axiales representativos:
# z=134 (inferior: solo coxales), z=220 (medio: sacro posterior), z=235, z=250 (alto)
Z_INDICES = [134, 220, 235, 250]
B = len(Z_INDICES)
TARGET_SIZE = (256, 256)

# Factor de escalado de espaciado físico al redimensionar de (512, 512) a (256, 256)
orig_h, orig_w = caso.ct.shape[1], caso.ct.shape[2]
escala_y = orig_h / TARGET_SIZE[0]  # normalmente 512 / 256 = 2.0
escala_x = orig_w / TARGET_SIZE[1]
spacing_256_yx = (sy_mm * escala_y, sx_mm * escala_x)
print(f"-> Espaciado 2D calibrado a resolución 256x256: sy={spacing_256_yx[0]:.4f} mm/px, sx={spacing_256_yx[1]:.4f} mm/px")

imagenes_batch = torch.zeros((B, 3, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=torch.float32)
mascaras_gt_batch = torch.zeros((B, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=torch.long)
boxes_gt_list = []
clases_slice_list = []

for b_idx, z in enumerate(Z_INDICES):
    # 1. Ventaneo clínico óseo (400 HU nivel, 1800 HU ancho -> [-500, +1300] HU)
    ct_slice = io.ventana_hu(caso.ct[z], nivel=400, ancho=1800)
    ct_256 = cv2.resize(ct_slice, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    imagenes_batch[b_idx] = torch.from_numpy(np.repeat(ct_256[np.newaxis, :, :], 3, axis=0))

    # 2. Máscara de segmentación discreta por región: 0: Fondo, 1: Sacro, 2: Coxal Izq, 3: Coxal Der
    lbl_slice = caso.etiqueta[z]
    # Mapear valores brutos PENGWIN (1..10 -> 1, 11..20 -> 2, 21..30 -> 3)
    lbl_region_2d = np.zeros_like(lbl_slice, dtype=np.uint8)
    for r_id, info in io.REGIONES.items():
        lo, hi = info["rango"]
        lbl_region_2d[(lbl_slice >= lo) & (lbl_slice <= hi)] = r_id

    # Redimensionar máscara con nearest neighbor para no crear valores espurios
    lbl_256 = cv2.resize(lbl_region_2d, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    mascaras_gt_batch[b_idx] = torch.from_numpy(lbl_256).long()

    # 3. Cajas delimitadoras de detección y presencia por corte
    cajas = dataset.extraer_bboxes_region(lbl_slice, min_pixeles=0, normalizado=True)
    clase_presente = [0.0, 0.0, 0.0]  # SA, LI, RI
    for c in cajas:
        c_idx = c["clase_idx"]
        clase_presente[c_idx] = 1.0
        xmin, ymin, xmax, ymax = c["bbox"]
        boxes_gt_list.append([float(b_idx), float(c_idx), xmin, ymin, xmax, ymax])

    clases_slice_list.append(clase_presente)
    regiones_str = ", ".join([c["sigla"] for c in cajas])
    print(f"  Corte {b_idx} (z={z}): Regiones={regiones_str} | Píxeles óseos 256x256={np.sum(lbl_256 > 0)}")

boxes_gt = torch.tensor(boxes_gt_list, dtype=torch.float32)
clases_slice = torch.tensor(clases_slice_list, dtype=torch.float32)

# ==============================================================================
# PASO 2: INICIALIZACIÓN DEL MODELO MULTITAREA CON CABEZA DE SEGMENTACIÓN (FASE 1)
# ==============================================================================
print("\n[Paso 2] Inicializando Arquitectura Multitarea de 3 Cabezas (Fase 1)...")
print("  - Backbone compartido: FundidoraPC con bloque de atención CBAM (kernel 9x9)")
print("  - Cabeza 1: Detección por Grid propio 16x16")
print("  - Cabeza 2: Clasificación multietiqueta del corte")
print("  - Cabeza 3 (Semana 10): PelvisSegmentationHead (8 canales latentes <= 10, DoubleConv, 256x256)")

modelo = PelvisDetector(
    backbone_tipo="fundidora",
    in_channels=3,
    num_clases=3,
    usar_cbam=True,
    con_segmentacion=True,
    num_clases_seg=4,
    canales_latentes_seg=8,  # Restricción docente: <= 10 canales latentes
)

criterio_det = PelvisDetectionLoss(lambda_obj=2.0, lambda_box=5.0, lambda_cls=1.0, lambda_slice=1.0, gamma=0.5)
criterio_seg = CombinedSegmentationLoss(lambda_ce=1.0, lambda_dice=1.5)
optimizador = optim.AdamW(modelo.parameters(), lr=1.2e-3, weight_decay=1e-4)

# ==============================================================================
# PASO 3: ENTRENAMIENTO MULTITAREA (FASE 2)
# ==============================================================================
num_epocas = 120
print(f"\n[Paso 3] Entrenando modelo multitarea durante {num_epocas} épocas (Fase 2)...")
modelo.train()

historial = {"total": [], "det": [], "seg": [], "dice": []}

for epoca in range(1, num_epocas + 1):
    optimizador.zero_grad()
    salidas_modelo = modelo(imagenes_batch)

    # 1. Pérdida de Detección
    loss_det_dict = criterio_det(salidas_modelo, boxes_gt, clases_slice)
    loss_det = loss_det_dict["loss_total"]

    # 2. Pérdida de Segmentación
    logits_seg = salidas_modelo["mascaras"]  # (B, 4, 256, 256)
    loss_seg_dict = criterio_seg(logits_seg, mascaras_gt_batch)
    loss_seg = loss_seg_dict["loss_total"]

    # 3. Pérdida Multitarea Compuesta
    loss_multitarea = loss_det + 1.2 * loss_seg

    loss_multitarea.backward()
    optimizador.step()

    historial["total"].append(loss_multitarea.item())
    historial["det"].append(loss_det.item())
    historial["seg"].append(loss_seg.item())
    historial["dice"].append(loss_seg_dict["loss_dice"].item())

    if epoca % 30 == 0 or epoca == 1:
        print(f"  Época [{epoca:03d}/{num_epocas}] | Loss Multitarea: {loss_multitarea.item():.4f} "
              f"| Det: {loss_det.item():.4f} | Seg: {loss_seg.item():.4f} (DiceLoss: {loss_seg_dict['loss_dice'].item():.4f})")

# ==============================================================================
# PASO 4: EVALUACIÓN CUANTITATIVA DE SEGMENTACIÓN (DICE & IOU)
# ==============================================================================
print("\n[Paso 4] Evaluando Métricas Clínicas de Segmentación (Dice Similarity & IoU)...")
modelo.eval()
with torch.no_grad():
    salidas_eval = modelo(imagenes_batch)
    pred_probs = torch.softmax(salidas_eval["mascaras"], dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1).cpu().numpy()  # (B, 256, 256)

metricas_globales = calcular_dice_iou_multiclase(
    pred_labels,
    mascaras_gt_batch.cpu().numpy(),
    num_classes=4,
    nombres_clases=["Fondo", "Sacro (SA)", "Coxal Izquierdo (LI)", "Coxal Derecho (RI)"],
)

filas_csv = []
print("  Resultados por Región Anatómica:")
for nombre_clase, valores in metricas_globales["por_clase"].items():
    print(f"    - {nombre_clase:22s} | Dice: {valores['dice']:.4f} | IoU: {valores['iou']:.4f} "
          f"| Área Pred: {valores['area_pred']} px | Área GT: {valores['area_gt']} px")
    filas_csv.append({
        "Estructura": nombre_clase,
        "Dice_Score": f"{valores['dice']:.4f}",
        "IoU_Jaccard": f"{valores['iou']:.4f}",
        "Area_Pred_px": valores["area_pred"],
        "Area_GT_px": valores["area_gt"],
    })

print(f"  -> Media Anatómica Dice (mDice): {metricas_globales['mDice_anatomico']:.4f}")
print(f"  -> Media Anatómica IoU (mIoU):   {metricas_globales['mIoU_anatomico']:.4f}")

df_metricas = pd.DataFrame(filas_csv)
df_metricas.to_csv(SALIDAS / "metricas_fase2_segmentacion.csv", index=False)
print(f"  Tabla de métricas guardada en: {SALIDAS / 'metricas_fase2_segmentacion.csv'}")

# ==============================================================================
# PASO 5: CÁLCULO MÉTRICO DE SEPARACIÓN EN MILÍMETROS (FASE 3 - EDT)
# ==============================================================================
print("\n[Paso 5] Midiendo Separación Física de Regiones y Fragmentos en Milímetros (Fase 3)...")
print("  Utilizando Transformada de Distancia Euclidiana Exacta (EDT) con espaciado real...")

# Analizar la separación física en el corte representativo z=220 (Muestra 1)
# Estructuras: Coxal Derecho (RI, clase 3) y Sacro (SA, clase 1)
idx_corte_edt = 1  # z=220
z_corte_val = Z_INDICES[idx_corte_edt]

gt_corte = mascaras_gt_batch[idx_corte_edt].cpu().numpy()
pred_corte = pred_labels[idx_corte_edt]

# Máscaras binarias de Coxal Derecho (RI, clase 3) y Sacro (SA, clase 1)
gt_sa = gt_corte == 1
gt_ri = gt_corte == 3

pred_sa = pred_corte == 1
pred_ri = pred_corte == 3

# 1. Distancia real en Ground Truth
dist_gt_mm, pt_a_gt, pt_b_gt = medir_distancia_minima_entre_regiones_mm(gt_ri, gt_sa, spacing_256_yx)
# 2. Distancia predicha por el modelo
dist_pred_mm, pt_a_pred, pt_b_pred = medir_distancia_minima_entre_regiones_mm(pred_ri, pred_sa, spacing_256_yx)

error_absoluto_mm = abs(dist_pred_mm - dist_gt_mm)

print(f"  Resultados métricos para corte z={z_corte_val}:")
print(f"    - Distancia GT (Sacro <-> Coxal Derecho):      {dist_gt_mm:.2f} mm")
print(f"    - Distancia Predicha (Sacro <-> Coxal Der):   {dist_pred_mm:.2f} mm")
print(f"    - Error absoluto de medición métrica:         {error_absoluto_mm:.2f} mm")

# También medir entre Coxal Izquierdo (clase 2) y Sacro (clase 1)
gt_li = gt_corte == 2
pred_li = pred_corte == 2
dist_gt_li_mm, _, _ = medir_distancia_minima_entre_regiones_mm(gt_li, gt_sa, spacing_256_yx)
dist_pred_li_mm, _, _ = medir_distancia_minima_entre_regiones_mm(pred_li, pred_sa, spacing_256_yx)
print(f"    - Distancia GT (Sacro <-> Coxal Izquierdo):   {dist_gt_li_mm:.2f} mm")
print(f"    - Distancia Predicha (Sacro <-> Coxal Izq):   {dist_pred_li_mm:.2f} mm")

# Guardar registro métrico en CSV
df_distancias = pd.DataFrame([
    {
        "Corte_z": z_corte_val,
        "Par_Estructuras": "Sacro <-> Coxal Derecho",
        "Distancia_GT_mm": round(dist_gt_mm, 2),
        "Distancia_Pred_mm": round(dist_pred_mm, 2),
        "Error_Absoluto_mm": round(error_absoluto_mm, 2),
        "Espaciado_Pixel_mm": round(spacing_256_yx[0], 4),
    },
    {
        "Corte_z": z_corte_val,
        "Par_Estructuras": "Sacro <-> Coxal Izquierdo",
        "Distancia_GT_mm": round(dist_gt_li_mm, 2),
        "Distancia_Pred_mm": round(dist_pred_li_mm, 2),
        "Error_Absoluto_mm": round(abs(dist_pred_li_mm - dist_gt_li_mm), 2),
        "Espaciado_Pixel_mm": round(spacing_256_yx[0], 4),
    },
])
df_distancias.to_csv(SALIDAS / "metricas_fase3_separacion_mm.csv", index=False)
print(f"  Tabla métrica de distancias guardada en: {SALIDAS / 'metricas_fase3_separacion_mm.csv'}")

# ==============================================================================
# PASO 6: GENERACIÓN DE FIGURAS DIAGNÓSTICAS DE ALTA CALIDAD
# ==============================================================================
print("\n[Paso 6] Generando figuras diagnósticas para sustentar Fases 1, 2 y 3...")

# Paleta de colores clínicos estandarizada
# 0: Fondo (transparente), 1: Sacro (Naranja #E4572E), 2: LI (Azul #29335C / Cian), 3: RI (Turquesa #17BEBB)
colores_cmap = np.array([
    [0.0, 0.0, 0.0, 0.0],       # Fondo transparente
    [0.89, 0.34, 0.18, 0.65],   # Sacro (Naranja)
    [0.16, 0.55, 0.95, 0.65],   # Coxal Izq (Azul eléctrico)
    [0.09, 0.75, 0.73, 0.65],   # Coxal Der (Turquesa)
])

# FIGURA 1: Comparativa de Segmentación Densa (GT vs Predicción del Modelo)
fig, axes = plt.subplots(2, B, figsize=(18, 9), facecolor="#0B132B")
nombres_cortes = ["z=134 (Inferior)", "z=220 (Medio/Sacro)", "z=235 (Medio-Alto)", "z=250 (Alto/Crestas)"]

for j in range(B):
    ct_img = imagenes_batch[j, 0].cpu().numpy()

    # Fila 0: Ground Truth
    axes[0, j].imshow(ct_img, cmap="gray", vmin=0, vmax=1)
    overlay_gt = colores_cmap[mascaras_gt_batch[j].cpu().numpy()]
    axes[0, j].imshow(overlay_gt)
    axes[0, j].set_title(f"GT Real Oficial · {nombres_cortes[j]}", color="#FFD166", fontsize=11, fontweight="bold")
    axes[0, j].axis("off")

    # Fila 1: Predicción Cabeza Segmentación
    axes[1, j].imshow(ct_img, cmap="gray", vmin=0, vmax=1)
    overlay_pred = colores_cmap[pred_labels[j]]
    axes[1, j].imshow(overlay_pred)
    axes[1, j].set_title(f"Predicción Segmenter · {nombres_cortes[j]}", color="#00E5FF", fontsize=11, fontweight="bold")
    axes[1, j].axis("off")

# Añadir leyenda de estructuras
patches_leyenda = [
    patches.Patch(facecolor="#E4572E", edgecolor="white", label="Sacro (SA)"),
    patches.Patch(facecolor="#298EDB", edgecolor="white", label="Coxal Izquierdo (LI)"),
    patches.Patch(facecolor="#17BEBB", edgecolor="white", label="Coxal Derecho (RI)"),
]
fig.legend(handles=patches_leyenda, loc="lower center", ncol=3, facecolor="#1C2541", edgecolor="#3A506B", labelcolor="white", fontsize=11)

fig.suptitle(
    "FASE 1 Y 2: RECONSTRUCCIÓN Y SEGMENTACIÓN MULTI-REGIÓN DE PELVIS (CASO 001)\n"
    f"Cabeza ligera (8 canales latentes <= 10) | mDice Anatómico: {metricas_globales['mDice_anatomico']:.4f} | mIoU: {metricas_globales['mIoU_anatomico']:.4f}",
    color="white", fontsize=13, fontweight="bold", y=0.98,
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(SALIDAS / "fase1_segmentacion_predicciones.png", dpi=130)
plt.close(fig)
print(f"  -> Figura de segmentación guardada en: {SALIDAS / 'fase1_segmentacion_predicciones.png'}")

# FIGURA 2: Curva de Pérdida Multitarea Compuesta
fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="#0B132B")
ax.set_facecolor("#1C2541")
ax.plot(historial["total"], label="Pérdida Total Compuesta (L_det + 1.2 L_seg)", color="#00E5FF", linewidth=2.2)
ax.plot(historial["det"], label="Pérdida Detección Grid + Boxes", color="#FFD166", linestyle="--")
ax.plot(historial["seg"], label="Pérdida Segmentación (CE + Dice)", color="#EF476F", linewidth=1.8)
ax.plot(historial["dice"], label="Dice Loss de la Cabeza Segmenter", color="#06D6A0", linestyle=":")

ax.set_title("CONVERGENCIA MULTITAREA: DETECCIÓN + SEGMENTACIÓN (SEMANA 10)", color="white", fontsize=12, fontweight="bold")
ax.set_xlabel("Época", color="white")
ax.set_ylabel("Pérdida (Loss)", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#3A506B")
ax.grid(True, linestyle="--", alpha=0.3, color="#3A506B")
ax.legend(facecolor="#1C2541", edgecolor="#3A506B", labelcolor="white")
plt.tight_layout()
fig.savefig(SALIDAS / "fase2_perdida_multitarea.png", dpi=120)
plt.close(fig)
print(f"  -> Figura de pérdida multitarea guardada en: {SALIDAS / 'fase2_perdida_multitarea.png'}")

# FIGURA 3: Mapa de Distancia Euclidiana FÍSICA (EDT) en Milímetros (Fase 3)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0B132B")

# Subplot 1: Tomografía con regiones de interés
axes[0].imshow(imagenes_batch[idx_corte_edt, 0].cpu().numpy(), cmap="gray")
overlay_sel = np.zeros((*TARGET_SIZE, 4), dtype=np.float32)
overlay_sel[gt_sa] = [0.89, 0.34, 0.18, 0.7]  # Sacro
overlay_sel[gt_ri] = [0.09, 0.75, 0.73, 0.7]  # Coxal Der
axes[0].imshow(overlay_sel)
axes[0].set_title(f"Corte z={z_corte_val}: Regiones Analizadas\n(Sacro y Coxal Derecho)", color="white", fontsize=11, fontweight="bold")
axes[0].axis("off")

# Subplot 2: Mapa EDT en milímetros desde el Coxal Derecho hacia el Sacro
mapa_edt_mm = calcular_mapa_distancia_edt_mm(gt_ri, spacing_256_yx)
im1 = axes[1].imshow(mapa_edt_mm, cmap="inferno", vmin=0, vmax=60)
axes[1].set_title("Transformada de Distancia EDT FÍSICA\n(Milímetros reales desde Coxal Derecho)", color="white", fontsize=11, fontweight="bold")
axes[1].axis("off")
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label("Distancia Física (mm)", color="white")
cbar1.ax.tick_params(colors="white")

# Subplot 3: Medición métrica exacta y comparación de separación
axes[2].imshow(imagenes_batch[idx_corte_edt, 0].cpu().numpy(), cmap="gray")
axes[2].contour(gt_sa, colors=["#E4572E"], linewidths=1.8)
axes[2].contour(gt_ri, colors=["#17BEBB"], linewidths=1.8)
axes[2].contour(pred_sa, colors=["#FFD166"], linewidths=1.5, linestyles="dashed")
axes[2].contour(pred_ri, colors=["#00E5FF"], linewidths=1.5, linestyles="dashed")

# Dibujar vector métrico de separación mínima Ground Truth
if pt_a_gt is not None and pt_b_gt is not None:
    axes[2].plot([pt_a_gt[1], pt_b_gt[1]], [pt_a_gt[0], pt_b_gt[0]], color="#06D6A0", linewidth=2.5, marker="o", markersize=5, label=f"GT Mín: {dist_gt_mm:.2f} mm")

# Dibujar vector métrico de separación mínima Predicho
if pt_a_pred is not None and pt_b_pred is not None:
    axes[2].plot([pt_a_pred[1], pt_b_pred[1]], [pt_a_pred[0], pt_b_pred[0]], color="#FF0055", linewidth=2.0, linestyle="--", marker="x", markersize=6, label=f"Pred Mín: {dist_pred_mm:.2f} mm")

axes[2].set_title(f"Distancia 2D entre centros de píxeles (mm)\nGT: {dist_gt_mm:.2f} mm | Pred: {dist_pred_mm:.2f} mm (Δ={error_absoluto_mm:.2f} mm)",
                  color="white", fontsize=11, fontweight="bold")
axes[2].axis("off")
axes[2].legend(facecolor="#1C2541", edgecolor="#3A506B", labelcolor="white", loc="upper right")

fig.suptitle(
    f"FASE 3: MEDICIÓN MÉTRICA DE SEPARACIÓN EN MILÍMETROS (EDT FÍSICA) · CORTE z={z_corte_val}\n"
    f"Header Spacing: sy={spacing_256_yx[0]:.3f} mm/px, sx={spacing_256_yx[1]:.3f} mm/px · Sin unidades en píxeles",
    color="white", fontsize=12.5, fontweight="bold", y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(SALIDAS / "fase3_separacion_edt_mm.png", dpi=130)
plt.close(fig)
print(f"  -> Figura de medición EDT guardada en: {SALIDAS / 'fase3_separacion_edt_mm.png'}")

# Guardar checkpoint actualizado
torch.save(modelo.state_dict(), SALIDAS / "checkpoint_multitarea_semana10.pth")
print(f"  -> Checkpoint guardado en: {SALIDAS / 'checkpoint_multitarea_semana10.pth'}")

print("\n" + "=" * 80)
print("EJECUCIÓN DE FASES 1, 2 Y 3 COMPLETADA CON ÉXITO")
print("=" * 80)
