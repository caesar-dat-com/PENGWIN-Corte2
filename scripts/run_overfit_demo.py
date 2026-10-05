"""Entrenamiento de Overfit intencional sobre 4 cortes axiales REALES de tomografía pélvica (Caso 001).

Utiliza cortes clínicos auténticos y máscaras oficiales de PENGWIN:
- Corte z=134 (inferior/acetábulo): solo Coxal Izquierdo y Coxal Derecho (NO hay sacro).
- Corte z=220 (medio/sacro posterior): Sacro (posterior), Coxal Izquierdo y Coxal Derecho.
- Corte z=235 (medio-alto): Sacro (posterior), Coxal Izquierdo y Coxal Derecho.
- Corte z=250 (alto/alas ilíacas): Sacro, Coxal Izquierdo y Coxal Derecho.

Demuestra la correctitud de la arquitectura FundidoraPC + CBAM (kernel 9x9) + NMS propio
sin recurrir a ninguna manipulación sintética ni figuras artificiales.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Asegurar import de pengwin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.optim as optim

from pengwin import io, dataset
from pengwin.models.detector import PelvisDetector, decodificar_grid
from pengwin.models.loss import PelvisDetectionLoss
from pengwin.models.nms import nms_por_clase

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SALIDAS = Path("salidas")
SALIDAS.mkdir(exist_ok=True)

# 1. Cargar volumen y máscaras REALES de Caso 001
DIR_IMG = Path("data/raw/images")
DIR_LBL = Path("data/raw/labels")

print("Cargando Caso 001 real desde data/raw...")
caso = io.cargar_caso("001", DIR_IMG, DIR_LBL)
assert caso.etiqueta is not None, "El caso 001 debe contener la máscara de anotación"
print(f"Caso 001 cargado: {caso.ct.shape} vóxeles, espaciado={caso.spacing}")

# 2. Selección de los 4 cortes axiales reales
Z_INDICES = [134, 220, 235, 250]
B = len(Z_INDICES)
TARGET_SIZE = (256, 256)

imagenes_batch = torch.zeros((B, 3, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=torch.float32)
boxes_gt_list = []
clases_slice_list = []

for b_idx, z in enumerate(Z_INDICES):
    # Ventana ósea clínica: nivel=400 HU, ancho=1800 HU -> rango [-500, +1300] HU
    ct_slice = io.ventana_hu(caso.ct[z], nivel=400, ancho=1800)
    ct_256 = cv2.resize(ct_slice, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # Formato 3 canales para el backbone
    imagenes_batch[b_idx] = torch.from_numpy(np.repeat(ct_256[np.newaxis, :, :], 3, axis=0))

    # Extracción de cajas Ground Truth reales
    cajas = dataset.extraer_bboxes_region(caso.etiqueta[z], min_pixeles=15, normalizado=True)
    clase_presente = [0.0, 0.0, 0.0]  # SA, LI, RI
    for c in cajas:
        c_idx = c["clase_idx"]
        clase_presente[c_idx] = 1.0
        xmin, ymin, xmax, ymax = c["bbox"]
        boxes_gt_list.append([float(b_idx), float(c_idx), xmin, ymin, xmax, ymax])

    clases_slice_list.append(clase_presente)
    siglas = [c["sigla"] for c in cajas]
    print(f"  Muestra {b_idx} (z={z}): Regiones={siglas}")

boxes_gt = torch.tensor(boxes_gt_list, dtype=torch.float32)
clases_slice = torch.tensor(clases_slice_list, dtype=torch.float32)

# 3. Inicializar Modelo con Backbone FundidoraPC + CBAM (kernel 9x9)
print("\nInicializando PelvisDetector (FundidoraPC + CBAM kernel 9x9)...")
modelo = PelvisDetector(backbone_tipo="fundidora", in_channels=3, usar_cbam=True)
criterio = PelvisDetectionLoss(lambda_obj=2.0, lambda_box=5.0, lambda_cls=1.0, lambda_slice=1.0, gamma=0.5)
optimizador = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. Bucle de Entrenamiento (Overfit sobre cortes reales)
num_epocas = 100
historial = {"total": [], "obj": [], "box": [], "cls": []}

print(f"Iniciando entrenamiento de overfit sobre {B} cortes clínicos reales durante {num_epocas} épocas...")
modelo.train()

for epoca in range(1, num_epocas + 1):
    optimizador.zero_grad()
    predicciones = modelo(imagenes_batch)
    losses = criterio(predicciones, boxes_gt, clases_slice)

    losses["loss_total"].backward()
    optimizador.step()

    historial["total"].append(losses["loss_total"].item())
    historial["obj"].append(losses["loss_obj"].item())
    historial["box"].append(losses["loss_box"].item())
    historial["cls"].append(losses["loss_cls"].item())

    if epoca % 20 == 0 or epoca == 1:
        print(f"Época [{epoca:03d}/{num_epocas}] | Loss Total: {losses['loss_total'].item():.4f} "
              f"(Obj: {losses['loss_obj'].item():.4f}, Box: {losses['loss_box'].item():.4f}, Cls: {losses['loss_cls'].item():.4f})")

# 5. Guardar Curva de Pérdida
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0B132B")
ax.set_facecolor("#1C2541")
ax.plot(historial["total"], label="Pérdida Total Compuesta", color="#00E5FF", linewidth=2.2)
ax.plot(historial["obj"], label="Focal Loss Objetidad (γ=0.5)", color="#FFD166", linestyle="--")
ax.plot(historial["box"], label="Smooth L1 Bounding Boxes", color="#06D6A0", linestyle=":")
ax.plot(historial["cls"], label="Cross-Entropy Clases", color="#EF476F", linestyle="-.")

ax.set_title("CONVERGENCIA DE OVERFIT SOBRE CORTES CT REALES (CASO 001)", color="white", fontsize=12, fontweight="bold")
ax.set_xlabel("Época", color="white")
ax.set_ylabel("Pérdida (Loss)", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#3A506B")
ax.grid(True, linestyle="--", alpha=0.3, color="#3A506B")
ax.legend(facecolor="#1C2541", edgecolor="#3A506B", labelcolor="white")
plt.tight_layout()
fig.savefig(SALIDAS / "overfit_loss_curve.png", dpi=120)
plt.close(fig)
print("Curva de pérdida guardada en salidas/overfit_loss_curve.png")

# 6. Evaluación de Inferencia con NMS Propio
predicciones = modelo.inferir_boxes(imagenes_batch, conf_threshold=0.35, iou_threshold=0.45)

# 7. Graficar Comparativa Anatómica Real
fig, axes = plt.subplots(2, B, figsize=(18, 9), facecolor="#0B132B")
colores = {0: "#FFD166", 1: "#00E5FF", 2: "#EF476F"}
nombres = {0: "Sacro (SA)", 1: "Coxal Izq (LI)", 2: "Coxal Der (RI)"}
nombres_cortes = ["Corte Real z=134 (Inferior)", "Corte Real z=220 (Medio)", "Corte Real z=235 (Medio-Alto)", "Corte Real z=250 (Alto)"]

for j in range(B):
    # Fila 0: Ground Truth Real
    axes[0, j].imshow(imagenes_batch[j, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(f"Muestra {j} · GT Real ({nombres_cortes[j]})", color="#FFD166", fontsize=10.5, fontweight="bold")
    axes[0, j].axis("off")

    mascara_gt = boxes_gt[:, 0] == float(j)
    cajas_muestra_gt = boxes_gt[mascara_gt]
    for caja in cajas_muestra_gt:
        c_idx = int(caja[1].item())
        xm, ym, xM, yM = caja[2:].tolist()
        axes[0, j].add_patch(patches.Rectangle((xm*256, ym*256), (xM-xm)*256, (yM-ym)*256,
                             linewidth=2.2, edgecolor=colores[c_idx], facecolor="none"))
        axes[0, j].text(xm*256+3, ym*256+13, nombres[c_idx], color="white", fontsize=8,
                        fontweight="bold", bbox=dict(facecolor=colores[c_idx], edgecolor="none", pad=1.5, alpha=0.85))

    # Fila 1: Predicción del Modelo + NMS Propio
    axes[1, j].imshow(imagenes_batch[j, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, j].set_title(f"Muestra {j} · Predicción Modelo + NMS Propio", color="#00E5FF", fontsize=10.5, fontweight="bold")
    axes[1, j].axis("off")

    for b, s, c in zip(predicciones[j]["boxes"], predicciones[j]["scores"], predicciones[j]["clases"]):
        c_idx = int(c.item())
        xm, ym, xM, yM = b.tolist()
        axes[1, j].add_patch(patches.Rectangle((xm*256, ym*256), (xM-xm)*256, (yM-ym)*256,
                             linewidth=2.2, edgecolor=colores[c_idx], facecolor="none", linestyle="--"))
        axes[1, j].text(xm*256+3, ym*256+13, f"{nombres[c_idx]}: {s.item():.2f}", color="white", fontsize=8,
                        fontweight="bold", bbox=dict(facecolor=colores[c_idx], edgecolor="none", pad=1.5, alpha=0.85))

fig.suptitle("PRUEBA DE CORRECTITUD: DETECCIÓN SOBRE CORTES CT REALES DE PELVIS (CASO 001)\n"
             "Línea sólida = Ground Truth oficial · Línea punteada = Predicción del modelo tras NMS propio",
             color="white", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig(SALIDAS / "overfit_predicciones_boxes.png", dpi=130)
plt.close(fig)
print("Figura de predicciones guardada en salidas/overfit_predicciones_boxes.png")

# 8. Guardar Checkpoint
torch.save(modelo.state_dict(), SALIDAS / "checkpoint_overfit_semana9.pth")
print("Checkpoint guardado en salidas/checkpoint_overfit_semana9.pth")
