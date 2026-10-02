"""Experimento de Overfit intencional sobre un batch pequeño (Prueba de correctitud Semana 9).

Requerimiento oficial de la sección 6:
"Overfit intencional sobre un batch pequeño como prueba de correctitud;
primeras predicciones de bounding boxes visualmente razonables."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.optim as optim

from pengwin.models.detector import PelvisDetector
from pengwin.models.loss import PelvisDetectionLoss

# Configuración y semilla
torch.manual_seed(42)
np.random.seed(42)

SALIDAS = Path("salidas")
SALIDAS.mkdir(exist_ok=True)

# 1. Crear batch pequeño de 4 cortes (simulando casos axiales con regiones óseas)
# Batch shape: (4, 3, 256, 256)
B = 4
imagenes = torch.zeros((B, 3, 256, 256), dtype=torch.float32)

# Añadimos fondo elíptico (simulando cuerpo del paciente)
for i in range(B):
    y, x = np.ogrid[:256, :256]
    mascara_cuerpo = ((x - 128) ** 2 / 100 ** 2 + (y - 128) ** 2 / 70 ** 2) <= 1.0
    imagenes[i, :, mascara_cuerpo] = 0.25

# Definimos cajas Ground Truth realistas para las 3 regiones en los 4 cortes:
# [batch_idx, class_idx (0: SA, 1: LI, 2: RI), xmin, ymin, xmax, ymax]
boxes_gt_list = [
    # Muestra 0: Sacro (centro) y Coxal Derecho (izquierda anatomica)
    [0.0, 0.0, 0.42, 0.25, 0.58, 0.45],  # Sacro
    [0.0, 2.0, 0.18, 0.40, 0.38, 0.70],  # Coxal Der
    # Muestra 1: Sacro, Coxal Izquierdo y Coxal Derecho (fractura bilateral)
    [1.0, 0.0, 0.40, 0.28, 0.60, 0.46],  # Sacro
    [1.0, 1.0, 0.62, 0.38, 0.82, 0.72],  # Coxal Izq
    [1.0, 2.0, 0.18, 0.38, 0.38, 0.72],  # Coxal Der
    # Muestra 2: Coxal Izquierdo y Coxal Derecho
    [2.0, 1.0, 0.60, 0.35, 0.85, 0.75],  # Coxal Izq
    [2.0, 2.0, 0.15, 0.35, 0.40, 0.75],  # Coxal Der
    # Muestra 3: Solo Sacro
    [3.0, 0.0, 0.38, 0.22, 0.62, 0.48],  # Sacro
]
boxes_gt = torch.tensor(boxes_gt_list, dtype=torch.float32)

# Pintamos las regiones en las imágenes para que tengan textura ósea brillante (0.8 - 1.0)
for caja in boxes_gt_list:
    b_idx = int(caja[0])
    xmin, ymin, xmax, ymax = int(caja[2] * 256), int(caja[3] * 256), int(caja[4] * 256), int(caja[5] * 256)
    imagenes[b_idx, :, ymin:ymax, xmin:xmax] = 0.85

clases_slice = torch.tensor([
    [1.0, 0.0, 1.0],  # Muestra 0: SA, RI
    [1.0, 1.0, 1.0],  # Muestra 1: SA, LI, RI
    [0.0, 1.0, 1.0],  # Muestra 2: LI, RI
    [1.0, 0.0, 0.0],  # Muestra 3: SA
], dtype=torch.float32)

# 2. Inicializar Modelo con Backbone FundidoraPC + CBAM
print("Inicializando PelvisDetector (FundidoraPC + CBAM)...")
modelo = PelvisDetector(backbone_tipo="fundidora", in_channels=3, usar_cbam=True)
criterio = PelvisDetectionLoss(lambda_obj=2.0, lambda_box=5.0, lambda_cls=1.0, lambda_slice=1.0, gamma=0.5)
optimizador = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)

# 3. Bucle de Entrenamiento (Overfit intentional)
num_epocas = 100
historial = {"total": [], "obj": [], "box": [], "cls": []}

print(f"Iniciando entrenamiento de overfit sobre {B} muestras durante {num_epocas} épocas...")
modelo.train()

for epoca in range(1, num_epocas + 1):
    optimizador.zero_grad()
    salida = modelo(imagenes)
    perdidas = criterio(salida, boxes_gt, clases_slice)

    loss_total = perdidas["loss_total"]
    loss_total.backward()
    optimizador.step()

    historial["total"].append(loss_total.item())
    historial["obj"].append(perdidas["loss_obj"].item())
    historial["box"].append(perdidas["loss_box"].item())
    historial["cls"].append(perdidas["loss_cls"].item())

    if epoca % 20 == 0 or epoca == 1:
        print(f"Época [{epoca:03d}/{num_epocas}] | Loss Total: {loss_total.item():.4f} "
              f"(Obj: {perdidas['loss_obj'].item():.4f}, Box: {perdidas['loss_box'].item():.4f})")

# 4. Guardar métricas y gráfica de la curva de convergencia
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="#141414")
for a in ax:
    a.set_facecolor("#1f1f1f")
    a.tick_params(colors="white")
    for spine in a.spines.values():
        spine.set_color("#444444")

ax[0].plot(historial["total"], color="#00E5FF", lw=2.2, label="Pérdida Total Compuesta")
ax[0].plot(historial["box"], color="#FFD600", lw=1.8, linestyle="--", label="Pérdida Regresión Cajas (Smooth L1)")
ax[0].set_title("Convergencia de Pérdida en Overfit (Semana 9)", color="white", fontsize=12, pad=10)
ax[0].set_xlabel("Épocas", color="white")
ax[0].set_ylabel("Pérdida", color="white")
ax[0].legend(facecolor="#262626", edgecolor="none", labelcolor="white")
ax[0].grid(True, color="#333333", linestyle=":")

ax[1].plot(historial["obj"], color="#FF4081", lw=2, label="Pérdida Objetidad (Focal gamma=0.5)")
ax[1].plot(historial["cls"], color="#76FF03", lw=2, label="Pérdida Clasificación Región")
ax[1].set_title("Desglose de Pérdidas de Detección", color="white", fontsize=12, pad=10)
ax[1].set_xlabel("Épocas", color="white")
ax[1].legend(facecolor="#262626", edgecolor="none", labelcolor="white")
ax[1].grid(True, color="#333333", linestyle=":")

fig.tight_layout()
fig.savefig(SALIDAS / "overfit_loss_curve.png", dpi=130, facecolor=fig.get_facecolor())
print("Curva de pérdida guardada en salidas/overfit_loss_curve.png")

# 5. Inferencia con NMS y visualización de predicciones vs Ground Truth
print("Evaluando inferencia con NMS propio...")
predicciones = modelo.inferir_boxes(imagenes, conf_threshold=0.25, iou_threshold=0.4)

colores = {0: "#E4572E", 1: "#4A90E2", 2: "#17BEBB"}
nombres = {0: "Sacro (SA)", 1: "Coxal Izq (LI)", 2: "Coxal Der (RI)"}

fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="#111111")
for j in range(4):
    # Fila superior: Ground Truth
    ax_gt = axes[0, j]
    ax_gt.imshow(imagenes[j, 0].numpy(), cmap="bone")
    ax_gt.set_title(f"Muestra {j} · Ground Truth", color="white", fontsize=11, pad=8)
    ax_gt.axis("off")

    cajas_m = boxes_gt[boxes_gt[:, 0] == j]
    for c in cajas_m:
        c_idx = int(c[1].item())
        xm, ym, xM, yM = c[2:].tolist()
        rect = patches.Rectangle((xm * 256, ym * 256), (xM - xm) * 256, (yM - ym) * 256,
                                 linewidth=2.2, edgecolor=colores[c_idx], facecolor=colores[c_idx], alpha=0.25)
        ax_gt.add_patch(rect)
        rect_b = patches.Rectangle((xm * 256, ym * 256), (xM - xm) * 256, (yM - ym) * 256,
                                   linewidth=2.2, edgecolor=colores[c_idx], facecolor="none")
        ax_gt.add_patch(rect_b)
        ax_gt.text(xm * 256 + 4, ym * 256 + 14, nombres[c_idx], color="white", fontsize=8,
                   fontweight="bold", bbox=dict(facecolor=colores[c_idx], edgecolor="none", pad=1.5))

    # Fila inferior: Predicciones del modelo tras NMS
    ax_pred = axes[1, j]
    ax_pred.imshow(imagenes[j, 0].numpy(), cmap="bone")
    ax_pred.set_title(f"Muestra {j} · Predicción (Modelo + NMS)", color="#00E5FF", fontsize=11, pad=8)
    ax_pred.axis("off")

    preds_m = predicciones[j]
    for b, s, c in zip(preds_m["boxes"], preds_m["scores"], preds_m["clases"]):
        c_idx = int(c.item())
        xm, ym, xM, yM = b.tolist()
        rect = patches.Rectangle((xm * 256, ym * 256), (xM - xm) * 256, (yM - ym) * 256,
                                 linewidth=2.2, edgecolor=colores[c_idx], facecolor="none", linestyle="--")
        ax_pred.add_patch(rect)
        ax_pred.text(xm * 256 + 4, ym * 256 + 14, f"{nombres[c_idx]}: {s.item():.2f}",
                     color="white", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor=colores[c_idx], edgecolor="none", pad=1.5))

fig.suptitle("PRUEBA DE CORRECTITUD (SEMANA 9): OVERFIT INTENCIONAL Y DETECCIÓN NMS", color="white", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig(SALIDAS / "overfit_predicciones_boxes.png", dpi=130, facecolor=fig.get_facecolor())
print("Figura de predicciones guardada en salidas/overfit_predicciones_boxes.png")

# Guardar checkpoint para reproducibilidad
torch.save(modelo.state_dict(), SALIDAS / "checkpoint_overfit_semana9.pth")
print("Checkpoint guardado en salidas/checkpoint_overfit_semana9.pth")
