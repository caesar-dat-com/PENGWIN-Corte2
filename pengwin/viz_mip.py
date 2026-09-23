"""Visualizador 1 — Volumen crudo: MIP 3D rotatorio del hueso por umbral HU.

Preprocesamiento clásico, sin modelo: se toma el CT sin procesar, se deja
solo lo que supera el umbral de hueso (HU) y se proyecta la intensidad
máxima a lo largo del rayo. Girando el volumen alrededor del eje
cráneo-caudal y proyectando en cada ángulo se obtiene la vista 3D.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

AIRE_HU = -1024


def a_isotropico(ct: np.ndarray, spacing: tuple, mm: float = 2.0) -> np.ndarray:
    """Remuestrea a vóxel cúbico de `mm` para que la proyección no quede
    deformada (el eje z suele tener otro espaciado). Solo para visualizar."""
    factores = [s / mm for s in spacing]
    return ndimage.zoom(ct.astype(np.float32), factores, order=1, prefilter=False)


def solo_hueso(ct: np.ndarray, umbral_hu: float = 200) -> np.ndarray:
    return np.where(ct >= umbral_hu, ct, AIRE_HU).astype(np.float32)


def mip(vol: np.ndarray, eje: int) -> np.ndarray:
    return vol.max(axis=eje)


def mip_rotatorio(vol: np.ndarray, angulos) -> list[np.ndarray]:
    """Vol (z, y, x) en LPS. Gira en el plano axial (y, x) y proyecta en y
    (antero-posterior) -> vista coronal desde cada ángulo."""
    vistas = []
    for a in angulos:
        rot = ndimage.rotate(vol, a, axes=(1, 2), reshape=False, order=1,
                             mode="constant", cval=AIRE_HU, prefilter=False)
        vistas.append(rot.max(axis=1)[::-1])        # z hacia arriba = craneal
    return vistas


def figura_mip(ct: np.ndarray, spacing: tuple, id_caso: str, umbral_hu: float = 200,
               mm: float = 2.0, paso_grados: int = 10, rango_hu=(200, 1500)) -> go.Figure:
    vol = solo_hueso(a_isotropico(ct, spacing, mm), umbral_hu)
    angulos = list(range(0, 360, paso_grados))
    vistas = mip_rotatorio(vol, angulos)

    def heat(z):
        return go.Heatmap(z=z, colorscale="Gray", zmin=rango_hu[0], zmax=rango_hu[1],
                          colorbar=dict(title="HU"), hovertemplate="HU %{z:.0f}<extra></extra>")

    fig = go.Figure(data=[heat(vistas[0])],
                    frames=[go.Frame(data=[heat(v)], name=str(a)) for a, v in zip(angulos, vistas)])
    pasos = [dict(method="animate", label=f"{a}°",
                  args=[[str(a)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for a in angulos]
    fig.update_layout(
        title=(f"Caso {id_caso} · MIP del hueso (HU ≥ {umbral_hu}) · vóxel {mm} mm"
               "<br><sup>Solo uso académico. No es un dispositivo médico ni apoya decisiones quirúrgicas.</sup>"),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="black", height=720, margin=dict(t=90, l=10, r=10, b=10),
        sliders=[dict(active=0, steps=pasos, currentvalue=dict(prefix="Rotación: "), pad=dict(t=30))],
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=0, xanchor="left", yanchor="top",
                          pad=dict(t=60), buttons=[
                              dict(label="▶ Girar", method="animate",
                                   args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
                              dict(label="■", method="animate",
                                   args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
    )
    return fig


def figura_tres_vistas(ct: np.ndarray, spacing: tuple, id_caso: str, umbral_hu: float = 200,
                       mm: float = 2.0):
    """MIP axial, coronal y sagital estáticos (para el informe / matplotlib)."""
    import matplotlib.pyplot as plt
    vol = solo_hueso(a_isotropico(ct, spacing, mm), umbral_hu)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="black")
    for a, (eje, nombre, flip) in zip(ax, [(0, "Axial", False), (1, "Coronal", True), (2, "Sagital", True)]):
        m = mip(vol, eje)
        a.imshow(m[::-1] if flip else m, cmap="gray", vmin=200, vmax=1500)
        a.set_title(nombre, color="white")
        a.axis("off")
    fig.suptitle(f"Caso {id_caso} · MIP hueso HU ≥ {umbral_hu} — uso académico, no clínico",
                 color="white")
    fig.tight_layout()
    return fig
