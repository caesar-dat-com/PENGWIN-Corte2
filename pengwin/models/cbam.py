"""Módulo de Atención Convolucional por Bloques (CBAM - Convolutional Block Attention Module).

Referencia: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
Requerimiento oficial de la sección 4.1 del documento de proyecto:
"Bloque de atención CBAM: deberá incorporarse atención de canal y espacial en el backbone,
antes de la bifurcación hacia las tres cabezas."
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Módulo de Atención de Canal (CAM).

    Aprende 'qué' características anatómicas son relevantes combinando
    agregación promedio y máxima a lo largo de las dimensiones espaciales,
    procesadas por un perceptrón multicapa (MLP) compartido con factor de reducción.
    """

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_planes = max(in_planes // ratio, 4)

        # MLP compartido implementado mediante convoluciones 1x1
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """Módulo de Atención Espacial (SAM).

    Aprende 'dónde' enfocar la atención espacialmente en el corte pélvico.
    Aplica pool de canal promedio y máximo, los concatena y utiliza una
    convolución 7x7 para generar el mapa de atención 2D.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "El tamaño de kernel debe ser 3 o 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(combined))
        return x * spatial_weights


class CBAM(nn.Module):
    """Bloque de atención dual compuesto: Canal + Espacio.

    Aplica primero atención de canal y sobre la salida aplica atención espacial.
    Opcionalmente suma la conexión residual x.
    """

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7, residual: bool = True):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(x)
        out = self.sa(out)
        if self.residual:
            return x + out
        return out
