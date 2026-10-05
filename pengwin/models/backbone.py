"""Backbone compartido con módulo de atención CBAM integrado.

Requisitos de la sección 4.1 del documento de proyecto:
- "Backbone compartido: extensión propia del backbone del curso (FundidoraPC) o equivalente."
- "Bloque de atención CBAM incorporado antes de la bifurcación hacia las tres cabezas."
- Soporte para transfer learning (e.g. pesos ImageNet en backbone) o entrenamiento desde cero.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .cbam import CBAM


class ConvBlock(nn.Module):
    """Bloque convolucional base con Normalización por Lote y activación LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Bloque residual con dos convoluciones y conexión shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3),
            ConvBlock(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FundidoraPC(nn.Module):
    """Arquitectura FundidoraPC (Backbone convolucional propio del curso).

    Procesa tensores 2D (B, C, H, W) a través de 4 etapas progresivas:
    Etapa 1: 3 -> 32 canales (downsample x2)
    Etapa 2: 32 -> 64 canales (downsample x2)
    Etapa 3: 64 -> 128 canales (downsample x2)
    Etapa 4: 128 -> 256 canales (downsample x2)
    Salida espacial: para entrada 256x256 -> mapa de características de 16x16 (o 32x32 si x8).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        self.etapa1 = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),  # 256 -> 128
            ResidualBlock(32),
        )
        self.etapa2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),           # 128 -> 64
            ResidualBlock(64),
        )
        self.etapa3 = nn.Sequential(
            ConvBlock(64, 128, stride=2),          # 64 -> 32
            ResidualBlock(128),
        )
        self.etapa4 = nn.Sequential(
            ConvBlock(128, out_channels, stride=2), # 32 -> 16
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.etapa1(x)
        x = self.etapa2(x)
        x = self.etapa3(x)
        x = self.etapa4(x)
        return x


class PelvisBackbone(nn.Module):
    """Backbone unificado con Bloque CBAM integrado al final.

    Permite seleccionar entre:
    - 'fundidora': Arquitectura FundidoraPC diseñada desde cero.
    - 'resnet18': ResNet-18 con pesos preentrenados (Transfer Learning).
    """

    def __init__(
        self,
        tipo: str = "fundidora",
        in_channels: int = 3,
        usar_cbam: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.tipo = tipo
        self.usar_cbam = usar_cbam

        if tipo == "fundidora":
            self.backbone = FundidoraPC(in_channels=in_channels, out_channels=256)
            self.out_channels = 256
        elif tipo == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            rn = resnet18(weights=weights)
            # Adaptar primera capa si in_channels != 3
            if in_channels != 3:
                rn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Tomamos hasta layer4 (sin avgpool ni fc) -> salida 512 canales, reducción x32 (8x8)
            self.backbone = nn.Sequential(
                rn.conv1, rn.bn1, rn.relu, rn.maxpool,
                rn.layer1, rn.layer2, rn.layer3, rn.layer4
            )
            self.out_channels = 512
        else:
            raise ValueError(f"Tipo de backbone no soportado: {tipo}")

        # Bloque de atención CBAM aplicado a la salida del backbone (kernel 9x9 para pelvis)
        if self.usar_cbam:
            self.cbam = CBAM(in_planes=self.out_channels, ratio=16, kernel_size=9)
        else:
            self.cbam = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features_atendidas = self.cbam(features)
        return features_atendidas
