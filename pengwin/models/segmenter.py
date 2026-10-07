"""Módulo de Segmentación de Instancias y Regiones Pélvicas (Semana 10 - Fase 1).

Implementa la 3ª cabeza de la arquitectura compartida sobre el backbone FundidoraPC + CBAM:
1. Reconstrucción ligera en no más de 10 canales latentes (por defecto 8 canales)
   para satisfacer la restricción de memoria y cómputo establecida por el docente.
2. Bloques de Doble Convolución (DoubleConv) en cada etapa de upsampling para refinamiento
   de fronteras anatómicas.
3. Conexiones de subida progresiva (16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256).
4. Emisión de mapas de segmentación a resolución completa 256x256.
5. Funciones de pérdida específicas para segmentación médica (Dice Loss + Cross-Entropy).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Bloque de doble convolución clásico (Conv2D -> BatchNorm -> LeakyReLU -> Conv2D -> BatchNorm -> LeakyReLU)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bloque de subida de resolución (Upsampling x2 + Doble Convolución)."""

    def __init__(self, in_channels: int, out_channels: int, modo: str = "bilinear"):
        super().__init__()
        self.modo = modo
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Interpolación bilinear para escalado suave x2
        x_up = F.interpolate(x, scale_factor=2.0, mode=self.modo, align_corners=False)
        return self.conv(x_up)


class PelvisSegmentationHead(nn.Module):
    """Cabeza de segmentación ligera para regiones y fragmentos pélvicos.

    Restricción del docente estrictamente cumplida:
    - 'Reconstruir en no más de 10 canales': la rama decodificadora opera
      con exactamente 8 canales latentes (latent_channels=8 <= 10).
    - Doble convolución en cada escala de resolución espacial.
    - Entrada: mapa latente (B, in_channels=256, H=16, W=16).
    - Salida: mapa de logits (B, num_clases=4, H=256, W=256) [0: fondo, 1: SA, 2: LI, 3: RI].
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_clases: int = 4,
        latent_channels: int = 8,
    ):
        super().__init__()
        assert latent_channels <= 10, f"Los canales de reconstrucción no deben superar 10 (se recibieron {latent_channels})"
        self.num_clases = num_clases
        self.latent_channels = latent_channels

        # Compresión inicial de canales (256 -> 8) antes de la reconstrucción espacial
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 4 etapas de reconstrucción espacial progresiva (16 -> 32 -> 64 -> 128 -> 256)
        # Todas las etapas mantienen latent_channels (8 canales) para optimizar memoria
        self.up1 = UpBlock(latent_channels, latent_channels)  # 16x16 -> 32x32
        self.up2 = UpBlock(latent_channels, latent_channels)  # 32x32 -> 64x64
        self.up3 = UpBlock(latent_channels, latent_channels)  # 64x64 -> 128x128
        self.up4 = UpBlock(latent_channels, latent_channels)  # 128x128 -> 256x256

        # Proyección final al número de clases anatómicas
        self.proyeccion_final = nn.Conv2d(latent_channels, num_clases, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 256, 16, 16)
        x = self.stem(features)     # (B, 8, 16, 16)
        x = self.up1(x)             # (B, 8, 32, 32)
        x = self.up2(x)             # (B, 8, 64, 64)
        x = self.up3(x)             # (B, 8, 128, 128)
        x = self.up4(x)             # (B, 8, 256, 256)
        logits = self.proyeccion_final(x)  # (B, num_clases, 256, 256)
        return logits


class DiceLoss(nn.Module):
    """Pérdida de similitud Dice multiclase con suavizado Laplace."""

    def __init__(self, smooth: float = 1e-5, ignore_index: int | None = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        # One-hot encoding de targets
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dice_total = 0.0
        clases_evaluadas = 0

        # Evaluar clases anatómicas (priorizar clases 1..C-1 sobre fondo)
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            p = probs[:, c].contiguous().view(-1)
            t = targets_one_hot[:, c].contiguous().view(-1)

            interseccion = (p * t).sum()
            union = p.sum() + t.sum()

            dice_c = (2.0 * interseccion + self.smooth) / (union + self.smooth)
            dice_total += dice_c
            clases_evaluadas += 1

        return 1.0 - (dice_total / max(clases_evaluadas, 1))


class CombinedSegmentationLoss(nn.Module):
    """Pérdida combinada: Cross-Entropy Ponderada + Dice Loss para mitigar severo desbalance de fondo."""

    def __init__(self, lambda_ce: float = 1.0, lambda_dice: float = 1.5, weight: torch.Tensor | None = None):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.ce_fn = nn.CrossEntropyLoss(weight=weight)
        self.dice_fn = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_ce = self.ce_fn(logits, targets)
        loss_dice = self.dice_fn(logits, targets)
        loss_total = self.lambda_ce * loss_ce + self.lambda_dice * loss_dice
        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
        }
