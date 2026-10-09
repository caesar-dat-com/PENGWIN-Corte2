"""Adaptación propia de las skips y supervisión profunda del notebook de Sesión 2.

No es U-Net++ completa: skips aditivas proyectadas a ocho canales, sin concat
anidada. Conserva FundidoraPC, CBAM9, grid y clasificación del proyecto.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .detector import PelvisDetector
from .segmenter import PelvisSegmentationHead


class DecoderConSkips(PelvisSegmentationHead):
    def __init__(self, supervision_profunda=True):
        super().__init__(in_channels=256, num_clases=4, latent_channels=8)
        self.supervision_profunda = supervision_profunda
        self.skips = nn.ModuleList([nn.Conv2d(c, 8, 1) for c in (128, 64, 32, 3)])
        self.auxiliares = nn.ModuleList([nn.Conv2d(8, 4, 1) for _ in range(3)])
        self.interiores = nn.Conv2d(8, 1, 1)

    def forward(self, features, pyramid, image):
        x = self.stem(features)
        aux = []
        for i, (up, skip, projection) in enumerate(zip(
                (self.up1, self.up2, self.up3, self.up4),
                (*reversed(pyramid[:3]), image), self.skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            # Suma en vez de concatenación: ninguna convolución decodificadora
            # recibe más de ocho canales, salvo las proyecciones del encoder.
            x = up.conv(x + projection(skip))
            if self.training and self.supervision_profunda and i < 3:
                aux.append(self.auxiliares[i](x))
        return {'mascaras': self.proyeccion_final(x),
                'bordes': self.bordes_fragmento(x)[:, 0],
                'interiores': self.interiores(x)[:, 0],
                'auxiliares': aux}


class PelvisSesion2(PelvisDetector):
    def __init__(self, supervision_profunda=True):
        super().__init__(backbone_tipo='fundidora', pretrained=False)
        self.segmentation_head = DecoderConSkips(supervision_profunda)

    def forward(self, x):
        h = x
        pyramid = []
        encoder = self.backbone.backbone
        for layer in (encoder.etapa1, encoder.etapa2, encoder.etapa3, encoder.etapa4):
            h = layer(h)
            pyramid.append(h)
        features = self.backbone.cbam(pyramid[-1])
        out = {'grid': self.detection_head(features),
               'clases': self.classification_head(features), 'features': features}
        out.update(self.segmentation_head(features, pyramid, x))
        return out
