"""Módulos y arquitecturas de Deep Learning para PENGWIN."""
from .backbone import PelvisBackbone
from .cbam import CBAM, ChannelAttention, SpatialAttention
from .detector import PelvisDetector, GridDetectionHead, decodificar_grid
from .loss import PelvisDetectionLoss
from .nms import nms_por_clase
from .segmenter import PelvisSegmentationHead, DiceLoss, CombinedSegmentationLoss
