"""Ablaciones académicas: backbone natural opcional, tres cabezas propias nuevas."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .sesion2 import PelvisSesion2,DecoderConSkips
from .cbam import CBAM
from .detector import GridDetectionHead

class ResNetPropio(nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__();self.encoder=resnet18(weights=None)
        if pretrained_path is not None:
            self.encoder.load_state_dict(torch.load(pretrained_path,map_location='cpu',weights_only=True))
        self.encoder.fc=nn.Identity()
        self.cbam=CBAM(512,kernel_size=9)
        self.detection_head=GridDetectionHead(512)
        self.classification_head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Linear(64,3))
        self.segmentation_head=DecoderConSkips()
        self.segmentation_head.stem=nn.Sequential(nn.Conv2d(512,8,1,bias=False),nn.BatchNorm2d(8),nn.LeakyReLU(.1,inplace=True))
        self.segmentation_head.skips=nn.ModuleList([nn.Conv2d(c,8,1) for c in (256,128,64,3)])
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,x):
        e=self.encoder;h=(x-self.mean)/self.std
        h=e.maxpool(e.relu(e.bn1(e.conv1(h))))
        pyramid=[]
        for layer in (e.layer1,e.layer2,e.layer3,e.layer4):h=layer(h);pyramid.append(h)
        features=self.cbam(h)
        out={'grid':self.detection_head(F.interpolate(features,size=(16,16),mode='bilinear',align_corners=False)),
             'clases':self.classification_head(features),'features':features}
        out.update(self.segmentation_head(features,pyramid,x));return out

def crear_ablacion(name,pretrained_path=None):
    if name.startswith('fundidora'):
        model=PelvisSesion2()
        if name=='fundidora_sin_cbam':model.backbone.cbam=nn.Identity()
        return model
    return ResNetPropio(pretrained_path if name=='resnet_transfer' else None)
