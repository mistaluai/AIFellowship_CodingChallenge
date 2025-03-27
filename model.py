import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, fc_layer=None, num_classes=103, freeze_backbone=True) -> None:
        super(ResNet50, self).__init__()
        self.model = self._init_backbone(
            models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, progress=False),
            fc_layer,
            num_classes,
            freeze_backbone
        )

    def _init_backbone(self, backbone, fc_layer, num_classes, freeze_backbone) -> nn.Module:
        in_features = backbone.fc.in_features

        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False

        if fc_layer is None:
            fc_layer = [nn.Linear(in_features, num_classes)]

        backbone.fc = nn.Sequential(
            *fc_layer
        )
        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)