import torch
from torch import nn
from torchvision import models

class CustomDeepLabV3Plus(nn.Module):
    """Custom DeepLabV3+ model for semantic segmentation."""
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        # Load pre-trained DeepLabV3+ model
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        # Replace classifier head for custom number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        """Forward pass for input tensor x."""
        return self.model(x)