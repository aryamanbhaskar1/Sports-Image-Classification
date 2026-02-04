import torch
import torch.nn as nn
from torchvision import models

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # remove ImageNet classifier

        # Freeze backbone for baseline
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)  # [batch_size, 2048]

# Custom Classifier Head
class SportsClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = ResNetBackbone()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
