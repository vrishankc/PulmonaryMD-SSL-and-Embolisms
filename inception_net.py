import torch
import torch.nn as nn
from torchvision import models

class InceptionEncoder(nn.Module):
    def __init__(self):
        super(InceptionEncoder, self).__init__()
        self.encoder = models.inception_v3(weights = models.Inception_V3_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        return self.encoder(x)
