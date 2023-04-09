import torch
import torch.nn as nn
from torchvision import models

class SwinEncoder(nn.Module):
    def __init__(self):
        super(SwinEncoder, self).__init__()
        self.encoder = models.swin_v2_b(weights = models.Swin_V2_B_Weights)
        self.encoder.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.encoder(x)
