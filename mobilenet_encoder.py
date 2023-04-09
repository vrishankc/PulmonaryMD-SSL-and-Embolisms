import torch
import torch.nn as nn
from torchvision import models

class MobileNetEncoder(nn.Module):
    def __init__(self):
        super(MobileNetEncoder, self).__init__()
        self.encoder = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        num_ftrs = self.encoder.classifier[3].in_features
        self.encoder.classifier[3] = nn.Linear(num_ftrs, 128) # generates 128-dim representation vector

        self.proj_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.proj_head(self.encoder(x))
    

