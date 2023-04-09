import torch.nn as nn
import torch
from torchvision import models

class FasterRCNNEncoder(nn.Module):
    def __init__(self):
        super(FasterRCNNEncoder, self).__init__()
        self.encoder = models.detection.fasterrcnn_resnet50_fpn_v2(weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        self.encoder.roi_heads.box_predictor.bbox_pred = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.encoder(x)
