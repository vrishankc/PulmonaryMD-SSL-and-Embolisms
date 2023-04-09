from torchvision import models as models
import torch
import torch.nn as nn
from transformers import AutoModel, ViTConfig, ViTFeatureExtractor, ViTModel
from transformers.models.vit.modeling_vit import ViTEncoder 

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device("cuda")

class Encoder(nn.Module):
    def __init__(self, out_dimensions):
        super(Encoder, self).__init__()
        
        
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.proj_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        x += 1
        x /= 2
        inputs = feature_extractor(images=x, return_tensors="pt").to(device)
        outputs = self.encoder(**inputs)
        feature_vector = outputs.pooler_output
        self.proj_head = nn.Sequential(
            feature_vector,
            self.proj_head
        ).to(device)
        return self.proj_head(x)
