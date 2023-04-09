import torch.nn as nn
import torchvision.models as models
import torch
device = "cuda" if torch.cuda.is_available else "cpu"
class ResnetPulmonaryMD(nn.Module):

    def __init__(self, out_dimensions):
        super(ResnetPulmonaryMD, self).__init__()
        #self.mobilenet_model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, num_classes = out_dimensions)
        self.resnet_model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2, num_classes = out_dimensions) 
        
        # mlp = nn.Linear(512 * 4, 512)
        mlp = self.resnet_model.fc.in_features # 2048
        #self.mobilenet_model = self.mobilenet_model.to(device)
        self.resnet_model.fc = nn.Sequential(
            nn.Linear(mlp, mlp),  
            nn.ReLU(inplace=True),
            self.resnet_model.fc, 
        )
        


    def forward(self, x):
        #return self.mobilenet_model(x)
        return self.resnet_model(x)
