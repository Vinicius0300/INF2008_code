import torch.nn as nn
from torchvision.models import mobilenet_v2

# Não é exatamente a pose net, pois não consideramos os offsets.
class PoseNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        
        mobilenet = mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone = mobilenet.features
        
        # Última camada tem 1280 canais
        self.head_heatmap = nn.Sequential(
            nn.Conv2d(1280, num_keypoints, 1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)
        )

        self.head_roi = nn.Sequential(
            nn.Conv2d(1280, 1, 1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False)
        )
    
    def forward(self, x):
        f = self.backbone(x)
        heatmaps = self.head_heatmap(f)
        roi = self.head_roi(f)
        
        return roi, heatmaps