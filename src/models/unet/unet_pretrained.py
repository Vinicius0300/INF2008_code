import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p  # O "down" é o valor que será passado pela skip connection

    
class UpSample(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels) # Muda Aqui

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
    
class UNet_ImageNet(nn.Module):
    def __init__(self, num_keypoints=2, in_channels=1):
        super().__init__()
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        
        # 1. Ajuste do Input para Grayscale
        if in_channels != 3:
            old_weight = base_model.conv1.weight.data
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            base_model.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)


        # 2. Encoder (ResNet18)
        self.enc1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu) # 64
        self.maxpool = base_model.maxpool 
        self.layer1 = base_model.layer1 # 64
        self.layer2 = base_model.layer2 # 128
        self.layer3 = base_model.layer3 # 256
        self.layer4 = base_model.layer4 # 512


        # 3. Bridge (BottleNeck)
        # Recebe 512 da layer4 e sobe para 1024
        self.bottle_neck = DoubleConv(512, 1024)


        # 4. Decoder (Canais Corrigidos para Concatenação)
        # UpSample(In_Canais, Out_Canais) 
        # Up 1: Recebe 1024, reduz p/ 512, concatena com e4 (256). Saída: 512.
        self.up_convolution_1 = UpSample(1024, 256, 512) 

        # Up 2: Recebe 512, reduz p/ 256, concatena com e3 (128). Saída: 256.
        self.up_convolution_2 = UpSample(512, 128, 256)

        # Up 3: Recebe 256, reduz p/ 128, concatena com e2 (64). Saída: 128.
        self.up_convolution_3 = UpSample(256, 64, 128)

        # Up 4: Recebe 128, reduz p/ 64, concatena com e1 (64). Saída: 64.
        self.up_convolution_4 = UpSample(128, 64, 64)

        
        self.head_roi = nn.Conv2d(64, 1, kernel_size=1)
        self.head_kp = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # 64 canais, res 1/2
        p1 = self.maxpool(e1)  # 64 canais, res 1/4
        
        e2 = self.layer1(p1)   # 64 canais, res 1/4
        e3 = self.layer2(e2)   # 128 canais, res 1/8
        e4 = self.layer3(e3)   # 256 canais, res 1/16
        e5 = self.layer4(e4)   # 512 canais, res 1/32

        # Bridge
        b = self.bottle_neck(e5) # 1024 canais

        # Decoder com Skip Connections corretas
        u1 = self.up_convolution_1(b, e4)  # 1024 -> 512 + 256 (e4)
        u2 = self.up_convolution_2(u1, e3) # 512 -> 256 + 128 (e3)
        u3 = self.up_convolution_3(u2, e2) # 256 -> 128 + 64 (e2)
        u4 = self.up_convolution_4(u3, e1) # 128 -> 64 + 64 (e1)

        # Voltando para a resolução original (256x256)
        out = F.interpolate(u4, scale_factor=2, mode='bilinear', align_corners=True)

        return self.head_roi(out), self.head_kp(out)