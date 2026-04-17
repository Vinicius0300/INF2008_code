import torch
import torch.nn as nn

# --- BLOCOS BÁSICOS ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    """Bloco Residual padrão usado na HRNet"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

# --- MÓDULO DE ALTA RESOLUÇÃO (MULTI-SCALE FUSION) ---

class HighResolutionModule(nn.Module):
    """Módulo que realiza a fusão entre diferentes resoluções"""
    def __init__(self, num_branches, blocks, num_blocks, num_channels):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_branches(self, num_branches, blocks, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, blocks, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        if self.num_branches == 1: return None
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i: # Upsampling
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j < i: # Downsampling
                    conv_downsamples = []
                    for k in range(i-j):
                        if k == i-j-1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i])))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
                else:
                    fuse_layer.append(None)
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

# --- ARQUITETURA FULL-HRNET COMPLETA ---

class FullHRNet_ImageNet(nn.Module):
    def __init__(self):
        super(FullHRNet_ImageNet, self).__init__()
        # Stem: 448x448 -> 112x112
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (Simplificado para este exemplo)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        
        # Canais das ramificações (ex: 32, 64, 128)
        self.num_channels = [32, 64, 128]
        
        # Camada de transição para criar as 3 resoluções
        self.transition = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)),
            nn.Sequential(nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1, bias=False), nn.ReLU(True)),
                          nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        ])

        # Stage de Alta Resolução
        self.high_res_module = HighResolutionModule(num_branches=3, blocks=BasicBlock, 
                                                    num_blocks=[4, 4, 4], num_channels=self.num_channels)

        # FULL RESOLUTION HEAD (448x448)
        # O artigo usa deconvoluções para voltar de 112x112 (Branch 0) para 448x448
        self.full_res_head = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 4, kernel_size=1, stride=1) # 4 Heatmaps (Hioide A/P, C2, C4)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        
        # Criar ramificações paralelas
        x_list = [trans(x) for trans in self.transition]
        
        # Fusão HRNet
        x_list = self.high_res_module(x_list)
        
        # Pegamos o fluxo de maior resolução (112x112) e aplicamos o head para 448x448
        out = self.full_res_head(x_list[0])
        return out

# Exemplo de uso:
# model = FullHRNet(cfg=None)
# input_tensor = torch.randn(1, 1, 448, 448)
# output = model(input_tensor) # [1, 4, 448, 448]