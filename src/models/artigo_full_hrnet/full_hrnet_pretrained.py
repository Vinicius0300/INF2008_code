import os
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
        self.num_channels = num_channels
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
                        nn.Conv2d(self.num_channels[j], self.num_channels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j < i: # Downsampling
                    conv_downsamples = []
                    for k in range(i-j):
                        if k == i-j-1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], self.num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(self.num_channels[i])))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(self.num_channels[j], self.num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(self.num_channels[j]),
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
    def __init__(self, num_keypoints=2, width=32, pretrained_path=None):
        super(FullHRNet_ImageNet, self).__init__()
        self.width = width

        # 1. Definição da Arquitetura (Stem, Layers, Transition, Stages)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))

        self.num_channels = [width, width * 2, width * 4]

        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, self.num_channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_channels[0]), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64, self.num_channels[1], 3, 2, 1, bias=False), nn.BatchNorm2d(self.num_channels[1]), nn.ReLU(True)),
            nn.Sequential(nn.Sequential(nn.Conv2d(64, 32, 3, 2, 1, bias=False), nn.ReLU(True)), # Ajuste de canal p/ transição
                          nn.Conv2d(32, self.num_channels[2], 3, 2, 1, bias=False), nn.BatchNorm2d(self.num_channels[2]), nn.ReLU(True))
        ])

        self.stage2 = HighResolutionModule(num_branches=3, blocks=BasicBlock,
                                           num_blocks=[4, 4, 4], num_channels=self.num_channels)

        self.full_res_head = nn.Sequential(
            nn.ConvTranspose2d(self.num_channels[0], 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, num_keypoints, kernel_size=1, stride=1)
        )

        if pretrained_path is not None:
            self._load_pretrained_internal(pretrained_path)

    def _load_pretrained_internal(self, weight_path):
        """Lógica de carregamento idêntica à anterior, mas interna ao __init__"""
        if not os.path.exists(weight_path):
            print(f"Arquivo de pesos não encontrado em: {weight_path}")
            return

        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_dict = self.state_dict()
        new_state_dict = {}

        for k, v in state_dict.items():
            if k == 'conv1.weight' and v.shape[1] == 3:
                v = v.mean(dim=1, keepdim=True)

            if k in model_dict and v.shape == model_dict[k].shape:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)
        print(f"HRNet {self.width} inicializada com {len(new_state_dict)} camadas da ImageNet.")

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Criar ramificações paralelas
        x_list = [trans(x) for trans in self.transition1]

        # Fusão HRNet
        x_list = self.stage2(x_list)

        # Pegamos o fluxo de maior resolução (112x112) e aplicamos o head para 448x448
        out = self.full_res_head(x_list[0])
        return None, out
