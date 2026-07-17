import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# NORMALIZAÇÃO: SWITCHABLE NORMALIZATION
# ============================================================

class SwitchNorm2d(nn.Module):
    """Combina Batch Norm, Layer Norm e Instance Norm via média ponderada
    aprendível (pesos passam por softmax). Substitui a Batch Normalization
    padrão em todos os blocos residuais da rede, conforme especificado."""
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        # pesos de mistura entre IN / LN / BN (na ordem: instance, layer, batch)
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()

        # ---- Instance Norm (por amostra, por canal) ----
        mean_in = x.mean(dim=[2, 3], keepdim=True)
        var_in = x.var(dim=[2, 3], keepdim=True, unbiased=False)

        # ---- Layer Norm (por amostra, entre todos os canais) ----
        mean_ln = mean_in.mean(dim=1, keepdim=True)
        var_ln = var_in.mean(dim=1, keepdim=True) + mean_in.var(dim=1, keepdim=True, unbiased=False)

        # ---- Batch Norm (entre amostras do lote) ----
        if self.training:
            mean_bn = mean_in.mean(dim=0, keepdim=True)
            var_bn = var_in.mean(dim=0, keepdim=True) + mean_in.var(dim=0, keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * mean_bn)
                self.running_var.mul_(self.momentum).add_((1 - self.momentum) * var_bn)
        else:
            mean_bn = self.running_mean
            var_bn = self.running_var

        mean_weight = F.softmax(self.mean_weight, dim=0)
        var_weight = F.softmax(self.var_weight, dim=0)

        mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
        var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ============================================================
# BLOCOS BÁSICOS (com Switchable Normalization no lugar da BatchNorm)
# ============================================================

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockSN(nn.Module):
    """Bloco residual padrão (usado no ResNet-34 / Estágio 2)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockSN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.sn1 = SwitchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.sn2 = SwitchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.sn1(self.conv1(x)))
        out = self.sn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class BottleneckSN(nn.Module):
    """Bloco residual bottleneck (usado no ResNet-50 / Estágio 1)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckSN, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.sn1 = SwitchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.sn2 = SwitchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.sn3 = SwitchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.sn1(self.conv1(x)))
        out = self.relu(self.sn2(self.conv2(out)))
        out = self.sn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


def _make_layer(block, inplanes, planes, num_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            SwitchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers), inplanes


# ============================================================
# ESTÁGIO 1: REDE DE DETECÇÃO GLOBAL (ResNet-50) -> ROI
# ============================================================

class GlobalStageResNet50(nn.Module):
    """Detecção grosseira da região das vértebras (treinada originalmente só
    com rótulos de C2 e C4). Saída: 4 valores normalizados (x1, y1, x2, y2)
    representando os cantos da caixa delimitadora, em [0, 1]."""
    def __init__(self, in_channels=1):
        super(GlobalStageResNet50, self).__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            SwitchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1, self.inplanes = _make_layer(BottleneckSN, self.inplanes, 64, 3, stride=1)
        self.layer2, self.inplanes = _make_layer(BottleneckSN, self.inplanes, 128, 4, stride=2)
        self.layer3, self.inplanes = _make_layer(BottleneckSN, self.inplanes, 256, 6, stride=2)
        self.layer4, self.inplanes = _make_layer(BottleneckSN, self.inplanes, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # camada totalmente conectada ajustada para prever a região (bbox), não classes
        self.fc_roi = nn.Linear(512 * BottleneckSN.expansion, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        box = torch.sigmoid(self.fc_roi(x))  # (N, 4) -> x1, y1, x2, y2 em [0, 1]
        return box


# ============================================================
# ESTÁGIO 2: REDE DE DETECÇÃO LOCAL (ResNet-34) -> Pontos
# ============================================================

class LocalStageResNet34(nn.Module):
    """Localização fina dos landmarks a partir do recorte da ROI.
    Saída: coordenadas normalizadas [-1, 1] (convenção grid_sample) dos
    pontos, relativas ao recorte."""
    def __init__(self, in_channels=1, num_points=2):
        super(LocalStageResNet34, self).__init__()
        self.inplanes = 64
        self.num_points = num_points

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            SwitchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1, self.inplanes = _make_layer(BasicBlockSN, self.inplanes, 64, 3, stride=1)
        self.layer2, self.inplanes = _make_layer(BasicBlockSN, self.inplanes, 128, 4, stride=2)
        self.layer3, self.inplanes = _make_layer(BasicBlockSN, self.inplanes, 256, 6, stride=2)
        self.layer4, self.inplanes = _make_layer(BasicBlockSN, self.inplanes, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_points = nn.Linear(512 * BasicBlockSN.expansion, num_points * 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        pts = torch.tanh(self.fc_points(x))  # (N, num_points*2) em [-1, 1]
        return pts.view(-1, self.num_points, 2)


# ============================================================
# ARQUITETURA COMPLETA: PIPELINE DE DOIS ESTÁGIOS
# ============================================================

class TwoStageResNet(nn.Module):
    """Pipeline completo: ResNet-50 (ROI grosseira) -> recorte diferenciável
    -> ResNet-34 (refinamento dos pontos C2 e C4).

    Retorno padronizado (roi, heatmap, points):
        roi     -> máscara de segmentação da ROI (N, 1, H, W)
        heatmap -> None (esta arquitetura não produz heatmaps)
        points  -> coordenadas (N, num_points, 2) em pixels, na imagem original
    """
    def __init__(self, in_channels=1, image_size=448, crop_size=224,
                 num_points=2, mask_sharpness=25.0):
        super(TwoStageResNet, self).__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.num_points = num_points
        self.mask_sharpness = mask_sharpness

        self.global_stage = GlobalStageResNet50(in_channels=in_channels)
        self.local_stage = LocalStageResNet34(in_channels=in_channels, num_points=num_points)

        self._init_weights()

    def _init_weights(self):
        """Inicialização Xavier para todas as camadas convolucionais e
        totalmente conectadas, conforme especificado."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _box_to_theta(self, box):
        """Converte a caixa (x1, y1, x2, y2) em [0, 1] para a matriz afim
        (N, 2, 3) usada por affine_grid/grid_sample (convenção [-1, 1])."""
        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        # garante x2 > x1 e y2 > y1 (caixa minimamente válida)
        eps = 1e-3
        x1c = torch.min(x1, x2 - eps)
        y1c = torch.min(y1, y2 - eps)

        # mapeia [0, 1] -> [-1, 1] (convenção grid_sample)
        x1n, x2n = x1c * 2 - 1, x2 * 2 - 1
        y1n, y2n = y1c * 2 - 1, y2 * 2 - 1

        theta = torch.zeros(box.size(0), 2, 3, device=box.device, dtype=box.dtype)
        theta[:, 0, 0] = (x2n - x1n) / 2
        theta[:, 0, 2] = (x2n + x1n) / 2
        theta[:, 1, 1] = (y2n - y1n) / 2
        theta[:, 1, 2] = (y2n + y1n) / 2
        return theta

    def _crop_roi(self, x, theta):
        """Recorte + redimensionamento diferenciável da ROI via
        affine_grid/grid_sample."""
        grid = F.affine_grid(theta, size=(x.size(0), x.size(1), self.crop_size, self.crop_size),
                              align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def _box_to_soft_mask(self, box):
        """Gera uma máscara de segmentação suave (diferenciável) da ROI, do
        mesmo tamanho da imagem de entrada, a partir da caixa delimitadora."""
        N = box.size(0)
        device, dtype = box.device, box.dtype
        size = self.image_size

        coords = torch.linspace(0, 1, size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        xx = xx.unsqueeze(0).expand(N, -1, -1)
        yy = yy.unsqueeze(0).expand(N, -1, -1)

        x1 = box[:, 0].view(N, 1, 1)
        y1 = box[:, 1].view(N, 1, 1)
        x2 = box[:, 2].view(N, 1, 1)
        y2 = box[:, 3].view(N, 1, 1)

        k = self.mask_sharpness
        mask_x = torch.sigmoid(k * (xx - x1)) * torch.sigmoid(k * (x2 - xx))
        mask_y = torch.sigmoid(k * (yy - y1)) * torch.sigmoid(k * (y2 - yy))
        mask = (mask_x * mask_y).unsqueeze(1)  # (N, 1, H, W)
        return mask

    def _points_to_original(self, points_local, theta):
        """Mapeia os pontos previstos no recorte (normalizados [-1, 1]) de
        volta para coordenadas de pixel na imagem original."""
        N, P, _ = points_local.shape
        px = points_local[..., 0]
        py = points_local[..., 1]

        sx = theta[:, 0, 0].unsqueeze(1)
        tx = theta[:, 0, 2].unsqueeze(1)
        sy = theta[:, 1, 1].unsqueeze(1)
        ty = theta[:, 1, 2].unsqueeze(1)

        orig_xn = sx * px + tx
        orig_yn = sy * py + ty

        # [-1, 1] -> pixels na imagem original
        orig_x = (orig_xn + 1) / 2 * self.image_size
        orig_y = (orig_yn + 1) / 2 * self.image_size
        return torch.stack([orig_x, orig_y], dim=-1)

    def forward(self, x):
        # --- Estágio 1: detecção global da ROI ---
        box = self.global_stage(x)                 # (N, 4) em [0, 1]
        theta = self._box_to_theta(box)
        roi_mask = self._box_to_soft_mask(box)      # (N, 1, H, W)

        # --- Recorte diferenciável da ROI ---
        crop = self._crop_roi(x, theta)             # (N, C, crop_size, crop_size)

        # --- Estágio 2: refinamento local dos landmarks ---
        points_local = self.local_stage(crop)       # (N, num_points, 2) em [-1, 1]
        points = self._points_to_original(points_local, theta)  # pixels na imagem original

        return roi_mask, None, points