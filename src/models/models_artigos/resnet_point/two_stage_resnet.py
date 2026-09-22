"""
Pipeline de treino em duas etapas (fiel ao artigo), integrado ao seu
framework (TrainingConfig / holdout / VFSSImageDataset).

Nada é salvo em disco além do que o próprio holdout() já salva por conta
própria (checkpoints, config.pkl, metrics_results.json -- isso está em
holdout.py/train_fold.py/checkpoint_manager e não é alterado aqui).
O recorte para o Estágio 2 é feito inteiramente em memória: nenhuma
imagem recortada e nenhum CSV novo são gravados.

Pré-requisitos deste arquivo (ajuste os imports para o seu projeto):
    from src.training.config import TrainingConfig
    from src.training.holdout import holdout
    from src.utils import custom_collate_fn
    from src.datasets.vfss_dataset import VFSSImageDataset
    from src.target.heatmap import generate_heatmap_from_points
    from src.target.roi import generate_roi_from_points

Este arquivo é autossuficiente: a arquitetura (SwitchNorm2d, blocos
residuais, GlobalStageResNet50, LocalStageResNet34) e as funções de
coordenadas/recorte (denormalize_points, compute_crop_box) estão
definidas aqui mesmo, sem depender de two_stage_resnet_revisado.py.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# 0a) NORMALIZAÇÃO: SWITCHABLE NORMALIZATION
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

        mean_in = x.mean(dim=[2, 3], keepdim=True)
        var_in = x.var(dim=[2, 3], keepdim=True, unbiased=False)

        mean_ln = mean_in.mean(dim=1, keepdim=True)
        var_ln = var_in.mean(dim=1, keepdim=True) + mean_in.var(dim=1, keepdim=True, unbiased=False)

        if self.training:
            mean_bn = mean_in.mean(dim=0, keepdim=True)
            var_bn = var_in.mean(dim=0, keepdim=True) + mean_in.var(dim=0, keepdim=True, unbiased=False)
            with torch.no_grad():
                safe_mean = mean_bn.detach().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                safe_var  = var_bn.detach().nan_to_num(nan=1.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
                self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * safe_mean)
                self.running_var.mul_(self.momentum).add_((1 - self.momentum) * safe_var)
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
# 0b) BLOCOS RESIDUAIS BÁSICOS
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
# 0c) ESTÁGIO 1: REDE DE DETECÇÃO GLOBAL (ResNet-50) -> pontos grosseiros
# ============================================================

class GlobalStageResNet50(nn.Module):
    """
    Saída = coordenadas dos landmarks (C2 e C4), normalizadas e
    centralizadas conforme a Eq. (1) do artigo, no referencial da
    imagem inteira (448x448). Ativação LINEAR (sem sigmoid/tanh), para
    não distorcer a escala de erro usada na perda Euclidiana (Eq. 2).

    Reflete o texto do artigo: a rede global foi treinada "só com
    rótulos de C2 e C4" -- ela já prevê diretamente os pontos (numa
    versão grosseira), com a MESMA função de perda usada no Estágio 2.
    """
    def __init__(self, in_channels=1, num_points=2):
        super(GlobalStageResNet50, self).__init__()
        self.inplanes = 64
        self.num_points = num_points

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
        self.fc_points = nn.Linear(512 * BottleneckSN.expansion, num_points * 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        pts = self.fc_points(x)                 # saída LINEAR -> coords normalizadas (Eq. 1)
        return pts.view(-1, self.num_points, 2)


# ============================================================
# 0d) ESTÁGIO 2: REDE DE DETECÇÃO LOCAL (ResNet-34) -> refinamento
# ============================================================

class LocalStageResNet34(nn.Module):
    """
    Sem ativação tanh na saída: o artigo não impõe limite artificial
    [-1, 1] aos pontos regredidos -- a rede aprende a regressão
    Euclidiana (Eq. 2) livremente sobre as coordenadas normalizadas do
    RECORTE.
    """
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
        pts = self.fc_points(x)                 # saída LINEAR -> coords normalizadas (Eq. 1), relativas ao recorte
        return pts.view(-1, self.num_points, 2)


# ============================================================
# 0e) FUNÇÕES DE COORDENADAS E RECORTE
# ============================================================

def denormalize_points(points_norm, size):
    """Inverte a Eq. (1) do artigo: de coordenadas centralizadas/normalizadas
    de volta para pixels, dado o tamanho de referência (imagem ou recorte)."""
    return points_norm * size + 0.5 * size


def normalize_points(points_px, size):
    """Aplica a Eq. (1) do artigo: centraliza e normaliza pontos em pixels.
    Use isto para gerar os alvos (labels) de treino a partir das anotações
    em pixels, antes de calcular a perda Euclidiana (Eq. 2)."""
    return (points_px - 0.5 * size) / size


def compute_crop_box(points_px, image_size, margin_ratio=0.6, min_side_ratio=0.25):
    x_min = points_px[..., 0].min(dim=1).values
    x_max = points_px[..., 0].max(dim=1).values
    y_min = points_px[..., 1].min(dim=1).values
    y_max = points_px[..., 1].max(dim=1).values

    w = (x_max - x_min).clamp(min=1.0)
    h = (y_max - y_min).clamp(min=1.0)
    side = torch.maximum(w, h) * (1.0 + margin_ratio)
    side = torch.clamp(side, min=min_side_ratio * image_size)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    x1 = (cx - side / 2).clamp(0, image_size - 1)
    y1 = (cy - side / 2).clamp(0, image_size - 1)
    x2 = (cx + side / 2).clamp(0, image_size)
    y2 = (cy + side / 2).clamp(0, image_size)

    x2 = torch.maximum(x2, x1 + 1)
    y2 = torch.maximum(y2, y1 + 1)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def euclidean_landmark_loss(pred_px, target_px):
    """Implementa a Eq. (2) do artigo:
    loss = 1/2 * sum_i [(x_i - x_i')^2 + (y_i - y_i')^2]
    pred_px, target_px: (N, num_points, 2) em pixels."""
    diff2 = (pred_px - target_px) ** 2
    return 0.5 * diff2.sum(dim=-1).sum(dim=-1).mean()


# ============================================================
# 1) WRAPPERS -- adaptam GlobalStageResNet50 / LocalStageResNet34
#    ao formato (pred_roi, pred_heatmap, pred_keypoints) que o seu
#    train_epoch.py espera.
# ============================================================

class GlobalStageWrapper(nn.Module):
    """Usado na config do Estágio 1. Recebe a imagem inteira (448x448)
    e devolve só os pontos grosseiros -- roi e heatmap ficam None,
    então o LossCalculator deve ignorá-los (peso 0 + checagem de None)."""
    def __init__(self, in_channels=1, num_points=2, image_size=448):
        super().__init__()
        self.net = GlobalStageResNet50(in_channels, num_points)
        self.image_size = image_size

    def forward(self, x):
        pts_norm = self.net(x)
        pts_px = denormalize_points(pts_norm, self.image_size)
        return None, None, pts_px


class LocalStageWrapper(nn.Module):
    """Usado na config do Estágio 2. Recebe o RECORTE já redimensionado
    para crop_size e devolve os pontos refinados, no referencial do
    próprio recorte (o dataset do Estágio 2 já gera gt_keypoints nesse
    mesmo referencial, então bate certinho com a loss)."""
    def __init__(self, in_channels=1, num_points=2, crop_size=224):
        super().__init__()
        self.net = LocalStageResNet34(in_channels, num_points)
        self.crop_size = crop_size

    def forward(self, x):
        pts_norm = self.net(x)
        pts_px = denormalize_points(pts_norm, self.crop_size)
        return None, None, pts_px


# ============================================================
# 2) GERAÇÃO DO crop_box EM MEMÓRIA, usando o Estágio 1 já treinado
# ============================================================

def add_crop_boxes_to_df(df, model_stage1, device, dataset_class,
                          image_size=448, margin_ratio=0.6,
                          min_side_ratio=0.25, base_transform=None,
                          batch_size=8, collate_fn=None):
    """
    Roda o modelo do Estágio 1 (já treinado, .eval()) sobre cada linha de
    `df`, usando SEMPRE um transform determinístico (sem augmentation --
    `base_transform`, tipicamente o mesmo transform_validation usado no
    Estágio 1), e adiciona a coluna 'crop_box' com [x1, y1, x2, y2] em
    pixels no referencial image_size x image_size.

    IMPORTANTE: `base_transform` precisa ser o MESMO transform (resize
    determinístico) usado para treinar o Estágio 1 -- é o que garante que
    o crop_box calculado aqui esteja no mesmo referencial de pixels que
    o CroppedVFSSImageDataset vai usar para recortar a imagem depois.

    Nada é salvo em disco: o resultado é só um DataFrame em memória.
    """
    model_stage1.eval()
    ds = dataset_class(df, output_dim=(image_size, image_size),
                        transform=base_transform, sigma_heatmap=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_fn)

    boxes = []
    with torch.no_grad():
        for images, _keypoints, _heatmaps, _roi in loader:
            images = images.to(device)
            _, _, coarse_points = model_stage1(images)
            batch_boxes = compute_crop_box(coarse_points, image_size,
                                            margin_ratio, min_side_ratio)
            boxes.extend(batch_boxes.cpu().tolist())

    df = df.reset_index(drop=True).copy()
    df['crop_box'] = boxes
    return df


# ============================================================
# 3) DATASET DO ESTÁGIO 2 -- recorta em memória, sem salvar nada
# ============================================================

class CroppedVFSSImageDataset:
    """
    Para cada amostra:
      1) Carrega o frame original e aplica `base_transform` (o MESMO
         transform_validation usado no Estágio 1 -- resize determinístico
         para `base_size`, ex.: 448x448), levando a imagem e os pontos
         para o MESMO referencial em que o crop_box foi calculado.
      2) Recorta a região indicada pela coluna 'crop_box' do dataframe
         (gerada por add_crop_boxes_to_df), em memória.
      3) Aplica `transform` (resize para output_dim + augmentation),
         igual ao VFSSImageDataset original -- e reaproveita a mesma
         lógica de geração de heatmap/roi/keypoints a partir daí.

    Assinatura compatível com o que holdout.py espera de dataset_class:
    dataset_class(df, output_dim, transform, sigma_heatmap=...).
    Os parâmetros extras (base_transform, base_size) devem ser fixados
    com functools.partial antes de passar como config.dataset_class
    (ver exemplo de config_stage2 abaixo).
    """
    def __init__(self, video_frame_df, output_dim=(224, 224),
                 transform=None, sigma_heatmap=10,
                 base_transform=None, base_size=448,
                 generate_roi_from_points=None,
                 generate_heatmap_from_points=None):
        self.video_frame_df = video_frame_df.reset_index(drop=True).copy()
        self.output_dim = output_dim
        self.transform = transform
        self.sigma_heatmap = sigma_heatmap
        self.base_transform = base_transform
        self.base_size = base_size
        self.video_frame_list = self.video_frame_df.to_dict('records')

        # injete aqui as funções reais do seu projeto
        # (src.target.roi.generate_roi_from_points, etc.)
        self._generate_roi_from_points = generate_roi_from_points
        self._generate_heatmap_from_points = generate_heatmap_from_points

    def __getitem__(self, idx):
        row = self.video_frame_list[idx]
        frame_path = row['frame_path']
        keypoints = row['keypoints']
        crop_box = row['crop_box']  # [x1, y1, x2, y2] no referencial base_size

        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {frame_path}")
        image = np.expand_dims(image, axis=-1)

        # 1) leva imagem + pontos pro MESMO referencial usado no crop_box
        if self.base_transform:
            base = self.base_transform(image=image, keypoints=keypoints)
            image, keypoints = base["image"], base["keypoints"]
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()

        # 2) recorta em memória (sem salvar nada em disco)
        x1, y1, x2, y2 = [int(round(v)) for v in crop_box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2 = min(x2, self.base_size)
        y2 = min(y2, self.base_size)

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, self.base_size, self.base_size

        image = image[y1:y2, x1:x2, :]
        keypoints = [(kx - x1, ky - y1) for (kx, ky) in keypoints]

        # 3) resize para output_dim + augmentation (igual ao dataset original)
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image, keypoints = transformed["image"], transformed["keypoints"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        h, w = self.output_dim
        roi = self._generate_roi_from_points(keypoints, h, w)
        heatmaps = self._generate_heatmap_from_points(keypoints, self.output_dim, self.sigma_heatmap)
        image = image.float() / 255.0

        return image, keypoints, heatmaps, roi

    def __len__(self):
        return self.video_frame_df.shape[0]

