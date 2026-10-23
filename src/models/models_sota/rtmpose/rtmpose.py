"""
RTMPose adaptado para detecção de keypoints em vértebras C2/C4.

Referência: RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose
            (Jiang et al., 2023) — https://arxiv.org/abs/2303.07399

Backbone: CSPResNeXt-50 via timm (disponível em timm==0.9.16, já no requirements.txt).
Neck:     CSP-PAN (fusão multi-escala P3/P4/P5).
Head:     Heatmap + ROI — retorno (roi, heatmap, None), compatível com o pipeline existente.

Pré-treinamento
---------------
O RTMPose original usa um backbone CSPNeXt pré-treinado na ImageNet e depois afina o
modelo completo (backbone + neck + head) no COCO para pose estimation.

Como a implementação aqui é independente do MMPose, os pesos disponíveis para
transferência são os do backbone CSPResNeXt-50 pré-treinado na **ImageNet** via timm.
Isso já provê uma inicialização forte que acelera a convergência.

    Treinar do zero : pretrained_path=None
    Fine-tuning     : pretrained_path='data/model_weights/RTMPose/model_rtmpose_coco.pth'
                      (pesos gerados por download_rtmpose_backbone_weights)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ─────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_channels, in_channels, kernel_size,
                             stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pw  = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CSPLayer(nn.Module):
    """Cross Stage Partial layer para o neck PAN."""
    def __init__(self, in_channels, out_channels, n_blocks=3):
        super().__init__()
        hidden = out_channels // 2
        self.conv_pre   = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.conv_short = nn.Conv2d(out_channels, hidden, 1, bias=False)
        self.conv_main  = nn.Conv2d(out_channels, hidden, 1, bias=False)
        self.blocks     = nn.Sequential(*[
            DepthwiseSeparableConv(hidden, hidden) for _ in range(n_blocks)])
        self.conv_out   = nn.Sequential(
            nn.Conv2d(hidden * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))

    def forward(self, x):
        x     = self.conv_pre(x)
        short = self.conv_short(x)
        main  = self.blocks(self.conv_main(x))
        return self.conv_out(torch.cat([main, short], dim=1))


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────
#  BACKBONE  (CSPResNeXt-50 via timm)
# ─────────────────────────────────────────────

class CSPResNeXtBackbone(nn.Module):
    """
    Wrapper do CSPResNeXt-50 do timm como extrator de features multi-escala.

    Produz 3 feature maps usados pelo neck:
        P3: stride 8   (512 canais)
        P4: stride 16  (1024 canais)
        P5: stride 32  (2048 canais)

    Suporta entrada monocanal (grayscale) convertendo os pesos da primeira
    convolução de RGB para 1 canal por média.

    Parâmetros
    ----------
    in_channels : int
        Canais de entrada (1 = grayscale, 3 = RGB).
    pretrained_timm : bool
        Se True, carrega pesos ImageNet do timm ao construir o backbone.
        Usado internamente por download_rtmpose_backbone_weights; durante o
        treinamento normal use pretrained_path no RTMPose, não este flag.
    """

    # Canais do cspresnext50 nas escalas P3, P4, P5
    OUT_CHANNELS = [512, 1024, 2048]

    def __init__(self, in_channels: int = 1, pretrained_timm: bool = False):
        super().__init__()
        self.body = timm.create_model(
            'cspresnext50',
            pretrained=pretrained_timm,
            features_only=True,
            out_indices=(2, 3, 4),   # P3 (stride8), P4 (stride16), P5 (stride32)
        )

        # Adaptação RGB → grayscale na primeira convolução
        if in_channels != 3:
            old_conv   = self.body.conv1
            new_weight = old_conv.weight.data.mean(dim=1, keepdim=True)
            self.body.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            self.body.conv1.weight.data = new_weight
            if old_conv.bias is not None:
                self.body.conv1.bias.data = old_conv.bias.data.clone()

    def forward(self, x):
        p3, p4, p5 = self.body(x)
        return p3, p4, p5


# ─────────────────────────────────────────────
#  NECK  (CSP-PAN)
# ─────────────────────────────────────────────

class CSPPAN(nn.Module):
    """
    Neck PAN com camadas CSP.
    Recebe (P3, P4, P5) e devolve (out_p3, out_p4) — strides 8 e 16.
    """
    def __init__(self, in_channels):
        super().__init__()
        c3, c4, c5 = in_channels   # 512, 1024, 2048

        # Top-down
        self.reduce_p5  = nn.Sequential(
            nn.Conv2d(c5, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.SiLU(True))
        self.fuse_p4_td = CSPLayer(c4 * 2, c4, n_blocks=3)

        self.reduce_p4  = nn.Sequential(
            nn.Conv2d(c4, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(True))
        self.fuse_p3_td = CSPLayer(c3 * 2, c3, n_blocks=3)

        # Bottom-up
        self.down_p3    = DownsampleConv(c3, c3)
        self.fuse_p4_bu = CSPLayer(c3 + c4, c4, n_blocks=3)

        self.out_channels = [c3, c4]

    def forward(self, features):
        p3, p4, p5 = features

        # Top-down
        p5r  = F.interpolate(self.reduce_p5(p5), size=p4.shape[-2:], mode='nearest')
        p4td = self.fuse_p4_td(torch.cat([p5r, p4], dim=1))

        p4r  = F.interpolate(self.reduce_p4(p4td), size=p3.shape[-2:], mode='nearest')
        p3td = self.fuse_p3_td(torch.cat([p4r, p3], dim=1))

        # Bottom-up
        p3d  = self.down_p3(p3td)
        p4bu = self.fuse_p4_bu(torch.cat([p3d, p4td], dim=1))

        return p3td, p4bu


# ─────────────────────────────────────────────
#  HEAD  (Heatmap + ROI)
# ─────────────────────────────────────────────

class RTMPoseHeatmapHead(nn.Module):
    """
    Cabeça de heatmap.
    Recebe o feature map P3 (stride 8) e produz heatmaps na resolução output_size.
    Retorna (roi, heatmap, None).
    """
    def __init__(self, in_channels: int, num_keypoints: int = 2,
                 output_size: tuple = (256, 256)):
        super().__init__()
        self.output_size = output_size
        self.refine  = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
        )
        self.head_kp  = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.head_roi = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feat, predict_roi: bool = True):
        feat    = self.refine(feat)
        feat_up = F.interpolate(feat, size=self.output_size,
                                mode='bilinear', align_corners=False)
        heatmap = self.head_kp(feat_up)
        roi     = self.head_roi(feat_up) if predict_roi else None
        return roi, heatmap, None


# ─────────────────────────────────────────────
#  MODELO PRINCIPAL
# ─────────────────────────────────────────────

class RTMPose(nn.Module):
    """
    RTMPose adaptado para keypoints em vértebras C2/C4.

    Parâmetros
    ----------
    num_keypoints : int
        Número de keypoints de saída (default=2).
    in_channels : int
        Canais de entrada — 1 para grayscale (default).
    output_size : tuple[int, int]
        Resolução dos heatmaps de saída (default=(256, 256)).
    pretrained_path : str | None
        Caminho para pesos pré-treinados locais gerados por
        download_rtmpose_backbone_weights(). Se None, o modelo é
        inicializado do zero.
    return_mode : str
        "both"         → (roi, heatmap, None)
        "heatmap_only" → (None, heatmap, None)
        "roi_only"     → (roi, None, None)
    """

    def __init__(
        self,
        num_keypoints: int = 2,
        in_channels: int = 1,
        output_size: tuple = (256, 256),
        pretrained_path: str | None = None,
        return_mode: str = "both",
    ):
        super().__init__()
        self.return_mode = return_mode

        # Backbone
        self.backbone = CSPResNeXtBackbone(in_channels=in_channels)

        # Neck
        self.neck = CSPPAN(CSPResNeXtBackbone.OUT_CHANNELS)

        # Head  (usa P3 — stride 8)
        self.head = RTMPoseHeatmapHead(
            in_channels=self.neck.out_channels[0],
            num_keypoints=num_keypoints,
            output_size=output_size,
        )

        # Inicialização do neck e head
        self._init_neck_head()

        # Pré-treinamento do backbone
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    # ------------------------------------------------------------------

    def _init_neck_head(self):
        """Kaiming para neck e head; backbone já inicializado pelo timm."""
        for module in (self.neck, self.head):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained(self, weight_path: str):
        """
        Carrega pesos do backbone a partir de um arquivo .pth gerado por
        download_rtmpose_backbone_weights().

        Apenas as camadas do backbone são atualizadas; neck e head são
        mantidos com inicialização aleatória (Kaiming).

        Conversão RGB→grayscale automática para a primeira convolução.
        """
        if not os.path.exists(weight_path):
            print(f"[RTMPose] Arquivo de pesos não encontrado: {weight_path}")
            return

        checkpoint   = torch.load(weight_path, map_location='cpu', weights_only=False)
        state_dict   = (checkpoint.get('backbone_state_dict')
                        or checkpoint.get('state_dict')
                        or checkpoint)

        model_dict   = self.backbone.state_dict()
        matched, skipped = {}, []

        for k, v in state_dict.items():
            # Remove prefixo 'body.' se o checkpoint foi salvo com ele
            k_clean = k[len('body.'):] if k.startswith('body.') else k

            # Tenta com e sem o prefixo 'body.'
            for target in (f'body.{k_clean}', k_clean, k):
                if target in model_dict:
                    v_load = v
                    # RGB → grayscale para a primeira conv
                    if ('conv1' in target and v.dim() == 4
                            and v.shape[1] == 3
                            and model_dict[target].shape[1] == 1):
                        v_load = v.mean(dim=1, keepdim=True)

                    if v_load.shape == model_dict[target].shape:
                        matched[target] = v_load
                    else:
                        skipped.append(target)
                    break

        self.backbone.load_state_dict(matched, strict=False)
        print(f"[RTMPose] Backbone: {len(matched)} camadas carregadas, "
              f"{len(skipped)} ignoradas (shape incompatível).")

    def set_return_mode(self, mode: str):
        self.return_mode = mode

    def forward(self, x):
        features        = self.backbone(x)              # (P3, P4, P5)
        p3, _           = self.neck(features)
        predict_roi     = self.return_mode in ("both", "roi_only")
        roi, heatmap, _ = self.head(p3, predict_roi=predict_roi)

        if self.return_mode == "both":
            return roi, heatmap, None
        elif self.return_mode == "heatmap_only":
            return None, heatmap, None
        elif self.return_mode == "roi_only":
            return roi, None, None
        else:
            raise ValueError(f"[RTMPose] return_mode inválido: {self.return_mode}")


# ─────────────────────────────────────────────
#  UTILITÁRIO — download de pesos
# ─────────────────────────────────────────────

def download_rtmpose_backbone_weights(
    save_path: str = 'data/model_weights/RTMPose/model_rtmpose_coco.pth'
):
    """
    Baixa os pesos do backbone CSPResNeXt-50 pré-treinado na ImageNet via timm
    e salva localmente para uso em RTMPose(pretrained_path=...).

    Sobre o nome "model_rtmpose_coco"
    ----------------------------------
    O RTMPose paper tem dois estágios de treinamento:
      1. Backbone (CSPNeXt) pré-treinado na **ImageNet**.
      2. Modelo completo (backbone + neck + head) afinado no **COCO** para pose.

    Os pesos aqui salvos correspondem ao estágio 1 — backbone ImageNet —
    disponíveis no timm 0.9.16 através do modelo cspresnext50.
    Os pesos COCO do estágio 2 (modelo completo) exigiriam a implementação
    exata do MMPose para ser carregados, o que foge do escopo deste projeto.

    Como usar
    ---------
    # 1. Baixar uma única vez:
    from src.models.models_sota.rtmpose.rtmpose import download_rtmpose_backbone_weights
    download_rtmpose_backbone_weights('data/model_weights/RTMPose/model_rtmpose_coco.pth')

    # 2. Usar no TrainingConfig:
    model_kwargs = dict(pretrained_path='data/model_weights/RTMPose/model_rtmpose_coco.pth', ...)

    Parâmetros
    ----------
    save_path : str
        Caminho completo onde o arquivo .pth será salvo.

    Retorna
    -------
    str : save_path confirmado.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[RTMPose] Baixando backbone CSPResNeXt-50 (ImageNet) via timm...")
    backbone = CSPResNeXtBackbone(in_channels=3, pretrained_timm=True)

    # Salva o state_dict do backbone inteiro (com prefixo 'body.')
    torch.save({'backbone_state_dict': backbone.state_dict()}, save_path)

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[RTMPose] Pesos salvos em: {save_path}")
    print(f"[RTMPose] Parâmetros do backbone: {n_params:,}")
    return save_path
