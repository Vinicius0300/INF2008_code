import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

def generate_heatmap_from_points(points, orig_size, sigma=10):
    """
    Gera mapas de calor (heatmaps) para pontos-chave.

    Parâmetros:
    -----------
    orig_size : tuple (H, W)
        Tamanho original da imagem onde os points estão definidos.
    points : list of tuples [(x1, y1), (x2, y2), ...]
        Lista de coordenadas dos pontos-chave (em pixels).
    target_size : tuple (H_new, W_new)
        Tamanho final desejado para o heatmap (após redimensionar).
    sigma : float
        Desvio padrão da Gaussiana que define a "espalhamento" do ponto.

    Retorna:
    --------
    torch.Tensor de shape (num_points, H_new, W_new)
    """

    H, W = orig_size
    num_points = len(points)

    # Inicializa heatmaps na dimensão original
    heatmaps = np.zeros((num_points, H, W), dtype=np.float32)

    # Cria grade de coordenadas
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    for i, (x, y) in enumerate(points):
        if x < 0 or y < 0 or x >= W or y >= H:
            continue  # ignora points fora da imagem
        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

    # Converte para tensor e adiciona dimensão de batch para redimensionar com F.interpolate
    heatmaps_t = torch.tensor(heatmaps).unsqueeze(0)  # shape (1, num_points, H, W)

    # Redimensiona para o tamanho desejado
    heatmaps_resized = F.interpolate(
        heatmaps_t, 
        size=orig_size, 
        mode='bilinear', 
        align_corners=False
    )

    # Remove batch dimension
    heatmaps_resized = heatmaps_resized.squeeze(0)
    heatmaps_resized = T.ToPILImage()(heatmaps_resized)

    return heatmaps_resized

