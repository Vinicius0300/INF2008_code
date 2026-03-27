import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

def generate_heatmap_from_points(points, orig_size, sigma=10):

    H, W = orig_size
    num_points = len(points)

    # Já cria o fundo do heatmap zerado em PyTorch
    heatmaps = torch.zeros((num_points, H, W), dtype=torch.float32)
    
    # Regra dos 3 Sigma: a Gaussiana morre (vira ~0) após 3x o sigma
    radius = int(3 * sigma)
    
    for i, (px, py) in enumerate(points):
        px, py = int(px), int(py)
        
        # Define os limites do "patch" ao redor do ponto
        x0 = max(0, px - radius)
        y0 = max(0, py - radius)
        x1 = min(W, px + radius + 1)
        y1 = min(H, py + radius + 1)
        
        # Se o ponto cair totalmente fora da imagem, ignora
        if x0 >= x1 or y0 >= y1:
            continue
            
        # Cria um grid APENAS para o tamanho do patch (ex: 60x60 pixels em vez de 512x512)
        y_grid = torch.arange(y0, y1, dtype=torch.float32).view(-1, 1)
        x_grid = torch.arange(x0, x1, dtype=torch.float32).view(1, -1)
        
        # Calcula a Gaussiana só no patch
        dist_sq = (x_grid - px)**2 + (y_grid - py)**2
        patch = torch.exp(-dist_sq / (2 * sigma**2))
        
        # Cola o patch calculado no heatmap original
        heatmaps[i, y0:y1, x0:x1] = patch
        
    # Normalização
    heatmaps = heatmaps / (heatmaps.amax(dim=(1,2), keepdim=True) + 1e-8)
    
    return heatmaps
    

