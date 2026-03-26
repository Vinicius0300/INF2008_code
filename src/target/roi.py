import math
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T

def generate_roi_from_points(points, img_height, img_width):
    """
    Gera bounding box (ROI) com base em dois pontos.
    A ROI terá:
        - largura = 2 * 0.7 distância entre os pontos
        - altura  = 2 * 0.7 distância entre os pontos
    E conterá os dois pontos.
    
    Retorna: (x_min, y_min, x_max, y_max)
    """
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Distância euclidiana entre os pontos
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Centro da bounding box: ponto médio entre p1 e p2
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Metade das dimensões da ROI
    half_w = dist * 0.7 
    half_h = dist * 0.7 # altura = 2 * dist → metade é dist

    # Coordenadas da ROI
    x_min = int(cx - half_w)
    x_max = int(cx + half_w)
    y_min = int(cy - half_h)
    y_max = int(cy + half_h)

    # Garantir que está dentro da imagem
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width - 1, x_max)
    y_max = min(img_height - 1, y_max)

    roi_mask = Image.new("L", (img_width, img_height), 0)  # fundo preto
    draw = ImageDraw.Draw(roi_mask)
    draw.rectangle([x_min, y_min, x_max, y_max], fill=255)  # ROI branca

    roi_mask = np.array(roi_mask, dtype=np.float32) / 255.0  # [0,1]
    roi_mask = torch.from_numpy(roi_mask).unsqueeze(0)  # (1, H, W)

    return roi_mask