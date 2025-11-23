import math
from PIL import Image, ImageDraw
import torchvision.transforms as T

def generate_roi_from_points(points, img_height, img_width):
    """
    Gera bounding box (ROI) com base em dois pontos.
    A ROI terá:
        - largura = distância entre os pontos
        - altura  = 2x distância entre os pontos
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
    half_w = dist 
    half_h = dist # altura = 2 * dist → metade é dist

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

    return roi_mask