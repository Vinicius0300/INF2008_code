"""
Execute este script UMA ÚNICA VEZ para baixar os pesos pré-treinados do backbone
do RTMPose (CSPResNeXt-50 / ImageNet) e salvá-los localmente.

Uso (na raiz do projeto):
    python download_rtmpose_weights.py

Requisito: huggingface_hub instalado e compatível com timm==0.9.16
    pip install "huggingface_hub>=0.19.0,<0.23.0"
"""
import sys, os

# Garante que o script roda da raiz do projeto
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from src.models.models_sota.rtmpose.rtmpose import download_rtmpose_backbone_weights

save_path = os.path.join('data', 'model_weights', 'RTMPose', 'model_rtmpose_coco.pth')
download_rtmpose_backbone_weights(save_path)
