import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch

from src.utils import load_points
from src.target.heatmap import generate_heatmap_from_points
from src.target.roi import generate_roi_from_points
from src.utils import get_script_relative_path, get_project_root_directory

# Classe que trabalha com ROI, Heatmaps e Pontos
class VFSSImageDataset():

    # Incia a classe
    def __init__(self,
                 video_frame_df: pd.DataFrame,
                 output_dim: tuple = (512, 512),
                 transform: A.Compose | None = None,
                 sigma_heatmap: int = 10):
        '''
        video_frame_df: dataframe com cada linha indicando aonde encontrar
                        o frame e os targets
        output_dim:     Dimensão de output do problema
        transforma:     Transformação que será aplicada no problema (SOMENTE NOS DADOS DE TREINO)
        sigma_heatmap:  aplicado na distribuição gaussiana que gera os heatmaps
        '''
        
        self.video_frame_df = video_frame_df.reset_index(drop=True).copy()
        self.sigma_heatmap = sigma_heatmap
        self.output_dim = output_dim
        self.transform = transform
        self.video_frame_list = self.video_frame_df.to_dict('records')

    # No seu __getitem__, altere o carregamento inicial para isto:
    def __getitem__(self, idx:int):
        row = self.video_frame_list[idx]
        
        # Como agora é um dicionário, acessamos pelas chaves
        frame_path = row['frame_path']
        keypoints = row['keypoints']

        # Usando OpenCV direto (MUITO mais rápido para ler do disco)
        # cv2.IMREAD_GRAYSCALE já carrega em 1 canal
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {frame_path}")

        image = np.expand_dims(image, axis=-1) # Adiciona o canal (H, W, 1)

        # Calculando Transformações
        if self.transform:
            transformed = self.transform(
                image=image,
                keypoints=keypoints,
            )
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        # Garantir que image é um tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Calcula Heatmap e Roi com base nos Keypoints Transformados
        h, w = self.output_dim    
        roi = generate_roi_from_points(keypoints, h, w)
        heatmaps = generate_heatmap_from_points(keypoints, self.output_dim, self.sigma_heatmap)
        image = image.float() / 255.0

        return image, keypoints, heatmaps, roi
    
    # Retorna o Tamanho da Base considera
    def __len__(self):
        return self.video_frame_df.shape[0]

    # Plot de Sample
    def plot_sample(self, idx,
                    display_keypoints = True,
                    display_heatmaps = True,
                    display_roi = True):

        image, keypoints, heatmaps, roi = self[idx]

        plt.figure(figsize=(6,6))
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        plt.imshow(image, cmap='gray')

        # Mostra Heatmaps
        if display_heatmaps:
            if isinstance(heatmaps, torch.Tensor):
                heatmaps = heatmaps.cpu().numpy()
            if heatmaps.ndim == 3:
                heatmap = heatmaps.max(axis=0)  # junta canais
            else:
                heatmap = heatmaps
            plt.imshow(heatmap, cmap='jet', alpha=0.2) # heatmap (overlay vermelho)

        # Mostra ROI
        if display_roi:
            if isinstance(roi, torch.Tensor):
                roi = roi.squeeze().cpu().numpy()  # (H, W)
            plt.contour(roi, colors='lime', linewidths=1)
        
        # Mostra Keypoints
        if display_keypoints:       
            keypoints = np.array(keypoints) # keypoints (pontos vermelhos)
            plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=20)

        plt.title(f"Sample {idx}")
        plt.axis("off")
        plt.show()