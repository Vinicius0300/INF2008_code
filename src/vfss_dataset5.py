from src.utils import load_points, resolve_path
from src.target.heatmap import generate_heatmap_from_points
from src.target.roi import generate_roi_from_points
from src.utils import get_script_relative_path, get_project_root_directory

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch

import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Classe que trabalha com ROI, Heatmaps e Pontos
class VFSSImageDataset():

    # Incia a classe
    def __init__(self,
                 video_frame_df: pd.DataFrame,
                 output_dim: tuple = (512, 512),
                 transform: A.Compose | None = None,
                 offline_augmentation: bool = False,
                 sigma: int = 10):
        '''
        video_frame_df: dataframe com cada linha indicando aonde encontrar
                        o frame e os targets
        output_dim:     Dimensão de output do problema
        transforma:     Transformação que será aplicada no problema (SOMENTE NOS DADOS DE TREINO)
        offiline_augmentation: True or False. Caso True espera que o data frame 
                        enviado tenha uma coluna para o caminho do frame e outra 
                        coluna com os keypoints já processados. Além disso, 
                        caso True, não devemos passar transform, visto que isso 
                        é feita na geração dos dados.
        sigma:          aplicado na distribuição gaussiana que gera os heatmaps
        '''
        self.video_frame_df = video_frame_df.reset_index(drop=True).copy()
        self.sigma = sigma
        self.output_dim = output_dim
        self.transform = transform
        self.offline_augmentation = offline_augmentation

    # Pega um item
    def __getitem__(self, idx:int):
        row = self.video_frame_df.iloc[idx]
        root = get_project_root_directory()

        # Carregamento Dados Imagem Original
        frame_path = resolve_path(root, row.frame_path)
        image = np.array(Image.open(frame_path).convert("L"))
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Carrega pontos pelo dataset com offline augmentation
        if self.offline_augmentation:
            keypoints = row.keypoints # [C2, C4]

        # Carrega Pontos pelo dataframe original
        else:
            # Carregando Dados dos Target
            target_path = resolve_path(root, row.target_dir)
            keypoints = load_points(target_path) # [C2, C4]

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
        heatmaps = generate_heatmap_from_points(keypoints, self.output_dim, self.sigma)
        
        
        return image, keypoints, heatmaps, roi
    
    # Retorna o Tamanho da Base considera
    def __len__(self):
        return len(self.video_frame_df)

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