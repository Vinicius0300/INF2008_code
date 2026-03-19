from src.utils import get_corners_from_angle
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

    def __init__(self,
                 video_frame_df: pd.DataFrame,
                 output_dim: tuple = (512, 512),
                 transform: A.Compose | None = None,
                 augmentation: str | None = None,
                 sigma: int = 10):
        
        self.video_frame_df = video_frame_df.reset_index(drop=True).copy()
        self.sigma = sigma
        self.output_dim = output_dim
        self.transform = transform
        self.augmentation = augmentation
    
    def __getitem__(self, idx:int):
        row = self.video_frame_df.iloc[idx]
        root = get_project_root_directory()

        # Carregamento Dados Imagem Original
        frame_path = self._resolve_path(root, row.frame_path)
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

        # Carregando Dados dos Target
        target_path = self._resolve_path(root, row.target_dir)
        keypoints = self._load_points(target_path) # [C2, C4]

        # Calculando Transformações
        if self.transform:
            transformed = self.transform(
                image=image,
                keypoints=keypoints,
            )
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        # Calcula Heatmap e Roi com base nos Keypoints Transformados
        h, w = self.output_dim    
        roi = generate_roi_from_points(keypoints, h, w)
        heatmaps = generate_heatmap_from_points(keypoints, self.output_dim, self.sigma)
        
        return image, keypoints, heatmaps, roi
    
    # Coleta Pontos
    def _load_points(self, path: str, filename: str = 'Results.csv') -> np.ndarray:
        """Carrega e converte os pontos do arquivo CSV."""
        full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Arquivo de pontos não encontrado: {full_path}")

        df = pd.read_csv(full_path)
        if df.empty:
            raise ValueError(f"Arquivo de pontos vazio: {full_path}")

        row = df.iloc[0]
        return get_corners_from_angle(
            row['BX'], row['BY'], row['Width'], row['Height'], row['Angle']
        )

    # Padroniza caminhos
    def _resolve_path(self, root: str, path: str) -> str:
        clean = path.replace("..\\", "").replace("../", "")
        return os.path.join(root, clean)
    
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
                roi = roi.cpu().numpy()
            plt.contour(roi, colors='lime', linewidths=1) # ROI (contorno verde)
        
        # Mostra Keypoints
        if display_keypoints:       
            keypoints = np.array(keypoints) # keypoints (pontos vermelhos)
            plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=20)

        plt.title(f"Sample {idx}")
        plt.axis("off")
        plt.show()