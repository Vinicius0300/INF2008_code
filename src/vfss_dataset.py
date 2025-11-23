from src.utils import get_corners_from_angle
from src.target.heatmap import generate_heatmap_from_points
from src.target.roi import generate_roi_from_points
from src.utils import get_script_relative_path, get_project_root_directory

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

import cv2 as cv
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class VFSSImageDataset(Dataset):
    def __init__(self, video_frame_df: pd.DataFrame, target='mask', output_dim=(256, 256), transform=None, target_transform: list=None, sigma: float=10):
        '''
        Dataset para carregar frames de vídeos VFSS e seus respectivos alvos (máscaras, pontos, ROI, mapas de calor).

        Args:
            video_frame_df (pd.DataFrame): DataFrame contendo colunas 'frame_path', 'frame_id' e 'target_dir'.
            target (str): Tipo de alvo a ser carregado ('mask', 'points', 'roi', 'heatmap' ou combinação deles separados por '+').
            output_dim (tuple): Dimensão de saída para mapas de calor ou ROI.
            transform (callable, optional): Transformação a ser aplicada às imagens.
            target_transform (callable, optional): Transformação a ser aplicada aos alvos.
            sigma (int): variance of the gaussian distribution used to create the heatmap
        '''

        self.video_frame_df = (
            video_frame_df
            .reset_index(drop=True)
            .copy()
        )
        
        self.__valid_target = ['mask', 'points', 'roi', 'heatmap']
        if self.__validate_target(target):
            self.target = target
        
        self.output_dim = output_dim
        
        if transform:
            self.transform = transform
        else:
            self.transform = T.Resize(output_dim)
        
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = T.Resize(output_dim)

        self.sigma = sigma
    
    def __len__(self):
        return self.video_frame_df.shape[0]

    def _repr_html_(self):
        # Jupyter uses this to render rich HTML
        return self.video_frame_df._repr_html_()

    def __repr__(self):
        # Fallback for console or plain-text output
        return repr(self.video_frame_df)
    
    def __validate_target(self, target):
        '''Validate if the target string contains only valid target types.'''
        for t in target.split('+'):
            if t not in self.__valid_target:
                raise ValueError(f"Invalid target: {t}. Must be one of {self.__valid_target}")
        return True

    def __load_frame_from_path(self, path: str, frame: int):
        ''' Load a specific frame from a video file given its path and frame index. '''

        root_dir = get_project_root_directory()
        path = path.replace("..\\", "").replace("../", "")
        relative_path = os.path.join(root_dir, path)

        if not os.path.exists(relative_path):
            raise FileNotFoundError(f"File not found: {relative_path}")

        cap = cv.VideoCapture(relative_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {relative_path}")
        
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        if frame <= 0 or frame > total_frames:
            raise IndexError(f"Frame index {frame} out of range for video with {total_frames} frames.")

        # Set frame position and read it
        cap.set(cv.CAP_PROP_POS_FRAMES, frame-1)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            raise IOError(f"Error reading frame {frame} from video file: {relative_path}")

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        return image

    def __load_mask_from_path(self, path: str, filename='Mask.tif'):
        ''' Load mask image from the given path. '''
        path = os.path.join(path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        mask = Image.open(path).convert("L")
        return mask        

    def __load_points_from_path(self, path: str, filename='Results.csv'):
        ''' Load points data from the given path. '''
        path = os.path.join(path, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        points_df = pd.read_csv(path)
        if points_df is None or points_df.empty:
            raise ValueError(f"No points data found in file: {path}")
        
        row = points_df.iloc[0]
        points = get_corners_from_angle(
            row['BX'], row['BY'], row['Width'], row['Height'], row['Angle']
        )
        return points

    def __load_target_from_path(self, target :str, path: str):
        ''''
        Load target data from the given path based on the target type.
        
        Args:
            target (str): The type of target to load ('mask' or 'points').
            path (str): The file path to load the target from.

        Returns:
            target (dict):
                A dictionary containing the loaded target data.
        '''

        root_dir = get_project_root_directory()
        path = path.replace("..\\", "").replace("../", "")
        relative_path = os.path.join(root_dir, path)

        target_output = {}
        for target_i in target.split('+'):
            if target_i == 'mask':
                target_output[target_i] = self.__load_mask_from_path(relative_path)
            
            elif target_i == 'points':
                target_output[target_i] = self.__load_points_from_path(relative_path)
            
            elif target_i == 'roi':
                points = self.__load_points_from_path(relative_path)
                original_dim = self.__current_image_original_dim    
                target_output[target_i] = generate_roi_from_points(points, original_dim[0], original_dim[1]) 

            elif target_i == 'heatmap':
                points = self.__load_points_from_path(relative_path)
                original_dim = self.__current_image_original_dim
                target_output[target_i] = generate_heatmap_from_points(points, original_dim, self.sigma)

            else:
                raise ValueError("Target must be either 'mask', 'points', 'roi', 'heatmap' or a combination of them separated by '+'")
        
        return target_output

    def __getitem__(self, idx):
        row = self.video_frame_df.iloc[idx]
        
        target_dir = row.target_dir
        frame_id = int(row.frame_id)
        frame_path = row.frame_path

        root = get_project_root_directory()
        frame_path = frame_path.replace("..\\", "").replace("../", "")
        frame_path_new = os.path.join(root, frame_path)
        image = np.array(Image.open(frame_path_new).convert("RGB"))
        
        self.__current_image_original_dim = (image.shape[0], image.shape[1])

        target = self.__load_target_from_path(self.target, target_dir) 
       
        if self.transform:
            image = T.ToTensor()(image)
            image = self.transform(image)
        
        if self.target_transform:
            for target_i in self.target.split('+'):
                target[target_i] = T.ToTensor()(target[target_i])
                if isinstance(self.target_transform, T.Resize) and target_i != 'points':
                    target[target_i] = self.target_transform(target[target_i])
        
        meta = {
            'frame_id': frame_id,
            'video_id': int(row.video_id),
            'paciente_id': row.paciente_id,
            'momento': row.momento,
            'procedimento': row.procedimento,
            'selected_labeler': row.selected_labeler,
            'frame_path': frame_path
        }
        
        return image, target, meta