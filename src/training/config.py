import os
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Callable
import albumentations as A
from dataclasses import dataclass, field

from src.split_data import split_data_k_fold
from src.utils import (modify_input,
                       custom_collate_fn,
                       get_project_root_directory,
                       resolve_dataframe_path,
                       load_points)

@dataclass
class TrainingConfig:
    """Centraliza configurações de treinamento."""

    path_dataframe: str

    model_class: Callable
    model_kwargs: dict
    model_name: str
    test_size: float
    n_folds: int
    dataset_class: Callable

    optimizer_kwargs: dict
    optimizer: Callable = torch.optim.Adam
    sigma_heatmap: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 8
    epochs: int = 200
    patience: int = 5
    lr_patience: float = 1e-10
    scheduler: Callable = torch.optim.lr_scheduler.ReduceLROnPlateau

    output_dim: tuple[int] = (256, 256)
    modify_input_fn: Callable|None = modify_input
    collate_fn: Callable|None = custom_collate_fn
    transform_train: A.Compose | None = None
    transform_validation: A.Compose | None = None

    criterion_roi: Callable = None
    criterion_heatmap: Callable = None
    weight_roi: float = 0.4        # 0.4
    weight_heatmap: float = 0.4    # 0.5
    weight_penalty: float = 0.2    # 0.1

    offline_augmentation: bool = False
    transform_augmentation: A.Compose | None = None
    augmentation_dir: str|bool = False
    n_aug: int = 5

    checkpoint_dir: str = "./checkpoints/unet"
    device: str = "cuda"
    predict_roi: bool = True

    # Definidos no __post_init__
    list_df_folds: list[pd.DataFrame] = field(init=False)
    df_test: pd.DataFrame = field(init=False)


    def __post_init__(self):

        # Instancia uma nova loss por Config — evita compartilhamento de estado
        self.criterion_roi      = self.criterion_roi      if self.criterion_roi      is not None else nn.BCELoss()
        self.criterion_heatmap  = self.criterion_heatmap  if self.criterion_heatmap  is not None else nn.MSELoss()

        # Outros
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        root = get_project_root_directory()
        final_path_dataframe = os.path.join(root, self.path_dataframe)
        video_frame_df = pd.read_csv(final_path_dataframe)
        video_frame_df = video_frame_df.apply(resolve_dataframe_path, axis = 1)
        video_frame_df["keypoints"] = video_frame_df["target_dir"].apply(load_points)

        self.list_df_folds, self.df_test = split_data_k_fold(video_frame_df, test_size=self.test_size, n_folds=self.n_folds)
