import os
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Callable
import albumentations as A
from dataclasses import dataclass, field
from datetime import datetime

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
    test_size: float
    n_folds: int
    dataset_class: Callable
    sigma_heatmap: int = 20

    epochs: int = 200
    learning_rate: float = 3e-4
    batch_size: int = 8
    patience: int = 5
    lr_patience: float = 1e-10
    optimizer: Callable = torch.optim.Adam
    optimizer_kwargs: dict = field(default_factory=dict)
    scheduler: Callable = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_kwargs: dict = field(default_factory=dict)

    offline_augmentation: bool = False
    transform_augmentation: A.Compose | None = None
    augmentation_dir: str|bool = False
    n_aug: int = 5

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

    device: str = "cuda"
    predict_roi: bool = True
    width: None|int = None

    # Definidos no __post_init__
    model_name: str = field(init=False)
    checkpoint_dir: str = field(init=False)
    list_df_folds: list[pd.DataFrame] = field(init=False)
    df_test: pd.DataFrame = field(init=False)
    id_experiment: int = field(init=False)


    def __post_init__(self):

        # Instancia uma nova loss por Config — evita compartilhamento de estado
        self.criterion_roi      = self.criterion_roi      if self.criterion_roi      is not None else nn.BCELoss()
        self.criterion_heatmap  = self.criterion_heatmap  if self.criterion_heatmap  is not None else nn.MSELoss()

        # Data frame - Conjunto de Teste e Folds
        root = get_project_root_directory()
        final_path_dataframe = os.path.join(root, self.path_dataframe)
        video_frame_df = pd.read_csv(final_path_dataframe)
        video_frame_df = video_frame_df.apply(resolve_dataframe_path, axis = 1)
        video_frame_df["keypoints"] = video_frame_df["target_dir"].apply(load_points)

        self.list_df_folds, self.df_test = split_data_k_fold(video_frame_df, test_size=self.test_size, n_folds=self.n_folds)
        
        # ID do experimento
        self.id_experiment = f"{int(datetime.now().timestamp())}"
        
        # Nome do modelo
        if self.width == None:
            self.model_name = f"{self.model_class.__name__}\\{self.criterion_heatmap.__class__.__name__}_{str(self.epochs)}ep\\{self.id_experiment}"
        else:
            self.model_name = f"{self.model_class.__name__}\\{self.criterion_heatmap.__class__.__name__}_{str(self.epochs)}ep_{str(self.width)}W\\{self.id_experiment}"

        # Checkpoint dos Modelos
        self.checkpoint_dir = f"data\\model_weights\\{self.model_name}"
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)