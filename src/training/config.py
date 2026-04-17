import torch
import torch.nn as nn
from pathlib import Path
from src.utils import (modify_input,
                       custom_collate_fn)
from typing import Callable
import albumentations as A


class TrainingConfig:
    """Centraliza configurações de treinamento."""

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        dataset_class,
        
        sigma_heatmap: int = 20,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        epochs: int = 200,
        patience: int = 5,
        lr_patience: float = 1e-10,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        
        output_dim: tuple[int] = (256, 256),
        modify_input_fn: Callable|None = modify_input,
        collate_fn: Callable|None = custom_collate_fn,
        transform_train: A.Compose | None = None,
        transform_validation: A.Compose | None = None,

        criterion_roi=None,       
        criterion_heatmap=None,   
        weight_roi: float = 0.4,        # 0.4
        weight_heatmap: float = 0.4,    # 0.5
        weight_penalty: float = 0.2,    # 0.1

        offline_augmentation: bool = False,
        transform_augmentation: A.Compose | None = None,
        augmentation_dir: str|bool = False,
        n_aug: int = 5,

        checkpoint_dir: str = "./checkpoints/unet",
        device: str = "cuda",
    ):
        # Relacionandos ao modelo
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.scheduler = scheduler
        
        # Métodos ou Aplicações ou Transformações
        self.dataset_class = dataset_class
        self.sigma_heatmap = sigma_heatmap

        self.modify_input_fn = modify_input_fn
        self.collate_fn = collate_fn
        self.output_dim = output_dim
        self.transform_train = transform_train
        self.transform_validation = transform_validation
        
        # Instancia uma nova loss por Config — evita compartilhamento de estado
        self.criterion_roi      = criterion_roi      if criterion_roi      is not None else nn.BCELoss()
        self.criterion_heatmap  = criterion_heatmap  if criterion_heatmap  is not None else nn.MSELoss()
        self.weight_roi = weight_roi
        self.weight_heatmap = weight_heatmap
        self.weight_penalty = weight_penalty

        # Relacionando a augmentation:
        self.offline_augmentation = offline_augmentation
        self.augmentation_dir = augmentation_dir
        self.transform_augmentation = transform_augmentation
        self.n_aug = n_aug

        # Outros
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        