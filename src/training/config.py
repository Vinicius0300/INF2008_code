import torch
import torch.nn as nn
from pathlib import Path

class TrainingConfig:
    """Centraliza configurações de treinamento."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        epochs: int = 200,
        patience: int = 5,
        lr_patience: float = 1e-10,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  
        criterion_roi=None,       
        criterion_heatmap=None,   
        weight_roi: float = 0.4,
        weight_heatmap: float = 0.4,
        weight_penalty: float = 0.2,
        checkpoint_dir: str = "./checkpoints/unet",
        device: str = "cuda",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.scheduler = scheduler

        # Instancia uma nova loss por Config — evita compartilhamento de estado
        self.criterion_roi      = criterion_roi      if criterion_roi      is not None else nn.BCELoss()
        self.criterion_heatmap  = criterion_heatmap  if criterion_heatmap  is not None else nn.MSELoss()

        self.weight_roi = weight_roi
        self.weight_heatmap = weight_heatmap
        self.weight_penalty = weight_penalty
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device