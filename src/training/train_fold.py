import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Callable

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator
from src.training.checkpoint_manager import CheckpointManager
from src.training.train_epoch import train_one_epoch
from src.training.validate import validate

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    fold: int,
) -> Tuple[nn.Module, List[Dict]]:
    """Treina um fold completo com early stopping e checkpoints"""

    # Configuração
    optimizer = config.optimizer(model.parameters(),
                                 lr=config.learning_rate,
                                 **config.optimizer_kwargs)
    scheduler = config.scheduler(
        optimizer,
        **config.scheduler_kwargs
    )

    loss_calculator = LossCalculator(config.criterion_roi, config.criterion_heatmap, config)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir, fold)

    # Tracking
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(config.epochs):

        # Treino
        train_loss, train_components = train_one_epoch(
            model, train_loader, loss_calculator, optimizer,
            config, scaler = scaler
        )

        # Validação
        val_loss, val_components = validate(
            model, val_loader, loss_calculator, config
        )

        # Scheduler
        scheduler.step(val_loss)

        # Logging
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_components': train_components,
            'val_components': val_components,
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(epoch_info)

        print(f"Época {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Components - ROI: {val_components['roi']:.4f}, "
              f"Heatmap: {val_components['heatmap']:.4f}, "
              f"Penalty: {val_components['penalty']:.4f}")

        # Scheduler Early Stopping (Learning Rate)
        if optimizer.param_groups[0]["lr"] <= config.lr_patience:
            print("LR mínima atingida, parando.")
            break

        # Checkpoint e Early Stopping
        is_best = val_loss < best_val_loss
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, train_loss, is_best
        )

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping ativado na época {epoch+1}")
                break

    # Carrega melhor modelo
    checkpoint_manager.load_checkpoint(model, optimizer, best=True)

    return model, history