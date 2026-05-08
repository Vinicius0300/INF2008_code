import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class CheckpointManager:
    """Gerencia salvamento e carregamento de checkpoints"""
    def __init__(self, checkpoint_dir: Path, fold: int):
        self.checkpoint_dir = checkpoint_dir
        self.fold = fold
        self.best_loss = float('inf')

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_loss: float,
        train_loss: float,
        is_best: bool = False
    ):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'fold': self.fold
        }

        # Salva checkpoint da última época
        last_path = self.checkpoint_dir / f"fold_{self.fold}_last.pth"
        torch.save(checkpoint, last_path)

        # Salva melhor modelo
        if is_best:
            best_path = self.checkpoint_dir / f"fold_{self.fold}_best.pth"
            torch.save(checkpoint, best_path)
            self.best_loss = val_loss
            print(f"✓ Melhor modelo salvo (val_loss: {val_loss:.4f})")

    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, best: bool = True):
        """Carrega checkpoint"""
        suffix = "best" if best else "last"
        checkpoint_path = self.checkpoint_dir / f"fold_{self.fold}_{suffix}.pth"

        if not checkpoint_path.exists():
            print(f"Checkpoint não encontrado: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✓ Checkpoint carregado: epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f}")
        return checkpoint