import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.config import TrainingConfig

class FocalMSELoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # F.mse_loss com reduction='none' é mais otimizado na VRAM
        mse = F.mse_loss(pred, target, reduction='none')

        # O peso focal só depende do target (que não precisa de gradientes!)
        with torch.no_grad():
            focal_weight = 1.0 + self.alpha * (target ** self.gamma)

        return (focal_weight * mse).mean()

class FocalMSEMaskedLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0, threshold=1e-3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none')

        # Isola os cálculos do GT para poupar memória do Backward Pass
        with torch.no_grad():
            focal_weight = 1.0 + self.alpha * (target ** self.gamma)
            mask = (target > self.threshold).float()
            mask_sum = mask.sum().clamp(min=1.0)

        focal_mse = focal_weight * mse
        masked_loss = focal_mse * mask

        # Evita torch.where computando os dois galhos inteiros
        if mask_sum > 1.0: # clamp(min=1) significa que se era 0 virou 1
            return masked_loss.sum() / mask_sum
        else:
            return focal_mse.mean()

class LossCalculator:
    """Centraliza o cálculo de perdas"""
    def __init__(self, criterion_roi, criterion_heatmap, config: TrainingConfig):
        self.criterion_roi = criterion_roi
        self.criterion_heatmap = criterion_heatmap
        self.config = config

    def calculate_loss(self, pred_roi, pred_heatmap, gt_roi, gt_heatmap):
        if pred_roi == None:
            loss_roi = torch.tensor(0.0, device=gt_roi.device)
        else:
            loss_roi = self.criterion_roi(pred_roi, gt_roi)

        if pred_heatmap == None:
            loss_heatmap = torch.tensor(0.0, device=gt_heatmap.device)
            mask_penalty = torch.tensor(0.0, device=gt_heatmap.device)
        else:
            loss_heatmap = self.criterion_heatmap(pred_heatmap, gt_heatmap)

            # Avisa o PyTorch que (1 - gt_roi) é estático
            with torch.no_grad():
                inv_gt_roi = 1.0 - gt_roi

            mask_penalty = torch.mean(pred_heatmap * inv_gt_roi)

        loss_total = (
            self.config.weight_roi * loss_roi +
            self.config.weight_heatmap * loss_heatmap +
            self.config.weight_penalty * mask_penalty
        )

        return loss_total, {
            'roi': loss_roi,
            'heatmap': loss_heatmap,
            'penalty': mask_penalty
        }