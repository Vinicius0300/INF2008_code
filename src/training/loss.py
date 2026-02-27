import torch
import torch.nn as nn

from src.training.config import TrainingConfig

class FocalMSELoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # L2 básico
        mse = (pred - target) ** 2
        
        # fator de foco baseado na região do pico
        focal_weight = 1 + self.alpha * (target ** self.gamma)
        
        # aplica foco
        loss = focal_weight * mse
        
        return loss.mean()

class FocalMSEMaskedLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0, threshold=1e-3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, pred, target):
        # L2 básico
        mse = (pred - target) ** 2
        
        # Focal: dá mais peso onde o GT é alto
        focal_weight = 1 + self.alpha * (target ** self.gamma)
        focal_mse = focal_weight * mse
        
        # MÁSCARA: só considera o que é relevante
        mask = (target > self.threshold).float()
        
        # Aplica máscara e evita divisões por zero
        masked_loss = (focal_mse * mask)
        
        if mask.sum() == 0:
            return focal_mse.mean()   # fallback
            
        return masked_loss.sum() / mask.sum()
    
class LossCalculator:
    """Centraliza o cálculo de perdas"""
    def __init__(self, criterion_roi, criterion_heatmap, config: TrainingConfig):
        self.criterion_roi = criterion_roi
        self.criterion_heatmap = criterion_heatmap
        self.config = config
    
    def calculate_loss(self, pred_roi, pred_heatmap, gt_roi, gt_heatmap):
        """Calcula perda combinada"""
        loss_roi = self.criterion_roi(pred_roi, gt_roi)
        loss_heatmap = self.criterion_heatmap(pred_heatmap, gt_heatmap)
        mask_penalty = torch.mean(pred_heatmap * (1 - gt_roi))
        
        loss_total = (
            self.config.weight_roi * loss_roi +
            self.config.weight_heatmap * loss_heatmap +
            self.config.weight_penalty * mask_penalty
        )
        
        return loss_total, {
            'roi': loss_roi.item(),
            'heatmap': loss_heatmap.item(),
            'penalty': mask_penalty.item()
        }