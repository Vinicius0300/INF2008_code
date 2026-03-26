import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Callable

from src.training.loss import LossCalculator

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_calculator: LossCalculator,
    device: str,
    modify_input_fn: Callable
) -> Tuple[float, Dict]:
    
    """Valida o modelo"""
    model.eval()
    val_loss = 0.0
    loss_components = {'roi': 0.0, 'heatmap': 0.0, 'penalty': 0.0}
    
    with torch.no_grad():
        for inputs, keypoints, heatmaps, roi in val_loader:

            # Manda pra GPU
            inputs = inputs.to(device, non_blocking=True)
            inputs = modify_input_fn(inputs)
            gt_heatmap = heatmaps.to(device)
            gt_roi = roi.to(device)
            
            pred_roi, pred_heatmap = model(inputs)
            
            loss_total, components = loss_calculator.calculate_loss(
                pred_roi, pred_heatmap, gt_roi, gt_heatmap
            )
            
            val_loss += loss_total.item()
            for key in loss_components:
                loss_components[key] += components[key]
    
    n_batches = len(val_loader)
    avg_loss = val_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components