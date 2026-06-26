import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Callable

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_calculator: LossCalculator,
    config: TrainingConfig
) -> Tuple[float, Dict]:

    """Valida o modelo"""
    model.eval()
    val_loss = torch.tensor(0.0, device=config.device)
    loss_components = {
        'roi': torch.tensor(0.0, device=config.device),
        'heatmap': torch.tensor(0.0, device=config.device),
        'penalty': torch.tensor(0.0, device=config.device)
    }

    with torch.no_grad():
        for inputs, keypoints, heatmaps, roi in val_loader:

            # Manda pra GPU
            inputs = inputs.to(config.device, non_blocking=True)
            if config.modify_input_fn != None:
                inputs = config.modify_input_fn(inputs)
            gt_heatmap = heatmaps.to(config.device)
            gt_roi = roi.to(config.device)
            gt_keypoints = keypoints.to(config.device)

            pred_roi, pred_heatmap, pred_keypoints = model(inputs)

            loss_total, components = loss_calculator.calculate_loss(
                pred_roi, pred_heatmap, pred_keypoints,
                gt_roi, gt_heatmap, gt_keypoints
            )

            val_loss += loss_total
            for key in loss_components:
                loss_components[key] += components[key]

    n_batches = len(val_loader)
    if n_batches == 0:
        return 0.0, {k: 0.0 for k in loss_components.keys()}

    avg_loss = (val_loss / n_batches).item()
    avg_components = {k: (v / n_batches).item() for k, v in loss_components.items()}
    return avg_loss, avg_components