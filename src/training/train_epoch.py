from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Callable

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_calculator: LossCalculator,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    accumulation_steps: int = 2,
    scaler: torch.cuda.amp.GradScaler = None
) -> Tuple[float, Dict]:

    """Treina uma época com gradient accumulation"""

    model.train()
    running_loss = torch.tensor(0.0, device=config.device)
    loss_components = {'roi': torch.tensor(0.0, device=config.device),
                       'heatmap': torch.tensor(0.0, device=config.device),
                       'penalty': torch.tensor(0.0, device=config.device)}

    # Zero grad no início
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, keypoints, heatmaps, roi) in enumerate(tqdm(train_loader, desc="Treinando", mininterval=0.5)):

        # Mover para GPU
        inputs = inputs.to(config.device, non_blocking=True)
        if config.modify_input_fn != None:
            inputs = config.modify_input_fn(inputs)
        gt_heatmap = heatmaps.to(config.device, non_blocking=True)
        gt_roi = roi.to(config.device, non_blocking=True)

        # Forward pass com ou sem mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred_roi, pred_heatmap = model(inputs)
                loss_total, components = loss_calculator.calculate_loss(
                    pred_roi, pred_heatmap, gt_roi, gt_heatmap
                )

            # Dividir loss pela acumulação
            loss_scaled = loss_total / accumulation_steps

            # Backward com scaler
            scaler.scale(loss_scaled).backward()

        else:
            pred_roi, pred_heatmap = model(inputs)
            loss_total, components = loss_calculator.calculate_loss(
                pred_roi, pred_heatmap, gt_roi, gt_heatmap
            )

            loss_scaled = loss_total / accumulation_steps
            loss_scaled.backward()

        # Acumular stats (usar loss ORIGINAL, não scaled)
        running_loss += loss_total.detach()
        for key in components:
            if key in loss_components:
                loss_components[key] += components[key].detach()
            else:
                loss_components[key] = components[key].detach()

        # Atualizar pesos a cada N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

    # Apenas se houver gradientes não processados - últimos gradientes calculados
    if len(train_loader) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    # Calcular médias (proteger contra divisão por zero)
    n_batches = len(train_loader)
    if n_batches == 0:
        return 0.0, {k: 0.0 for k in loss_components.keys()}

    avg_loss = (running_loss / n_batches).item()
    avg_components = {k: (v / n_batches).item() for k, v in loss_components.items()}

    return avg_loss, avg_components