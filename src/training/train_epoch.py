from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Callable

from src.training.loss import LossCalculator


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_calculator: LossCalculator,
    optimizer: optim.Optimizer,
    device: str,
    modify_input_fn: Callable,
    accumulation_steps: int = 2,
    scaler: torch.cuda.amp.GradScaler = None
) -> Tuple[float, Dict]:
    
    """Treina uma época com gradient accumulation"""

    model.train()
    running_loss = 0.0
    loss_components = {'roi': 0.0, 'heatmap': 0.0, 'penalty': 0.0}

    # Zero grad no início
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_loader, desc="Treinando")):

        # Mover para GPU
        inputs = inputs.to(device, non_blocking=True)
        inputs = modify_input_fn(inputs)
        gt_heatmap = targets['heatmap'].to(device, non_blocking=True)
        gt_roi = targets['roi'].to(device, non_blocking=True)
        
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
        running_loss += loss_total.item()
        for key in components:
            if key in loss_components:
                loss_components[key] += components[key]
            else:
                loss_components[key] = components[key]
        
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
        
        optimizer.zero_grad()
    
    # Calcular médias (proteger contra divisão por zero)
    n_batches = len(train_loader)
    if n_batches == 0:
        return 0.0, loss_components
    
    avg_loss = running_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components