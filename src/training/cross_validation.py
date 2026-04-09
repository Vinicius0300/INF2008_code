import pandas as pd
import json
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Callable

import albumentations as A

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator
from src.training.validate import validate
from src.training.train_fold import train_one_fold
from src.models.utils_model import init_weights
from src.offline_augmentation import generate_offline_dataset


def cross_validate(
    model_class,
    folds: List[pd.DataFrame],
    config: TrainingConfig,
    output_dim: Tuple[int, int],
    modify_input_fn: Callable,
    dataset_class,
    transform_augmentation: A.Compose | None,
    transform_train: A.Compose | None,
    transform_validation: A.Compose | None,
    sigma: float,
    collate_fn: Callable,
    offline_augmentation: bool = False,
    augmentation_dir: str = "data/frames_augmentation",
    model_kwargs: Dict = None
) -> Dict:
    """Executa validação cruzada K-Fold"""
    
    model_kwargs = model_kwargs or {}
    k = len(folds)
    results = {
        'fold_losses': [],
        'fold_histories': [],
        'best_fold': None,
        'best_loss': float('inf')
    }
    
    for i in range(k):
        print(f"\n{'='*50}")
        print(f"FOLD {i+1}/{k}")
        print(f"{'='*50}")
        
        # Preparação dos dados
        df_val = folds[i]
        df_train = pd.concat([folds[j] for j in range(k) if j != i], ignore_index=True)

        # Aplicação de Offline Augmentation (Caso Requisitado)
        if offline_augmentation:
            df_train = generate_offline_dataset(df_train,
                                                transform=transform_augmentation,
                                                save_dir=augmentation_dir,
                                                n_aug=5)
        
        train_set = dataset_class(df_train, output_dim, transform_train, sigma = sigma)
        val_set = dataset_class(df_val, output_dim, transform_validation, sigma = sigma)

        num_cores = min(8, os.cpu_count()//4 or 1)

        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_cores, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_set, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_cores, pin_memory=True, persistent_workers=True
        )
        
        # Novo modelo para cada fold
        model = model_class(**model_kwargs)
        model.apply(init_weights)
        model.to(config.device)
        print(config.device)
        
        # Treina fold
        model, history = train_one_fold(
            model, train_loader, val_loader, config, i+1, modify_input_fn
        )
        
        # Validação final
        final_val_loss, _ = validate(
            model, val_loader, 
            LossCalculator(config.criterion_roi, config.criterion_heatmap, config),
            config.device, modify_input_fn
        )
        
        results['fold_losses'].append(final_val_loss)
        results['fold_histories'].append(history)
        
        # Atualiza melhor fold
        if final_val_loss < results['best_loss']:
            results['best_loss'] = final_val_loss
            results['best_fold'] = i + 1
        
        print(f"\nFold {i+1} Loss Final: {final_val_loss:.4f}")
    
    # Estatísticas finais
    results['mean_loss'] = np.mean(results['fold_losses'])
    results['std_loss'] = np.std(results['fold_losses'])
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS FINAIS")
    print(f"{'='*50}")
    print(f"Média dos Folds: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
    print(f"Melhor Fold: {results['best_fold']} (Loss: {results['best_loss']:.4f})")
    
    # Salva resultados
    results_path = config.checkpoint_dir / "cross_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'fold_losses': results['fold_losses'],
            'mean_loss': results['mean_loss'],
            'std_loss': results['std_loss'],
            'best_fold': results['best_fold'],
            'best_loss': results['best_loss']
        }, f, indent=2)
    
    return results