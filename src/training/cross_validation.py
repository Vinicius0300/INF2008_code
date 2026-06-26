import pandas as pd
import json
import numpy as np
import os
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Callable

import albumentations as A

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator
from src.training.validate import validate
from src.training.train_fold import train_one_fold
from src.models.utils_model import init_weights
from src.offline_augmentation import generate_offline_dataset
from src.training.test_control import att_test_control


def cross_validate(
    config: TrainingConfig
) -> Dict:
    """Executa validação cruzada K-Fold"""

    model_kwargs = config.model_kwargs or {}
    k = config.n_folds
    results = {
        'fold_losses': [],
        'fold_histories': [],
        'best_fold': None,
        'best_loss': float('inf')
    }

    # Aplicação de Offline Augmentation (Caso Requisitado)
    if config.offline_augmentation:
        folds_aug = generate_offline_dataset(config.list_df_folds,
                                         transform=config.transform_augmentation,
                                         save_dir=config.augmentation_dir,
                                         n_aug=config.n_aug)

    for i in range(k):
        print(f"\n{'='*50}")
        print(f"FOLD {i+1}/{k}")
        print(f"{'='*50}")

        # Preparação dos dados
        df_val = config.list_df_folds[i]
        if config.offline_augmentation:
            df_train = pd.concat([folds_aug[j] for j in range(k) if j != i], ignore_index=True)
        else:
            df_train = pd.concat([config.list_df_folds[j] for j in range(k) if j != i], ignore_index=True)

        train_set = config.dataset_class(df_train,
                                         config.output_dim,
                                         config.transform_train,
                                         sigma_heatmap = config.sigma_heatmap)
        val_set = config.dataset_class(df_val,
                                       config.output_dim,
                                       config.transform_validation,
                                       sigma_heatmap = config.sigma_heatmap)

        num_cores = min(8, os.cpu_count()//4 or 1)

        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True,
            collate_fn=config.collate_fn, num_workers=num_cores, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_set, batch_size=config.batch_size, shuffle=False,
            collate_fn=config.collate_fn, num_workers=num_cores, pin_memory=True, persistent_workers=True
        )

        # Novo modelo para cada fold
        model = config.model_class(**model_kwargs)
        model.apply(init_weights)
        model.to(config.device)

        # Treina fold
        model, history = train_one_fold(
            model, train_loader, val_loader, config, i+1
        )

        # Validação final
        final_val_loss, _ = validate(
            model, val_loader,
            LossCalculator(config),
            config
        )

        results['fold_losses'].append(final_val_loss)
        results['fold_histories'].append(history)

        # Atualiza melhor fold
        if final_val_loss < results['best_loss']:
            results['best_loss'] = final_val_loss
            results['best_fold'] = i + 1

        print(f"\nFold {i+1} Loss Final: {final_val_loss:.4f}")

        # Salva o teste feito no Controle de Testes
        att_test_control(f"Fold {i+1}", df_train, history, config)

    # Estatísticas finais
    results['mean_loss'] = np.mean(results['fold_losses'])
    results['std_loss'] = np.std(results['fold_losses'])

    print(f"\n{'='*50}")
    print(f"RESULTADOS FINAIS")
    print(f"{'='*50}")
    print(f"Média dos Folds: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
    print(f"Melhor Fold: {results['best_fold']} (Loss: {results['best_loss']:.4f})")

    # Salva resultados
    results_path = config.checkpoint_dir / "metrics_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'fold_histories': results['fold_histories'],
            'fold_losses': results['fold_losses'],
            'mean_loss': results['mean_loss'],
            'std_loss': results['std_loss'],
            'best_fold': results['best_fold'],
            'best_loss': results['best_loss']
        }, f, indent=2)

    return results