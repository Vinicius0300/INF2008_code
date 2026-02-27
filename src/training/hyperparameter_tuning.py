import pandas as pd
import torch
from typing import Dict, List, Tuple, Callable

import optuna

from src.training.config import TrainingConfig
from src.training.cross_validation import cross_validate




def objective_optuna(
    trial: optuna.Trial,
    model_class,
    folds: List[pd.DataFrame],
    target: str,
    output_dim: Tuple[int, int],
    modify_input_fn: Callable,
    dataset_class,
    sigma: float,
    collate_fn: Callable,
    device: str,
    epochs: int,
    model_kwargs: Dict = None
) -> float:
    """Função objetivo para otimização Optuna"""
    
    # Hiperparâmetros a serem otimizados
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    weight_roi = trial.suggest_float("weight_roi", 0.2, 0.6)
    weight_heatmap = trial.suggest_float("weight_heatmap", 0.3, 0.7)
    weight_penalty = trial.suggest_float("weight_penalty", 0.05, 0.2)
    
    # Configuração com hiperparâmetros sugeridos
    config = TrainingConfig(
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,  # Reduzido para Optuna
        patience=10,
        weight_roi=weight_roi,
        weight_heatmap=weight_heatmap,
        weight_penalty=weight_penalty,
        checkpoint_dir=f"./optuna_trials/trial_{trial.number}",
        device=device
    )
    
    # Executa cross-validation
    results = cross_validate(
        model_class, folds, config, target, output_dim,
        modify_input_fn, dataset_class, sigma, collate_fn, model_kwargs
    )

    # Limpa memória após cada trial
    torch.cuda.empty_cache()
    
    # Pruning - reporta loss intermediária para parar trials ruins
    # Optuna pode cancelar trials que claramente não vão dar certo
    trial.report(results['mean_loss'], step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return results['mean_loss']


def tune_hyperparameters(
    model_class,
    folds: List[pd.DataFrame],
    target: str,
    output_dim: Tuple[int, int],
    modify_input_fn: Callable,
    dataset_class,
    sigma: float,
    collate_fn: Callable,
    device: str,
    epochs: int = 50,
    n_trials: int = 20,
    model_kwargs: Dict = None
) -> optuna.Study:
    """Tunagem de hiperparâmetros com Optuna"""
    
    study = optuna.create_study(
        direction="minimize",
        study_name="unet_hyperparameter_tuning"
    )
    
    study.optimize(
        lambda trial: objective_optuna(
            trial, model_class, folds, target, output_dim,
            modify_input_fn, dataset_class, sigma, collate_fn, device, epochs, model_kwargs
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*50)
    print("MELHORES HIPERPARÂMETROS")
    print("="*50)
    print(f"Melhor Loss: {study.best_value:.4f}")
    print("Parâmetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study