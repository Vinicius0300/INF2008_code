import json
from pathlib import Path
from typing import Dict, Callable

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator
from src.evalutation.evaluation import TestEvaluator

def evaluate_model_on_test(
    config: TrainingConfig,
    checkpoint_path: str,
    metrics_path: str,
    fold_number: int = 1,
    output_dir: str = "",
    show_results: bool = True
):
    """
    Pipeline completo de avaliação no conjunto de teste

    Args:
        checkpoint_path: Caminho do checkpoint a ser avaliado
        output_dir: Diretório para salvar resultados
        config: Configuração de treinamento
    """
    test_dataset = config.dataset_class(
                video_frame_df = config.df_test,
                output_dim = config.output_dim,
                transform = config.transform_validation,      # Sempre sem augmentation para o Teste, então deve ser deterministico
                sigma_heatmap = config.sigma_heatmap,
                )

    if show_results:
        print("\n" + "="*70)
        print("INICIANDO AVALIAÇÃO NO CONJUNTO DE TESTE")
        print("="*70)

    # Inicializa avaliador
    evaluator = TestEvaluator(
        checkpoint_path=checkpoint_path,
        config=config
    )

    # Avalia conjunto de teste
    loss_calculator = LossCalculator(
        criterion_roi=config.criterion_roi,
        criterion_heatmap=config.criterion_heatmap,
        config=config
    )

    # Pega os dados de loss de todos os folds e todas as épocas:
    with open(metrics_path, "r", encoding = "utf-8") as file:
        loss_history = json.load(file)

    results = evaluator.evaluate_test_set(test_dataset, loss_calculator)
    results["loss_history"] = loss_history

    if show_results:

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Gera visualizações
        print("\n" + "="*70)
        print("GERANDO VISUALIZAÇÕES")
        print("="*70)

        # 1. Loss do treinamento
        evaluator.plot_loss_training(
            results["loss_history"],
            fold_number,
            save_path = str(output_path / "loss_history")
        )

        # 2. Distribuições de loss
        evaluator.plot_loss_distributions(
            results,
            save_path=str(output_path / "loss_distributions.png")
        )

        # 3. Análise por keypoint
        evaluator.plot_keypoint_analysis(
            results,
            save_path=str(output_path / "keypoint_analysis.png")
        )

        # 4. Visualizações de predições (por distância)
        evaluator.visualize_predictions(
            results,
            metric_type='distance',
            save_path=str(output_path / "predictions_by_distance")
        )

        # 5. Visualizações de predições (por total loss)
        evaluator.visualize_predictions(
            results,
            metric_type='total_loss',
            save_path=str(output_path / "predictions_by_loss")
        )

        # 6. Gera relatório
        evaluator.generate_report(
            results,
            save_path=str(output_path / "evaluation_report.txt")
        )

        print("\n" + "="*70)
        print("AVALIAÇÃO CONCLUÍDA!")
        print(f"Resultados salvos em: {output_path}")
        print("="*70)

    return results