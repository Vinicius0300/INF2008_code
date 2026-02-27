from pathlib import Path
from typing import Dict, Callable

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator
from src.evalutation.evaluation import TestEvaluator

def evaluate_model_on_test(
    model_class,
    model_kwargs: Dict,
    checkpoint_path: str,
    test_dataset,
    config: TrainingConfig,
    modify_input_fn: Callable,
    output_dir: str = "./evaluation_results"
):
    """
    Pipeline completo de avaliação no conjunto de teste
    
    Args:
        model_class: Classe do modelo
        model_kwargs: Argumentos do modelo
        checkpoint_path: Caminho do checkpoint a ser avaliado
        test_dataset: Dataset de teste
        config: Configuração de treinamento
        modify_input_fn: Função para modificar input
        output_dir: Diretório para salvar resultados
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("INICIANDO AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*70)
    
    # Inicializa avaliador
    evaluator = TestEvaluator(
        model_class=model_class,
        model_kwargs=model_kwargs,
        checkpoint_path=checkpoint_path,
        device=config.device,
        modify_input_fn=modify_input_fn
    )
    
    # Avalia conjunto de teste
    loss_calculator = LossCalculator(
        criterion_roi=config.criterion_roi,
        criterion_heatmap=config.criterion_heatmap,
        config=config
    )
    
    results = evaluator.evaluate_test_set(test_dataset, loss_calculator)
    
    # Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES")
    print("="*70)
    
    # 1. Distribuições de loss
    evaluator.plot_loss_distributions(
        results,
        save_path=str(output_path / "loss_distributions.png")
    )
    
    # 2. Análise por keypoint
    evaluator.plot_keypoint_analysis(
        results,
        save_path=str(output_path / "keypoint_analysis.png")
    )
    
    # 3. Visualizações de predições (por distância)
    evaluator.visualize_predictions(
        results,
        metric_type='distance',
        save_path=str(output_path / "predictions_by_distance")
    )
    
    # 4. Visualizações de predições (por total loss)
    evaluator.visualize_predictions(
        results,
        metric_type='total_loss',
        save_path=str(output_path / "predictions_by_loss")
    )
    
    # 5. Gera relatório
    evaluator.generate_report(
        results,
        save_path=str(output_path / "evaluation_report.txt")
    )
    
    print("\n" + "="*70)
    print("AVALIAÇÃO CONCLUÍDA!")
    print(f"Resultados salvos em: {output_path}")
    print("="*70)
    
    return results