import pandas as pd
from datetime import datetime
import os

from src.training.config import TrainingConfig


def att_test_control(val_type: str,
                     df_train: pd.DataFrame,
                     history: list[dict],
                     config: TrainingConfig):
    """Atualiza excel com os testes feitos.
    
    Args:
    val_type (str): Tipo de validação usada (Holdout ou Fold-{k}; k é o número do fold considerado)
    df_train (pd.DataFrame): Conjunto de treino usado
    history (list[dict]): Dados de treino de cada época
    config (TrainingConfig): Configurações do Teste"""

    # Localizar os componentes da MELHOR ÉPOCA dentro do history retornado
    val_losses_history = [epoch_data["val_loss"] for epoch_data in history]
    best_epoch_idx = val_losses_history.index(min(val_losses_history))
    best_epoch_data = history[best_epoch_idx]

    # Capturar transformações de aumento de dados (se houver) de forma amigável
    train_transforms = [t.__class__.__name__ for t in config.transform_train] if config.transform_train else ["None"]
    aug_status = "Offline" if config.offline_augmentation else "Online/Regular"

    excel_row_data = {
        # Metadados
        "ID Teste": [f"EXP-{int(datetime.now().timestamp())}"],
        "Data/Hora": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Nome do Modelo": [config.model_class.__name__],
        "Tipo de Validação": [val_type],
        
        # DNA do Dataset (Comparabilidade)
        "Dataset Base": [config.path_dataframe],
        "Qtd Imagens Treino": [len(df_train)],
        "Resolução Input": [f"{config.output_dim[0]}x{config.output_dim[1]}"],
        "Augmentation Tipo": [aug_status],
        "Transforms Aplicadas": [", ".join(train_transforms)],
        
        # Hiperparâmetros (Vindos da sua classe TrainingConfig)
        "Optimizer": [config.optimizer.__name__],
        "Learning Rate": [config.learning_rate],
        "Batch Size": [config.batch_size],
        "Epochs Máximas": [config.epochs],
        "Patience": [config.patience],
        "Pesos de Loss (ROI/Heat/Pen)": [f"{config.weight_roi} / {config.weight_heatmap} / {config.weight_penalty}"],
        
        # Melhores Métricas Obtidas (Baseadas na Época Selecionada)
        "Best Epoch": [best_epoch_data["epoch"]],
        "Final Val Loss Total": [best_epoch_data["val_loss"]],
        "Final Train Loss Total": [best_epoch_data["train_loss"]],
        "Best ROI Val Loss": [best_epoch_data["val_components"]["roi"]],
        "Best Heatmap Val Loss": [best_epoch_data["val_components"]["heatmap"]],
        "Best Penalty Val Loss": [best_epoch_data["val_components"]["penalty"]],
        
        # Rastreabilidade
        "Caminho Checkpoint": [os.path.abspath(config.checkpoint_dir)]
    }
    
    test_control_path = r"data\\model_weights\\test_control.xlsx"
    dfNewTest = pd.DataFrame(excel_row_data)
    if os.path.exists(test_control_path):
        dfTestControl = pd.read_excel(test_control_path)
        dfTestControl = pd.concat([dfTestControl, dfNewTest], axis = 0, ignore_index = True)
    else:
        dfTestControl = dfNewTest
    dfTestControl.to_excel(test_control_path, index=False)

    return 