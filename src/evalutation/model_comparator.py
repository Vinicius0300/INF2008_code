'''Comparador de modelos'''
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.evalutation.inference import evaluate_model_on_test

class ModelComparator:
    def __init__(self, dict_config):
        self.dict_config = dict_config
        self.n_models = len(dict_config.keys())
        self.results = {}
        self._load_results()

    def _load_results(self):
        for model in self.dict_config:
            config = self.dict_config[model]
            root = config.checkpoint_dir
            pattern = re.compile(r"fold_([1-9])_best\.pth$")
            if os.path.exists(root):
                for filename in os.listdir(root):
                    match = pattern.search(filename)
                    if match:
                        checkpoint_path = os.path.join(root, filename)
                        results = evaluate_model_on_test(config, checkpoint_path, metric_path, "", False)
                        self.results[checkpoint_path] = results

    def calculate_metrics(self):
        rows = []
        for model in self.results:
            row = {}

            # Distância Euclidiana
            row["mean_distance_C2"] = self.results[model]["keypoint_distances"][0].mean()
            row["median_distance_C2"] = np.median(self.results[model]["keypoint_distances"][0])
            row["mean_distance_C4"] = self.results[model]["keypoint_distances"][1].mean()
            row["median_distance_C2"] = np.median(self.results[model]["keypoint_distances"][1])

            # Loss Heatmap
            row["mean_heatmap_loss"] = self.results[model]["heatmap_losses"].mean()
            row["median_heatmap_loss"] = np.median(self.results[model]["heatmap_losses"])

            # Loss Roi
            row["mean_roi_loss"] = self.results[model]["roi_losses"].mean()
            row["median_roi_loss"] = np.median(self.results[model]["roi_losses"])

            # Loss Geral
            row["mean_total_loss"] = self.results[model]["total_losses"].mean()
            row["median_total_loss"] = np.median(self.results[model]["total_losses"])

            rows.append(row)

        dfMetrics = pd.DataFrame(rows)
        dfMetrics.index = self.results.keys()
        return dfMetrics

    def plot_boxplot_metrics(self, cols):
        formatted_labels = [self.format_model_label(p) for p in self.results.keys()]

        # Configurações de estilo
        box_color = '#A8DADC'      # Azul claro para a caixa
        median_color = '#E63946'   # Vermelho vibrante para a mediana
        whisker_color = '#457B9D'  # Azul escuro para os "bigodes"

        for col in cols:
            if col == "keypoint_distances":
                points_c2 = []
                points_c4 = []
                for model in self.results:
                    points_c2.append(self.results[model][col][0])
                    points_c4.append(self.results[model][col][1])

                fig, ax = plt.subplots(1, 2, figsize=(16, 4)) # Aumentei um pouco a altura para os labels longos
                data_list = [points_c2, points_c4]
                titles = ["Boxplot - Distância Euclidiana - C2", "Boxplot - Distância Euclidiana - C4"]
                for i, data in enumerate(data_list):
                    bp = ax[i].boxplot(data, labels=formatted_labels, vert=False, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor(box_color)
                        patch.set_edgecolor(whisker_color)
                    for median in bp['medians']:
                        median.set(color=median_color, linewidth=2)
                    for whisker in bp['whiskers']:
                        whisker.set(color=whisker_color, linewidth=1.5, linestyle="--")

                    # Título e ajustes de eixos
                    ax[i].set_title(titles[i], fontsize=14)
                    ax[i].set_xlabel("Distância Euclidiana")
                    ax[i].tick_params(axis='y', labelsize=9) # ha='right' garante que o texto longo não sobreponha o gráfico
                    ax[i].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            else:
                model_metric = []
                for model in self.results:
                    model_metric.append(self.results[model][col])

                plt.figure(figsize=(8, 4))
                bp = plt.boxplot(model_metric, labels=formatted_labels, vert=False, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(box_color)
                    patch.set_edgecolor(whisker_color)
                for median in bp['medians']:
                    median.set(color=median_color, linewidth=2)
                for whisker in bp['whiskers']:
                    whisker.set(color=whisker_color, linewidth=1.5, linestyle="--")

                # Adicionando o grid com alpha baixo
                plt.grid(True, alpha=0.3)
                plt.title(f"Boxplot - {col}", fontsize=14, pad=15)
                plt.xlabel("Valor da Métrica")
                plt.tight_layout()
                plt.show()
        return

    def percentage_correct_keypoints(self, model, threshold):
        distances_C2 = self.results[model]["keypoint_distances"][0]
        distances_C4 = self.results[model]["keypoint_distances"][1]
        certos_C2 = 0
        certos_C4 = 0
        for x in distances_C2:
            if x <= threshold:
                certos_C2 += 1
        for x in distances_C4:
            if x <= threshold:
                certos_C4 += 1
        return certos_C2/len(distances_C2), certos_C4/len(distances_C4)

    def plot_percentage_correct_keypoint(self, threshold_min = 0.0, threshold_max = 20.0):
        # Calculando Porcentagens
        dict_percentage_C2 = {model: [] for model in self.results.keys()}
        dict_percentage_C4 = {model: [] for model in self.results.keys()}
        threshold_list = np.arange(threshold_min, threshold_max, 0.1)
        for threshold in threshold_list:
            for model in self.results:
                perc_C2, perc_C4 = self.percentage_correct_keypoints(model, threshold)
                dict_percentage_C2[model].append(perc_C2)
                dict_percentage_C4[model].append(perc_C4)

        # Visualizando Porcentagens
        fig, ax = plt.subplots(2,1, figsize = (8,6))
        cmap = plt.get_cmap('tab10')
        data_list = [dict_percentage_C2, dict_percentage_C4]
        titles = ["Percentual de Pontos Corretos - C2", "Percentual de Pontos Corretos - C4"]
        for i, data in enumerate(data_list):
            for idx, model in enumerate(self.results):
                ax[i].plot(
                    threshold_list,
                    data[model],
                    label=self.format_model_label(model), # Sua função de label bonita
                    color=cmap(idx),
                    linewidth=2
                )
            # Estilização do Gráfico
            ax[i].set_title(titles[i], fontsize=12)
            ax[i].set_xlabel("Limiar de Erro (Pixels / Distância)")
            ax[i].set_ylabel("Porcentagem de Acerto (%)")
            ax[i].set_ylim(-0.05, 1.05) # Garante que o eixo Y vá de 0 a 100%
            ax[i].grid(True, alpha=0.3)

            # Legenda fora do gráfico para não tampar as linhas
            ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.show()
        return

    @staticmethod
    def format_model_label(path):
        # Limpar e quebrar o caminho
        clean_path = path.replace('data\\model_weights\\', '').replace('.pth', '')
        parts = clean_path.split('\\')

        # Extrair componentes (ex: UNet_ImageNet)
        arch = parts[0].replace('_', ' ')

        # Detalhes do treino (ex: FocalMSEMaskedLoss_200ep)
        details = parts[1].split('_')
        loss = details[0].replace('Loss', ' Loss')
        epochs = details[1]

        # Se houver largura (ex: 32W)
        width = ""
        if len(details) > 2:
            width = f", Width {details[2].replace('W', '')}"

        # 2. Montar a string com quebras de linha (\n)
        label = (
            f"Architecture: {arch}\n"
            f"Loss: {loss}\n"
            f"Model Details: {epochs}{width}\n"
            #f"Model ID: {path}" # Mantendo o ID original para referência
        )
        return label
