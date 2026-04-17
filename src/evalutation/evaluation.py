import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from typing import Dict, Callable
from scipy.stats import pearsonr

from src.training.config import TrainingConfig
from src.training.loss import LossCalculator

class TestEvaluator:
    """Avalia modelo no conjunto de teste e gera visualizações"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: TrainingConfig
    ):
        self.model_class = config.model_class
        self.model_kwargs = config.model_kwargs or {}
        self.device = config.device
        self.modify_input_fn = config.modify_input_fn
        
        # Carrega modelo
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Modelo carregado: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    def extract_keypoints_from_heatmap(self, heatmap: torch.Tensor, roi: torch.Tensor|None = None) -> torch.Tensor:
        """Extrai coordenadas dos keypoints do heatmap, opcionalmente filtrando por uma ROI"""
        num_keypoints = heatmap.shape[0]
        points = torch.zeros(num_keypoints, 2)
        
        # Transforma a ROI em binária
        if roi is not None:
            mask = (roi > 0).float().squeeze()
        else:
            mask = None

        # Calcula os pontos
        for k in range(num_keypoints):
            heatmap_k = heatmap[k] # Mantemos no PyTorch para ser mais rápido

            # Se houver máscara, aplicamos multiplicando ponto a ponto
            if mask is not None:
                # Heatmap apenas dentro da ROI. O que está fora vira 0.
                target_heatmap = heatmap_k * mask.to(heatmap_k.device)
            else:
                target_heatmap = heatmap_k

            # Convertendo para numpy apenas para o argmax
            target_np = target_heatmap.detach().cpu().numpy()
            
            # Encontra o índice do valor máximo
            idx = np.unravel_index(np.argmax(target_np), target_np.shape)
            points[k] = torch.tensor([idx[0], idx[1]])  # (y, x)
    
        return points
    
    def euclidean_distance(self, point1: torch.Tensor, point2: torch.Tensor) -> float:
        """Calcula distância euclidiana entre dois pontos"""
        return torch.sqrt(torch.sum((point1 - point2) ** 2)).item()
    
    def evaluate_test_set(
        self,
        test_dataset,
        loss_calculator: LossCalculator
    ) -> Dict:
        """Avalia modelo no conjunto de teste"""
        
        num_keypoints = self.model_kwargs['num_keypoints']
        
        results = {
            'keypoint_distances': [[] for _ in range(num_keypoints)],
            'heatmap_losses': [],
            'roi_losses': [],
            'total_losses': [],
            'predictions': [],
            'ground_truths': [],
            'images': []
        }
        
        print(f"\nAvaliando {len(test_dataset)} amostras do conjunto de teste...")
        
        for idx in tqdm(range(len(test_dataset)), desc="Avaliando"):
            input_img, keypoint, heatmap, roi = test_dataset[idx]
            
            # Preparação
            input_tensor = self.modify_input_fn(input_img).float().to(self.device)
            gt_roi = roi.float().to(self.device)
            gt_heatmap = heatmap.float().to(self.device)
            
            # Inferência
            with torch.no_grad():
                pred_roi, pred_heatmap = self.model(input_tensor)
            
            if pred_heatmap != None:
                pred_heatmap = pred_heatmap.squeeze(0)
            if pred_roi != None:
                pred_roi = pred_roi.squeeze(0)
            
            # Extrai pontos
            gt_points = self.extract_keypoints_from_heatmap(gt_heatmap)
            if pred_heatmap != None:
                pred_points = self.extract_keypoints_from_heatmap(pred_heatmap, pred_roi)
            else: 
                pred_points = torch.tensor([[0.0, 0.0], [0.0, 0.0]]) # Só pra ter alguma coisa, mas de fato nada é previsto.
            
            # Calcula distâncias por keypoint
            for k in range(num_keypoints):
                dist = self.euclidean_distance(gt_points[k], pred_points[k])
                results['keypoint_distances'][k].append(dist)
            
            # Calcula losses
            loss_total, components = loss_calculator.calculate_loss(
                pred_roi, pred_heatmap, gt_roi, gt_heatmap
            )
            
            results['heatmap_losses'].append(components['heatmap'])
            results['roi_losses'].append(components['roi'])
            results['total_losses'].append(loss_total.item())
            
            # Armazena para visualização
            results['predictions'].append({
                'points': pred_points,
                'heatmap': pred_heatmap.cpu() if pred_heatmap != None else None,
                'roi': pred_roi.cpu() if pred_roi != None else None
            })
            results['ground_truths'].append({
                'points': gt_points,
                'heatmap': gt_heatmap.cpu(),
                'roi': gt_roi.cpu()
            })
            results['images'].append(input_tensor.squeeze(0).cpu())
        
        # Converte para arrays
        results['keypoint_distances'] = [np.array(dist) for dist in results['keypoint_distances']]
        # Usamos list comprehension para garantir .cpu().numpy() em cada tensor da lista
        results['heatmap_losses'] = np.array([l.detach().cpu().item() if torch.is_tensor(l) else l for l in results['heatmap_losses']])
        results['roi_losses'] = np.array([l.detach().cpu().item() if torch.is_tensor(l) else l for l in results['roi_losses']])
        results['total_losses'] = np.array([l.detach().cpu().item() if torch.is_tensor(l) else l for l in results['total_losses']])
        
        return results
    
    def plot_loss_distributions(
        self,
        results: Dict,
        save_path: str = None
    ):
        """Plota histogramas das distribuições de loss"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribuição das Losses no Conjunto de Teste', fontsize=16, y=1.0)
        
        # 1. Heatmap Loss
        ax = axes[0, 0]
        ax.hist(results['heatmap_losses'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['heatmap_losses']), color='red', linestyle='--', 
                   label=f'Média: {np.mean(results["heatmap_losses"]):.4f}')
        ax.axvline(np.median(results['heatmap_losses']), color='green', linestyle='--',
                   label=f'Mediana: {np.median(results["heatmap_losses"]):.4f}')
        ax.set_xlabel('Heatmap Loss')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição: Heatmap Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Total Loss
        ax = axes[0, 1]
        ax.hist(results['total_losses'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['total_losses']), color='red', linestyle='--',
                   label=f'Média: {np.mean(results["total_losses"]):.4f}')
        ax.axvline(np.median(results['total_losses']), color='green', linestyle='--',
                   label=f'Mediana: {np.median(results["total_losses"]):.4f}')
        ax.set_xlabel('Total Loss')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição: Total Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Scatter: Heatmap vs Total Loss
        ax = axes[1, 0]
        ax.scatter(results['heatmap_losses'], results['total_losses'], 
                  alpha=0.5, s=20, color='purple')
        
        # Adiciona linha de tendência
        if min(results["heatmap_losses"]) != max(results["heatmap_losses"]):
            z = np.polyfit(results['heatmap_losses'], results['total_losses'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(results['heatmap_losses'].min(), results['heatmap_losses'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Tendência')
        
        # Correlação
        corr, _ = pearsonr(results['heatmap_losses'], results['total_losses'])
        ax.set_xlabel('Heatmap Loss')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'Heatmap Loss vs Total Loss (Corr: {corr:.3f})')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. ROI Loss
        ax = axes[1, 1]
        ax.hist(results['roi_losses'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['roi_losses']), color='red', linestyle='--',
                   label=f'Média: {np.mean(results["roi_losses"]):.4f}')
        ax.axvline(np.median(results['roi_losses']), color='green', linestyle='--',
                   label=f'Mediana: {np.median(results["roi_losses"]):.4f}')
        ax.set_xlabel('ROI Loss')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição: ROI Loss')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}")
        
        plt.show()
    
    def plot_keypoint_analysis(
        self,
        results: Dict,
        save_path: str = None
    ):
        """Plota análise detalhada por keypoint"""
        
        num_keypoints = len(results['keypoint_distances'])
        fig, axes = plt.subplots(1, num_keypoints, figsize=(6*num_keypoints, 5))
        
        if num_keypoints == 1:
            axes = [axes]
        
        fig.suptitle('Análise de Distâncias por Keypoint', fontsize=16)
        
        for k in range(num_keypoints):
            ax = axes[k]
            distances = results['keypoint_distances'][k]
            
            # Histograma
            ax.hist(distances, bins=30, color='orange', edgecolor='black', alpha=0.7)
            
            # Estatísticas
            mean_dist = np.mean(distances)
            median_dist = np.median(distances)
            std_dist = np.std(distances)
            
            ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
                      label=f'Média: {mean_dist:.2f}px')
            ax.axvline(median_dist, color='green', linestyle='--', linewidth=2,
                      label=f'Mediana: {median_dist:.2f}px')
            
            ax.set_xlabel('Distância Euclidiana (pixels)')
            ax.set_ylabel('Frequência')
            ax.set_title(f'Keypoint {k+1}\n(Std: {std_dist:.2f}px)')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}")
        
        plt.show()
    
    def visualize_predictions(
        self,
        results: Dict,
        metric_type: str = 'total_loss',
        save_path: str = None
    ):
        """Visualiza melhor, mediana e pior predição por keypoint"""
        
        num_keypoints = len(results['keypoint_distances'])
        
        for k in range(num_keypoints):
            print(f"\n--- Visualizações para Keypoint {k+1} ---")
            
            # Define métrica
            if metric_type == 'distance':
                metric = results['keypoint_distances'][k]
                metric_name = "Distância Euclidiana"
            elif metric_type == 'heatmap_loss':
                metric = results['heatmap_losses']
                metric_name = "Heatmap Loss"
            else:  # total_loss
                metric = results['total_losses']
                metric_name = "Total Loss"
            
            # Encontra índices
            idx_min = np.argmin(metric)
            idx_max = np.argmax(metric)
            idx_med = np.argsort(metric)[len(metric)//2]
            
            indices = [idx_min, idx_med, idx_max]
            values = [metric[idx_min], metric[idx_med], metric[idx_max]]
            titles = ["Melhor Caso", "Caso Mediano", "Pior Caso"]
            
            # Cria figura
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Keypoint {k+1} - Análise por {metric_name}', fontsize=16)
            
            for col, (idx, val, title) in enumerate(zip(indices, values, titles)):
                # Imagem original com pontos
                ax_img = axes[0, col]
                img = results['images'][idx][0].numpy()
                ax_img.imshow(img, cmap='gray')
                
                # Ground truth
                gt_point = results['ground_truths'][idx]['points'][k]
                ax_img.scatter(gt_point[1], gt_point[0], color='lime', marker='x', 
                              s=150, linewidths=3, label='Ground Truth')
                
                # Predição
                pred_point = results['predictions'][idx]['points'][k]
                ax_img.scatter(pred_point[1], pred_point[0], color='red', marker='o',
                              s=100, linewidths=2, label='Predição')
                
                # Linha conectando
                ax_img.plot([gt_point[1], pred_point[1]], [gt_point[0], pred_point[0]],
                           'yellow', linewidth=2, alpha=0.7)
                
                dist = self.euclidean_distance(gt_point, pred_point)
                ax_img.set_title(f'{title}\n{metric_name}: {val:.4f}\nDist: {dist:.2f}px')
                ax_img.axis('off')
                ax_img.legend(loc='upper right')
                
                # Heatmap e ROI predito
                ax_pred = axes[1, col]
                im = None
                pred_heatmap = results['predictions'][idx]['heatmap'][k].numpy() if results['predictions'][idx]['heatmap'] != None else None
                pred_roi = results['predictions'][idx]['roi'].squeeze().numpy() if results['predictions'][idx]['roi'] != None else None
                ground_truths_roi = results['ground_truths'][idx]['roi'].squeeze().numpy()
                if pred_heatmap is not None:
                    im = ax_pred.imshow(pred_heatmap, cmap='hot', vmin = 0, vmax = 1)
                if pred_roi is not None:
                    ax_pred.contour(pred_roi, levels=[0.5], colors='blue', linewidths=4)
                    roi_img = ax_pred.imshow(np.ma.masked_where(pred_roi < 0.5, pred_roi), 
                                   cmap='Greens', alpha=0.5, vmin=0, vmax=1)
                    if im is None: 
                        im = roi_img              
                ax_pred.scatter(pred_point[1], pred_point[0], color='cyan', marker='o',
                               s=100, linewidths=2)
                ax_pred.set_title('Heatmap + ROI Predita')
                ax_pred.axis('off')
                plt.colorbar(im, ax=ax_pred, fraction=0.046)
            
            plt.tight_layout()
            
            if save_path:
                path_final = f"{save_path}_keypoint{k+1}.png"
                plt.savefig(path_final, dpi=300, bbox_inches='tight')
                print(f"✓ Figura salva: {path_final}")
            
            plt.show()
    
    def generate_report(self, results: Dict, save_path: str = None):
        """Gera relatório textual completo"""
        
        report = []
        report.append("="*70)
        report.append("RELATÓRIO DE AVALIAÇÃO NO CONJUNTO DE TESTE")
        report.append("="*70)
        report.append(f"\nNúmero de amostras: {len(results['total_losses'])}")
        
        # Loss Total
        report.append("\n" + "-"*70)
        report.append("LOSS TOTAL")
        report.append("-"*70)
        report.append(f"Média:    {np.mean(results['total_losses']):.6f}")
        report.append(f"Mediana:  {np.median(results['total_losses']):.6f}")
        report.append(f"Std:      {np.std(results['total_losses']):.6f}")
        report.append(f"Min:      {np.min(results['total_losses']):.6f}")
        report.append(f"Max:      {np.max(results['total_losses']):.6f}")
        
        # Heatmap Loss
        report.append("\n" + "-"*70)
        report.append("HEATMAP LOSS")
        report.append("-"*70)
        report.append(f"Média:    {np.mean(results['heatmap_losses']):.6f}")
        report.append(f"Mediana:  {np.median(results['heatmap_losses']):.6f}")
        report.append(f"Std:      {np.std(results['heatmap_losses']):.6f}")
        
        # ROI Loss
        report.append("\n" + "-"*70)
        report.append("ROI LOSS")
        report.append("-"*70)
        report.append(f"Média:    {np.mean(results['roi_losses']):.6f}")
        report.append(f"Mediana:  {np.median(results['roi_losses']):.6f}")
        report.append(f"Std:      {np.std(results['roi_losses']):.6f}")
        
        # Por keypoint
        num_keypoints = len(results['keypoint_distances'])
        for k in range(num_keypoints):
            distances = results['keypoint_distances'][k]
            report.append("\n" + "-"*70)
            report.append(f"KEYPOINT {k+1} - DISTÂNCIA EUCLIDIANA (pixels)")
            report.append("-"*70)
            report.append(f"Média:    {np.mean(distances):.2f}")
            report.append(f"Mediana:  {np.median(distances):.2f}")
            report.append(f"Std:      {np.std(distances):.2f}")
            report.append(f"Min:      {np.min(distances):.2f}")
            report.append(f"Max:      {np.max(distances):.2f}")
            report.append(f"Quartis:  Q1={np.percentile(distances, 25):.2f}, "
                         f"Q3={np.percentile(distances, 75):.2f}")
        
        # Correlações
        report.append("\n" + "-"*70)
        report.append("CORRELAÇÕES")
        report.append("-"*70)
        corr_heat_total, _ = pearsonr(results['heatmap_losses'], results['total_losses'])
        report.append(f"Heatmap Loss vs Total Loss: {corr_heat_total:.4f}")
        
        corr_roi_total, _ = pearsonr(results['roi_losses'], results['total_losses'])
        report.append(f"ROI Loss vs Total Loss:     {corr_roi_total:.4f}")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n✓ Relatório salvo: {save_path}")
        
        return report_text