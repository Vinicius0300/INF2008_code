"""
Avaliação do pipeline Two-Stage (GlobalStageResNet50 → LocalStageResNet34)
no conjunto de teste.

Importar em model_comparator.py e em notebooks de avaliação:
    from src.evalutation.two_stage_inference import TwoStageEvaluator, evaluate_two_stage
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from src.models.models_artigos.resnet_point.two_stage_resnet import compute_crop_box
from src.training.loss import LossCalculator


class TwoStageEvaluator:
    """
    Avalia o pipeline completo de dois estágios no conjunto de teste.

    Stage 1 (GlobalStageWrapper, 448×448) → compute_crop_box → crop
    Stage 2 (LocalStageWrapper, 224×224)  → mapeamento de volta para imagem completa

    Todos os resultados e distâncias são reportados em coordenadas
    da imagem completa (448×448).

    Nota sobre coordenadas:
      pts_px do modelo tem dim-0 = coluna (x) e dim-1 = linha (y),
      conforme compute_crop_box: x1 = pts[:,0]-side/2, y1 = pts[:,1]-side/2
      e CroppedVFSSImageDataset: img[:, y1:y2, x1:x2].
    """

    def __init__(
        self,
        checkpoint_stage1: str,
        checkpoint_stage2: str,
        config_stage1,
        config_stage2,
    ):
        self.device            = config_stage1.device
        self.modify_input_fn   = config_stage1.modify_input_fn
        self.output_dim_stage1 = config_stage1.output_dim   # ex.: (448, 448)
        self.output_dim_stage2 = config_stage2.output_dim   # ex.: (224, 224)

        # ---- Stage 1 ----
        self.model_stage1 = config_stage1.model_class(**config_stage1.model_kwargs).to(self.device)
        ckpt1 = torch.load(checkpoint_stage1, map_location=self.device)
        self.model_stage1.load_state_dict(ckpt1["model_state_dict"])
        self.model_stage1.eval()
        print(f"✓ Stage 1 carregado: {checkpoint_stage1}")
        print(f"  Epoch: {ckpt1.get('epoch', 'N/A')} | Val Loss: {ckpt1.get('val_loss', 0.0):.4f}")

        # ---- Stage 2 ----
        self.model_stage2 = config_stage2.model_class(**config_stage2.model_kwargs).to(self.device)
        ckpt2 = torch.load(checkpoint_stage2, map_location=self.device)
        self.model_stage2.load_state_dict(ckpt2["model_state_dict"])
        self.model_stage2.eval()
        print(f"✓ Stage 2 carregado: {checkpoint_stage2}")
        print(f"  Epoch: {ckpt2.get('epoch', 'N/A')} | Val Loss: {ckpt2.get('val_loss', 0.0):.4f}")

        # Número de keypoints — suporta 'num_points' e 'num_keypoints'
        mkw = config_stage2.model_kwargs
        self.num_keypoints = mkw.get("num_points", mkw.get("num_keypoints", 2))

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _crop_and_resize(
        self, img_tensor: torch.Tensor, crop_box: torch.Tensor
    ) -> torch.Tensor:
        """
        Recorta img_tensor com crop_box e redimensiona para output_dim_stage2.

        img_tensor : (C, H, W) na CPU
        crop_box   : (1, 4) float tensor [x1, y1, x2, y2]
        Retorno    : (1, C, H2, W2) na CPU
        """
        x1, y1, x2, y2 = crop_box[0]
        x1_i = max(int(x1.item()), 0)
        y1_i = max(int(y1.item()), 0)
        x2_i = min(int(x2.item()), img_tensor.shape[-1])   # W
        y2_i = min(int(y2.item()), img_tensor.shape[-2])   # H
        # Garante pelo menos 1 pixel de cada lado
        x2_i = max(x2_i, x1_i + 1)
        y2_i = max(y2_i, y1_i + 1)

        cropped = img_tensor[:, y1_i:y2_i, x1_i:x2_i].unsqueeze(0)  # (1,C,ch,cw)
        h2, w2  = self.output_dim_stage2
        resized = F.interpolate(cropped.float(), size=(h2, w2),
                                mode="bilinear", align_corners=False)
        return resized  # (1, C, H2, W2)

    def _map_to_full_image(
        self, pts_stage2: torch.Tensor, crop_box: torch.Tensor
    ) -> torch.Tensor:
        """
        Mapeia predições Stage 2 (espaço 224×224) → coordenadas da imagem completa.

        pts_stage2 : (1, num_points, 2) — dim-0=col(x), dim-1=row(y) no crop
        crop_box   : (1, 4) float tensor [x1, y1, x2, y2] na imagem completa
        Retorno    : (1, num_points, 2) na imagem completa
        """
        x1, y1, x2, y2 = crop_box[0]
        crop_w = (x2 - x1).float()
        crop_h = (y2 - y1).float()
        w2 = float(self.output_dim_stage2[1])
        h2 = float(self.output_dim_stage2[0])

        full = pts_stage2.clone().float()
        full[:, :, 0] = pts_stage2[:, :, 0] / w2 * crop_w + x1   # coluna → x
        full[:, :, 1] = pts_stage2[:, :, 1] / h2 * crop_h + y1   # linha  → y
        return full

    @staticmethod
    def euclidean_distance(p1: torch.Tensor, p2: torch.Tensor) -> float:
        return torch.sqrt(torch.sum((p1.float() - p2.float()) ** 2)).item()

    # ------------------------------------------------------------------
    # Avaliação principal
    # ------------------------------------------------------------------

    def evaluate_test_set(
        self,
        test_dataset_stage1,
        loss_calculator: LossCalculator,
    ) -> dict:
        """
        Avalia o pipeline two-stage no conjunto de teste.

        test_dataset_stage1 : VFSSImageDataset com imagens completas (448×448).
        """
        results = {
            "keypoint_distances":  [[] for _ in range(self.num_keypoints)],
            "stage1_distances":    [[] for _ in range(self.num_keypoints)],
            "keypoint_losses":     [],
            "heatmap_losses":      [],
            "roi_losses":          [],
            "total_losses":        [],
            "predictions":         [],   # Stage 2 em coords da imagem completa
            "predictions_stage1":  [],   # Stage 1 em coords da imagem completa
            "ground_truths":       [],
            "images":              [],
            "crop_boxes":          [],   # (4,) [x1, y1, x2, y2]
        }

        print(f"\nAvaliando {len(test_dataset_stage1)} amostras (two-stage)…")

        for idx in tqdm(range(len(test_dataset_stage1)), desc="Avaliando"):
            input_img, keypoint, heatmap, roi = test_dataset_stage1[idx]

            # --- Prepara entrada Stage 1 ---
            if self.modify_input_fn is not None:
                input_t = self.modify_input_fn(input_img).float().to(self.device)
            else:
                input_t = input_img.unsqueeze(0).float().to(self.device)

            gt_kpts    = torch.tensor(keypoint).float().to(self.device)
            gt_heatmap = heatmap.float().to(self.device)
            gt_roi     = roi.float().to(self.device)

            with torch.no_grad():
                # ---- Stage 1 ----
                _, _, pts_s1 = self.model_stage1(input_t)
                # pts_s1 : (1, num_kpts, 2) em coords 448×448

                image_size = self.output_dim_stage1[0]
                crop_box   = compute_crop_box(pts_s1, image_size)   # (1, 4)

                img_cpu  = input_t.squeeze(0).cpu()
                cbox_cpu = crop_box.cpu()
                cropped  = self._crop_and_resize(img_cpu, cbox_cpu).to(self.device)

                # ---- Stage 2 ----
                _, _, pts_s2 = self.model_stage2(cropped)
                # pts_s2 : (1, num_kpts, 2) em coords 224×224 do crop

            # Mapeia Stage 2 de volta para imagem completa
            pts_final = self._map_to_full_image(pts_s2.cpu(), cbox_cpu)
            pts_final = pts_final.squeeze(0).to(self.device)
            pts_s1_sq = pts_s1.squeeze(0).cpu()

            # --- Distâncias ---
            for k in range(self.num_keypoints):
                results["keypoint_distances"][k].append(
                    self.euclidean_distance(gt_kpts[k], pts_final[k]))
                results["stage1_distances"][k].append(
                    self.euclidean_distance(gt_kpts[k].cpu(), pts_s1_sq[k]))

            # --- Losses ---
            loss_total, components = loss_calculator.calculate_loss(
                pred_roi=None,
                pred_heatmap=None,
                pred_keypoints=pts_final.unsqueeze(0),
                gt_roi=gt_roi.unsqueeze(0),
                gt_heatmap=gt_heatmap.unsqueeze(0),
                gt_keypoints=gt_kpts.unsqueeze(0),
            )
            results["keypoint_losses"].append(components["keypoints"])
            results["heatmap_losses"].append(components["heatmap"])
            results["roi_losses"].append(components["roi"])
            results["total_losses"].append(loss_total.item())

            # --- Armazena para visualização ---
            results["predictions"].append({
                "points":   pts_final.cpu(),
                "heatmap":  None,
                "roi":      None,
                "keypoint": pts_final.cpu(),
            })
            results["predictions_stage1"].append({"points": pts_s1_sq})
            results["ground_truths"].append({
                "points":   gt_kpts.cpu(),
                "heatmap":  gt_heatmap.cpu(),
                "roi":      gt_roi.cpu(),
                "keypoint": gt_kpts.cpu(),
            })
            results["images"].append(input_t.squeeze(0).cpu())
            results["crop_boxes"].append(cbox_cpu.squeeze(0))

        # Converte para arrays numpy
        results["keypoint_distances"] = [np.array(d) for d in results["keypoint_distances"]]
        results["stage1_distances"]   = [np.array(d) for d in results["stage1_distances"]]

        def _to_np(lst):
            return np.array([v.detach().cpu().item() if torch.is_tensor(v) else v for v in lst])

        results["keypoint_losses"] = _to_np(results["keypoint_losses"])
        results["heatmap_losses"]  = _to_np(results["heatmap_losses"])
        results["roi_losses"]      = _to_np(results["roi_losses"])
        results["total_losses"]    = _to_np(results["total_losses"])

        return results

    # ------------------------------------------------------------------
    # Visualizações
    # ------------------------------------------------------------------

    def plot_loss_training(
        self,
        loss_history: dict,
        fold_number: int = 1,
        save_path: str = None,
    ):
        """Histórico de loss — Total Loss e Keypoint Loss."""
        fold_hist = loss_history["fold_histories"][fold_number - 1]
        df = pd.DataFrame(fold_hist)
        df["train_kp"] = df["train_components"].apply(lambda x: x.get("keypoints", 0))
        df["val_kp"]   = df["val_components"].apply(lambda x: x.get("keypoints", 0))

        idx_best = df["val_loss"].idxmin()
        best_ep  = df.iloc[idx_best]["epoch"]
        best_val = df.iloc[idx_best]["val_loss"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Evolução da Loss — Fold {fold_number}\n"
            f"Melhor época: {best_ep} (val_loss={best_val:.4f})",
            fontsize=13,
        )
        for ax, (tr, va, title) in zip(axes, [
            ("train_loss", "val_loss", "Total Loss"),
            ("train_kp",   "val_kp",   "Keypoint Loss"),
        ]):
            ax.plot(df["epoch"], df[tr], label="Treino",    color="#1f77b4", linewidth=2)
            ax.plot(df["epoch"], df[va], label="Validação", color="#ff7f0e", linewidth=2)
            ax.scatter(best_ep, df.iloc[idx_best][va], color="red", marker="*",
                       s=200, zorder=5, label=f"Melhor ({best_ep})")
            ax.axvline(x=best_ep, color="red", linestyle="--", alpha=0.5)
            ax.set_title(title); ax.set_xlabel("Época"); ax.set_ylabel("Loss")
            ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_fold{fold_number}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_loss_distributions(self, results: dict, save_path: str = None):
        """Histogramas de Keypoint Loss e Total Loss."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Distribuição das Losses — Two-Stage (Conjunto de Teste)", fontsize=14)

        for ax, (data, title, color) in zip(axes, [
            (results["keypoint_losses"], "Keypoint Loss", "steelblue"),
            (results["total_losses"],    "Total Loss",    "coral"),
        ]):
            ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.7)
            ax.axvline(np.mean(data),   color="red",   linestyle="--",
                       label=f"Média: {np.mean(data):.4f}")
            ax.axvline(np.median(data), color="green", linestyle="--",
                       label=f"Mediana: {np.median(data):.4f}")
            ax.set_xlabel(title); ax.set_ylabel("Frequência")
            ax.set_title(f"Distribuição: {title}")
            ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Salvo: {save_path}")
        plt.show()

    def plot_keypoint_analysis(self, results: dict, save_path: str = None):
        """Histogramas de distância — Stage 1 (coarse) vs Stage 2 (final)."""
        nk = self.num_keypoints
        fig, axes = plt.subplots(2, nk, figsize=(6 * nk, 10))
        if nk == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle("Análise de Distâncias por Keypoint", fontsize=16)

        for k in range(nk):
            for row, (distances, label, color) in enumerate([
                (results["stage1_distances"][k],   "Stage 1 (coarse)", "steelblue"),
                (results["keypoint_distances"][k], "Stage 2 (final)",  "orange"),
            ]):
                ax = axes[row, k]
                ax.hist(distances, bins=30, color=color, edgecolor="black", alpha=0.7)
                m, med, s = np.mean(distances), np.median(distances), np.std(distances)
                ax.axvline(m,   color="red",   linestyle="--", linewidth=2,
                           label=f"Média: {m:.2f}px")
                ax.axvline(med, color="green", linestyle="--", linewidth=2,
                           label=f"Mediana: {med:.2f}px")
                ax.set_xlabel("Distância Euclidiana (px)")
                ax.set_ylabel("Frequência")
                ax.set_title(f"Keypoint {k+1} — {label}\n(Std: {s:.2f}px)")
                ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Salvo: {save_path}")
        plt.show()

    def visualize_predictions(
        self,
        results: dict,
        metric_type: str = "distance",
        save_path: str = None,
    ):
        """
        Melhor, mediano e pior caso por keypoint.
        GT (verde ×), Stage 1 (azul △), Stage 2 (vermelho ●), crop box (amarelo --).

        Convenção: scatter(pt[0], pt[1]) = scatter(col, row).
        Se os pontos aparecerem transpostos, troque para scatter(pt[1], pt[0]).
        """
        for k in range(self.num_keypoints):
            print(f"\n--- Keypoint {k+1} ---")
            if metric_type == "distance":
                metric      = results["keypoint_distances"][k]
                metric_name = "Dist Final (px)"
            else:
                metric      = results["total_losses"]
                metric_name = "Total Loss"

            idx_min = np.argmin(metric)
            idx_max = np.argmax(metric)
            idx_med = np.argsort(metric)[len(metric) // 2]
            cases   = [(idx_min, "Melhor"), (idx_med, "Mediano"), (idx_max, "Pior")]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Keypoint {k+1} — {metric_name}", fontsize=14)

            for col, (idx, title) in enumerate(cases):
                ax  = axes[col]
                img = results["images"][idx]
                ax.imshow(img[0].numpy(), cmap="gray")

                gt_pt   = results["ground_truths"][idx]["points"][k]
                pred_pt = results["predictions"][idx]["points"][k]
                s1_pt   = results["predictions_stage1"][idx]["points"][k]
                cbox    = results["crop_boxes"][idx]

                rect = Rectangle(
                    (cbox[0].item(), cbox[1].item()),
                    (cbox[2] - cbox[0]).item(),
                    (cbox[3] - cbox[1]).item(),
                    linewidth=1.5, edgecolor="yellow",
                    facecolor="none", linestyle="--",
                )
                ax.add_patch(rect)

                ax.scatter(gt_pt[0],   gt_pt[1],   color="lime",       marker="x",
                           s=160, linewidths=3, label="GT",      zorder=6)
                ax.scatter(s1_pt[0],   s1_pt[1],   color="dodgerblue", marker="^",
                           s=110, linewidths=2, label="Stage 1", zorder=5)
                ax.scatter(pred_pt[0], pred_pt[1], color="red",        marker="o",
                           s=110, linewidths=2, label="Stage 2", zorder=5)

                dist_final = self.euclidean_distance(gt_pt, pred_pt)
                ax.set_title(
                    f"{title}\n{metric_name}: {metric[idx]:.4f}\n"
                    f"Dist Final: {dist_final:.2f}px"
                )
                ax.axis("off")
                ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            if save_path:
                path_final = f"{save_path}_keypoint{k+1}.png"
                plt.savefig(path_final, dpi=300, bbox_inches="tight")
                print(f"✓ Salvo: {path_final}")
            plt.show()

    def generate_report(self, results: dict, save_path: str = None) -> str:
        lines = [
            "=" * 70,
            "RELATÓRIO — AVALIAÇÃO TWO-STAGE NO CONJUNTO DE TESTE",
            "=" * 70,
            f"\nAmostras avaliadas: {len(results['total_losses'])}",
            "\n" + "-" * 70,
            "KEYPOINT LOSS",
            "-" * 70,
            f"Média:   {np.mean(results['keypoint_losses']):.6f}",
            f"Mediana: {np.median(results['keypoint_losses']):.6f}",
            f"Std:     {np.std(results['keypoint_losses']):.6f}",
        ]

        for k in range(self.num_keypoints):
            for stage_label, dists in [
                ("Stage 1 (coarse)", results["stage1_distances"][k]),
                ("Stage 2 (final)",  results["keypoint_distances"][k]),
            ]:
                lines += [
                    "\n" + "-" * 70,
                    f"KEYPOINT {k+1} — {stage_label} — DISTÂNCIA EUCLIDIANA (px)",
                    "-" * 70,
                    f"Média:   {np.mean(dists):.2f}",
                    f"Mediana: {np.median(dists):.2f}",
                    f"Std:     {np.std(dists):.2f}",
                    f"Min:     {np.min(dists):.2f}",
                    f"Max:     {np.max(dists):.2f}",
                    f"Q1/Q3:   {np.percentile(dists, 25):.2f} / {np.percentile(dists, 75):.2f}",
                ]

        lines.append("\n" + "=" * 70)
        text = "\n".join(lines)
        print(text)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\n✓ Relatório salvo: {save_path}")
        return text


# -----------------------------------------------------------------------
# Função de conveniência
# -----------------------------------------------------------------------

def evaluate_two_stage(
    config_stage1,
    config_stage2,
    checkpoint_stage1: str,
    checkpoint_stage2: str,
    output_dir: str = "figs/two_stage",
    show_results: bool = True,
) -> dict:
    """
    Pipeline completo de avaliação two-stage no conjunto de teste.

    Args:
        config_stage1     : TrainingConfig do Stage 1 (imagem completa, 448×448)
        config_stage2     : TrainingConfig do Stage 2 (crop, 224×224)
        checkpoint_stage1 : caminho para o .pth do melhor modelo Stage 1
        checkpoint_stage2 : caminho para o .pth do melhor modelo Stage 2
        output_dir        : pasta onde salvar figuras e relatório (usado se show_results=True)
        show_results      : se True, gera e salva todas as visualizações
    """
    test_dataset = config_stage1.dataset_class(
        video_frame_df=config_stage1.df_test,
        output_dim=config_stage1.output_dim,
        transform=config_stage1.transform_validation,
        sigma_heatmap=config_stage1.sigma_heatmap,
    )

    evaluator = TwoStageEvaluator(
        checkpoint_stage1=checkpoint_stage1,
        checkpoint_stage2=checkpoint_stage2,
        config_stage1=config_stage1,
        config_stage2=config_stage2,
    )

    loss_calculator = LossCalculator(config=config_stage2)

    print("\n" + "=" * 70)
    print("AVALIANDO PIPELINE TWO-STAGE NO CONJUNTO DE TESTE")
    print("=" * 70)

    results = evaluator.evaluate_test_set(test_dataset, loss_calculator)

    if show_results:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        evaluator.plot_loss_distributions(
            results, save_path=str(out / "loss_distributions.png"))
        evaluator.plot_keypoint_analysis(
            results, save_path=str(out / "keypoint_analysis.png"))
        evaluator.visualize_predictions(
            results, metric_type="distance",
            save_path=str(out / "predictions_by_distance"))
        evaluator.visualize_predictions(
            results, metric_type="total_loss",
            save_path=str(out / "predictions_by_loss"))
        evaluator.generate_report(
            results, save_path=str(out / "evaluation_report.txt"))

        print("\n" + "=" * 70)
        print(f"Avaliação concluída! Resultados em: {out}")
        print("=" * 70)

    return results
