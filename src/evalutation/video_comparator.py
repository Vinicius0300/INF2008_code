"""
VideoComparator — inferência comparativa de modelos em vídeo.

Carrega todos os modelos em dict_config, processa cada frame do vídeo
medindo o tempo de inferência, e disponibiliza:
  - widget interativo (multi-seleção de modelos + play/pause + slider)
  - gráficos de tempo de inferência por modelo
"""

import os
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from src.models.models_artigos.resnet_point.two_stage_resnet import compute_crop_box

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_COLORS = [
    '#E63946', '#457B9D', '#2DC653', '#FF9F1C', '#9B5DE5',
    '#F15BB5', '#00BBF9', '#FEE440', '#00F5D4', '#FF6B6B',
]
_KP_LABELS = ['C2', 'C4']


# ---------------------------------------------------------------------------
# Helpers de pré-processamento e extração de keypoints
# ---------------------------------------------------------------------------

def _frame_to_gray_tensor(frame_bgr: np.ndarray, out_h: int, out_w: int,
                          transform=None) -> torch.Tensor:
    """
    Replica o pré-processamento do VFSSImageDataset:
      BGR -> escala de cinza (H, W, 1) -> transform_validation (se houver)
      -> resize para output_dim -> tensor (1, H, W) em [0, 1].
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=-1)                     # (H, W, 1)

    if transform is not None:
        try:
            gray = transform(image=gray, keypoints=[])["image"]
        except Exception:
            gray = transform(image=gray)["image"]

    if isinstance(gray, torch.Tensor):                       # ToTensorV2 -> (1, H, W)
        t = gray.float()
    else:
        t = torch.from_numpy(np.ascontiguousarray(gray)).permute(2, 0, 1).float()

    if t.shape[-2:] != (out_h, out_w):
        t = F.interpolate(t.unsqueeze(0), size=(out_h, out_w),
                          mode='bilinear', align_corners=False).squeeze(0)
    return t / 255.0                                         # (1, H, W)


def _extract_kp_from_heatmap(heatmap: torch.Tensor, roi: torch.Tensor = None) -> list:
    """
    Mesma lógica do TestEvaluator.extract_keypoints_from_heatmap
    (argmax por canal, opcionalmente restrito à ROI).
    heatmap : (1, K, H, W) ou (K, H, W)
    Retorno : lista de (x, y)
    """
    if heatmap.dim() == 4:
        heatmap = heatmap[0]
    mask = None
    if roi is not None:
        mask = (roi > 0).float().squeeze()
        if mask.shape != heatmap.shape[-2:] or mask.sum() == 0:
            mask = None
    pts = []
    for k in range(heatmap.shape[0]):
        hk = heatmap[k] * mask.to(heatmap.device) if mask is not None else heatmap[k]
        idx = hk.argmax().item()
        row, col = divmod(idx, hk.shape[-1])
        pts.append((col, row))  # (x, y)
    return pts


def _scale_pts_to_frame(pts_model, model_w, model_h, frame_w, frame_h):
    """Escala pontos de (model_w, model_h) para coordenadas de frame."""
    return [
        (int(x / model_w * frame_w), int(y / model_h * frame_h))
        for x, y in pts_model
    ]


# ---------------------------------------------------------------------------
# VideoComparator
# ---------------------------------------------------------------------------

class VideoComparator:
    """
    Carrega todos os modelos de dict_config e roda inferência em cada frame
    de video_path, registrando predições e tempo de inferência.

    Parâmetros
    ----------
    dict_config : dict
        Mesmo dict usado em ModelComparator.
        Valor único → single-stage.
        Valor tupla  → (config_stage1, config_stage2).
    video_path  : str
        Caminho para o arquivo de vídeo.
    device      : str | None
        'cuda', 'cpu' ou None (auto-detect).
    """

    def __init__(self, dict_config: dict, video_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dict_config = dict_config
        self.video_path  = video_path
        self.device      = device

        self.frames: list        = []   # BGR numpy arrays originais
        self.results: dict       = {}   # ckpt_key → {predictions, times}
        self.model_labels: dict  = {}   # ckpt_key → label de exibição
        self.fps_video: float    = 30.0

        self._read_video()
        self._run_all_models()

    # ------------------------------------------------------------------ #
    #  Leitura do vídeo                                                    #
    # ------------------------------------------------------------------ #

    def _read_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {self.video_path}")
        self.fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        total = len(self.frames)
        dur   = total / self.fps_video
        print(f"Vídeo carregado: {total} frames  |  {dur:.1f}s  |  {self.fps_video:.1f} FPS")

    # ------------------------------------------------------------------ #
    #  Carregamento de modelo                                              #
    # ------------------------------------------------------------------ #

    def _load_model(self, config, checkpoint_path: str) -> torch.nn.Module:
        kwargs = config.model_kwargs if hasattr(config, 'model_kwargs') else {}
        model  = config.model_class(**kwargs)
        ckpt   = torch.load(checkpoint_path, map_location=self.device)
        state  = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model

    def _output_dim(self, config, fallback=(448, 448)):
        """Retorna (H, W) esperado pelo modelo."""
        od = getattr(config, 'output_dim', None)
        if od is not None:
            if isinstance(od, (tuple, list)) and len(od) >= 2:
                return int(od[0]), int(od[1])
            if isinstance(od, int):
                return od, od
        return fallback

    def _preprocess(self, frame_bgr: np.ndarray, config) -> torch.Tensor:
        """
        Frame BGR -> tensor (1, C, H, W), igual ao TestEvaluator:
          img = dataset(frame)            # (1, H, W) cinza em [0,1]
          modify_input_fn(img)            # (1, 3, H, W) [original, CLAHE, double CLAHE]
          ou img.unsqueeze(0)             # (1, 1, H, W)
        """
        H, W = self._output_dim(config)
        t = _frame_to_gray_tensor(frame_bgr, H, W,
                                  getattr(config, 'transform_validation', None))
        fn = getattr(config, 'modify_input_fn', None)
        t = fn(t) if fn is not None else t.unsqueeze(0)
        return t.float().to(self.device)

    # ------------------------------------------------------------------ #
    #  Inferência single-stage                                            #
    # ------------------------------------------------------------------ #

    def _run_single_stage(self, config, ckpt_path: str, label: str):
        print(f"  Carregando: {label}")
        model = self._load_model(config, ckpt_path)
        H, W  = self._output_dim(config)
        predictions, times = [], []

        with torch.no_grad():
            for frame in self.frames:
                fh, fw = frame.shape[:2]
                t0     = time.perf_counter()
                inp    = self._preprocess(frame, config)
                out    = model(inp)

                # Modelos retornam (pred_roi, pred_heatmap, pred_keypoints)
                if isinstance(out, (tuple, list)) and len(out) == 3:
                    pred_roi, pred_hm, pred_kp = out
                else:
                    pred_roi, pred_hm, pred_kp = None, out, None

                if pred_kp is not None:
                    kp = pred_kp.squeeze(0).cpu()
                    pts_model = [(float(kp[k, 0]), float(kp[k, 1])) for k in range(kp.shape[0])]
                else:
                    pts_model = _extract_kp_from_heatmap(pred_hm, pred_roi)
                pts_frame = _scale_pts_to_frame(pts_model, W, H, fw, fh)
                t1 = time.perf_counter()

                predictions.append(pts_frame)
                times.append((t1 - t0) * 1000)

        del model
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        mean_t = np.mean(times)
        print(f"  ✓ {label}:  {mean_t:.1f} ms/frame  ({1000/mean_t:.1f} FPS)")
        self.results[ckpt_path]      = {'predictions': predictions, 'times': times}
        self.model_labels[ckpt_path] = label

    # ------------------------------------------------------------------ #
    #  Inferência two-stage                                               #
    # ------------------------------------------------------------------ #

    def _run_two_stage(
        self,
        config_s1, config_s2,
        ckpt_s1: str, ckpt_s2: str,
        label: str,
    ):
        print(f"  Carregando: {label}")
        model_s1 = self._load_model(config_s1, ckpt_s1)
        model_s2 = self._load_model(config_s2, ckpt_s2)
        H1, W1   = self._output_dim(config_s1)
        H2, W2   = self._output_dim(config_s2, fallback=(224, 224))
        predictions, times = [], []

        with torch.no_grad():
            for frame in self.frames:
                fh, fw = frame.shape[:2]
                t0     = time.perf_counter()

                # ---- Stage 1 ----
                inp1          = self._preprocess(frame, config_s1)
                _, _, pts_s1  = model_s1(inp1)          # (1, K, 2) em W1×H1

                crop_box = compute_crop_box(pts_s1, W1)  # (1, 4) [x1,y1,x2,y2]

                # Recorta e redimensiona no espaço do tensor
                img_cpu  = inp1.squeeze(0).cpu()
                cbox_cpu = crop_box.cpu()
                x1i = max(int(cbox_cpu[0][0].item()), 0)
                y1i = max(int(cbox_cpu[0][1].item()), 0)
                x2i = min(int(cbox_cpu[0][2].item()), img_cpu.shape[-1])
                y2i = min(int(cbox_cpu[0][3].item()), img_cpu.shape[-2])
                x2i = max(x2i, x1i + 1)
                y2i = max(y2i, y1i + 1)
                cropped = img_cpu[:, y1i:y2i, x1i:x2i].unsqueeze(0)
                cropped = F.interpolate(
                    cropped.float(), size=(H2, W2),
                    mode='bilinear', align_corners=False
                ).to(self.device)

                # ---- Stage 2 ----
                _, _, pts_s2 = model_s2(cropped)         # (1, K, 2) em W2×H2

                # Mapeia de volta para imagem completa (W1×H1)
                crop_w = float(x2i - x1i)
                crop_h = float(y2i - y1i)
                pts_full = pts_s2.clone().float().cpu()
                pts_full[:, :, 0] = pts_s2[:, :, 0].cpu() / W2 * crop_w + x1i
                pts_full[:, :, 1] = pts_s2[:, :, 1].cpu() / H2 * crop_h + y1i
                # pts_full em espaço W1×H1; agora escala para frame real
                pts_s = pts_full.squeeze(0)
                pts_frame = [
                    (int(pts_s[k, 0].item() / W1 * fw),
                     int(pts_s[k, 1].item() / H1 * fh))
                    for k in range(pts_s.shape[0])
                ]
                t1 = time.perf_counter()

                predictions.append(pts_frame)
                times.append((t1 - t0) * 1000)

        del model_s1, model_s2
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        mean_t = np.mean(times)
        print(f"  ✓ {label}:  {mean_t:.1f} ms/frame  ({1000/mean_t:.1f} FPS)")
        self.results[ckpt_s1]      = {'predictions': predictions, 'times': times}
        self.model_labels[ckpt_s1] = label

    # ------------------------------------------------------------------ #
    #  Orquestração                                                       #
    # ------------------------------------------------------------------ #

    def _run_all_models(self):
        _fold_pat = re.compile(r'fold_([1-9])_best\.pth$')
        print(f"\nRodando inferência em {len(self.frames)} frames por modelo…")
        for key, cfg in self.dict_config.items():
            if isinstance(cfg, tuple):
                cfg_s1, cfg_s2 = cfg
                root = cfg_s1.checkpoint_dir
                if not os.path.exists(root):
                    print(f"  [AVISO] Diretório não encontrado: {root}")
                    continue
                for fname in sorted(os.listdir(root)):
                    m = _fold_pat.search(fname)
                    if not m:
                        continue
                    fold   = m.group(1)
                    ck_s1  = os.path.join(root, fname)
                    ck_s2  = os.path.join(cfg_s2.checkpoint_dir, f'fold_{fold}_best.pth')
                    if not os.path.exists(ck_s2):
                        print(f"  [AVISO] Stage 2 não encontrado: {ck_s2}")
                        continue
                    label = f"Two-Stage ResNet — Fold {fold}"
                    self._run_two_stage(cfg_s1, cfg_s2, ck_s1, ck_s2, label)
            else:
                root = cfg.checkpoint_dir
                if not os.path.exists(root):
                    print(f"  [AVISO] Diretório não encontrado: {root}")
                    continue
                for fname in sorted(os.listdir(root)):
                    m = _fold_pat.search(fname)
                    if not m:
                        continue
                    fold  = m.group(1)
                    ck    = os.path.join(root, fname)
                    arch  = os.path.basename(key)
                    label = f"{arch} — Fold {fold}"
                    self._run_single_stage(cfg, ck, label)

        n = len(self.results)
        print(f"\nInferência concluída: {n} checkpoint(s) processado(s).\n")

    # ------------------------------------------------------------------ #
    #  Widget interativo                                                  #
    # ------------------------------------------------------------------ #

    def display_interactive(self):
        """
        Exibe widget interativo no notebook:
          - SelectMultiple para escolher um ou mais modelos
          - Play + IntSlider para navegar pelos frames
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return

        keys    = list(self.results.keys())
        options = [(self.model_labels[k], k) for k in keys]

        select = widgets.SelectMultiple(
            options=options,
            value=[keys[0]],
            description='Modelos:',
            layout=widgets.Layout(width='420px', height='160px'),
            style={'description_width': 'initial'},
        )
        play = widgets.Play(
            value=0, min=0, max=len(self.frames) - 1, step=1,
            interval=max(33, int(1000 / self.fps_video)),
            description='▶',
        )
        slider = widgets.IntSlider(
            value=0, min=0, max=len(self.frames) - 1,
            description='Frame:',
            layout=widgets.Layout(width='500px'),
            style={'description_width': 'initial'},
        )
        widgets.jslink((play, 'value'), (slider, 'value'))
        out = widgets.Output()

        def _render(frame_idx: int, selected_keys):
            frame_bgr = self.frames[frame_idx]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            fh, fw    = frame_bgr.shape[:2]
            sel       = list(selected_keys)
            n         = len(sel)

            with out:
                clear_output(wait=True)
                if n == 0:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.imshow(frame_rgb)
                    ax.set_title(f"Frame {frame_idx} / {len(self.frames)-1}", fontsize=11)
                    ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                    return

                cols = min(n, 3)
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(6.5 * cols, 5.2 * rows),
                                         squeeze=False)
                ax_flat = [axes[r][c] for r in range(rows) for c in range(cols)]

                for i, ck in enumerate(sel):
                    ax = ax_flat[i]
                    ax.imshow(frame_rgb)
                    pts   = self.results[ck]['predictions'][frame_idx]
                    t_ms  = self.results[ck]['times'][frame_idx]
                    color = _COLORS[keys.index(ck) % len(_COLORS)]

                    for j, (px, py) in enumerate(pts):
                        ax.scatter(px, py, s=90, color=color,
                                   edgecolors='white', linewidths=1.2, zorder=5)
                        lbl = _KP_LABELS[j] if j < len(_KP_LABELS) else str(j)
                        ax.annotate(
                            lbl, (px, py),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, color='white', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc=color, alpha=0.75),
                        )
                    ax.set_title(
                        f"{self.model_labels[ck]}\n{t_ms:.1f} ms",
                        fontsize=9, pad=4,
                    )
                    ax.axis('off')

                # Oculta eixos não usados
                for i in range(n, len(ax_flat)):
                    ax_flat[i].set_visible(False)

                fig.suptitle(
                    f"Frame {frame_idx} / {len(self.frames)-1}",
                    fontsize=12, y=1.01,
                )
                plt.tight_layout()
                plt.show()

        def _on_change(_):
            _render(slider.value, select.value)

        slider.observe(_on_change, names='value')
        select.observe(_on_change, names='value')

        _render(0, select.value)
        controls = widgets.VBox([
            widgets.HBox([play, slider]),
            widgets.Label("Segure Ctrl / Cmd para selecionar múltiplos modelos:"),
            select,
        ])
        display(widgets.VBox([controls, out]))

    # ------------------------------------------------------------------ #
    #  Gráficos de tempo                                                  #
    # ------------------------------------------------------------------ #

    def plot_timing_metrics(self):
        """
        Exibe dois gráficos:
          1. Boxplot de tempo de inferência por modelo (ms/frame)
          2. Barra horizontal de média ± desvio padrão
        E imprime um resumo em FPS.
        """
        if not self.results:
            print("Nenhum resultado disponível.")
            return

        keys        = list(self.results.keys())
        labels      = [self.model_labels[k] for k in keys]
        times_list  = [np.array(self.results[k]['times']) for k in keys]
        means       = [t.mean() for t in times_list]
        stds        = [t.std()  for t in times_list]

        box_color     = '#A8DADC'
        median_color  = '#E63946'
        whisker_color = '#457B9D'
        bar_h         = max(4.5, len(keys) * 0.85)

        # ---- 1. Boxplot ----
        fig1, ax1 = plt.subplots(figsize=(11, bar_h))
        bp = ax1.boxplot(times_list, labels=labels, vert=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(box_color)
            patch.set_edgecolor(whisker_color)
        for med in bp['medians']:
            med.set(color=median_color, linewidth=2)
        for wh in bp['whiskers']:
            wh.set(color=whisker_color, linewidth=1.5, linestyle='--')
        for flier in bp['fliers']:
            flier.set(marker='o', color=whisker_color, alpha=0.4, markersize=4)
        ax1.set_title("Tempo de inferência por modelo (ms/frame)",
                      fontsize=13, pad=12)
        ax1.set_xlabel("Tempo (ms)")
        ax1.tick_params(axis='y', labelsize=8)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ---- 2. Barra média ± std ----
        fig2, ax2 = plt.subplots(figsize=(11, bar_h))
        y_pos = np.arange(len(keys))
        ax2.barh(y_pos, means, xerr=stds, align='center',
                 color=box_color, edgecolor=whisker_color,
                 error_kw=dict(ecolor=median_color, lw=2, capsize=5))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel("Tempo médio (ms)")
        ax2.set_title("Tempo médio de inferência ± desvio padrão",
                      fontsize=13, pad=12)
        ax2.grid(True, alpha=0.3, axis='x')
        x_max = max(m + s for m, s in zip(means, stds))
        for i, (m, s) in enumerate(zip(means, stds)):
            ax2.text(m + s + x_max * 0.01, i,
                     f"{m:.1f} ± {s:.1f} ms",
                     va='center', fontsize=8)
        plt.tight_layout()
        plt.show()

        # ---- 3. Resumo textual ----
        print("\n─── Resumo de Velocidade ───────────────────────────────────────────")
        print(f"{'Modelo':<45}  {'Média':>8}  {'Mediana':>8}  {'FPS':>7}")
        print("─" * 75)
        for k, m, s, t in zip(keys, means, stds, times_list):
            fps    = 1000.0 / m if m > 0 else float('inf')
            median = float(np.median(t))
            print(f"{self.model_labels[k]:<45}  {m:>6.1f}ms  {median:>6.1f}ms  {fps:>6.1f}")
        print("─" * 75)
