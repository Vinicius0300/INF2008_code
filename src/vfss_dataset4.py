from src.utils import get_corners_from_angle
from src.target.heatmap import generate_heatmap_from_points
from src.target.roi import generate_roi_from_points
from src.utils import get_script_relative_path, get_project_root_directory

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch

import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Modos de augmentation suportados
# ---------------------------------------------------------------------------
_VALID_MODES = frozenset({'online', 'offline', 'none'})

class VFSSImageDataset(Dataset):
    
    __valid_targets = frozenset({'mask', 'points', 'roi', 'heatmap'})

    def __init__(
        self,
        video_frame_df: pd.DataFrame,
        target: str = 'mask',
        output_dim: tuple = (256, 256),
        transform: A.Compose | None = None,
        mode: str = 'none',
        sigma: float = 10,
        preload: bool = True,
    ):
        """
        Args:
            video_frame_df (pd.DataFrame): DataFrame com colunas 'frame_path',
                'frame_id', 'video_id', 'paciente_id', 'momento',
                'procedimento', 'selected_labeler' e 'target_dir'.
            target (str): Tipo(s) de alvo separados por '+'.
                Valores válidos: 'mask', 'points', 'roi', 'heatmap'.
                Exemplo: 'mask+heatmap'.
            output_dim (tuple): Dimensão (H, W) usada para o Resize no modo
                ``'none'``. Ignorada nos modos ``'online'`` e ``'offline'``
                (o Resize deve estar dentro da pipeline Albumentations).
            transform (A.Compose | None): Pipeline Albumentations. Obrigatória
                nos modos ``'online'`` e ``'offline'``; ignorada em ``'none'``.
            mode (str): Modo de augmentation. Um de: ``'none'``, ``'online'``,
                ``'offline'``.
            sigma (float): Desvio-padrão da gaussiana para geração de heatmaps.
            preload (bool): Se True, pré-carrega todos os itens em RAM.
                Forçado para True quando ``mode='offline'``.
        """
        self.video_frame_df = video_frame_df.reset_index(drop=True).copy()
        self.target = self._validate_target(target)
        self.target_keys = target.split('+')
        self.output_dim = output_dim
        self.sigma = sigma
        self.mode = self._validate_mode(mode)
        self.transform = transform

        # Offline exige preload para garantir que o transform só rode uma vez
        if self.mode == 'offline':
            preload = True

        # Valida que transform foi fornecido quando necessário
        if self.mode in ('online', 'offline') and self.transform is None:
            raise ValueError(
                f"mode='{self.mode}' requer uma pipeline 'transform' "
                f"(albumentations.Compose). Recebido: None."
            )

        # Transformações para modo 'none': apenas resize + to_tensor (torchvision)
        self._to_tensor_tv = T.ToTensor()
        self._resize_none = T.Resize(output_dim)
        self._resize_mask_none = T.Resize(
            output_dim, interpolation=T.InterpolationMode.NEAREST
        )

        # Cache em memória
        self._cache: dict = {}
        if preload:
            self._preload_all()

    # ------------------------------------------------------------------
    # Dunder / representação
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.video_frame_df)

    def __repr__(self) -> str:
        return (
            f"VFSSImageDataset("
            f"n={len(self)}, "
            f"target='{self.target}', "
            f"mode='{self.mode}', "
            f"output_dim={self.output_dim}, "
            f"preloaded={bool(self._cache)})"
        )

    def _repr_html_(self) -> str:
        return self.video_frame_df._repr_html_()

    def __getitem__(self, idx: int):
        # Modo offline e preload: cache já tem o item transformado
        if idx in self._cache:
            if self.mode in ('offline', 'none'):
                return self._cache[idx]
            # Modo online: cache tem o item RAW (sem augmentation)
            image_np, target_np, meta = self._cache[idx]
            return self._apply_albumentations(image_np, target_np, meta)

        # Sem cache (preload=False, mode='online' ou 'none')
        return self._load_item(idx)

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode='{mode}' inválido. Valores válidos: {_VALID_MODES}"
            )
        return mode

    def _validate_target(self, target: str) -> str:
        parts = target.split('+')
        invalid = set(parts) - self.__valid_targets
        if invalid:
            raise ValueError(
                f"Tipo(s) de alvo inválido(s): {invalid}. "
                f"Valores válidos: {self.__valid_targets}"
            )
        return target

    # ------------------------------------------------------------------
    # Pre-loading
    # ------------------------------------------------------------------

    def _preload_all(self) -> None:
        """
        Pré-carrega todos os itens em RAM.

        - mode='none'    → armazena (tensor_image, dict_tensor_targets, meta)
        - mode='offline' → aplica o transform UMA vez e armazena tensores
        - mode='online'  → armazena arrays NumPy RAW; transform aplicado em
                           __getitem__ a cada acesso
        """
        print(f"[VFSSImageDataset] Pré-carregando {len(self)} itens "
              f"(mode='{self.mode}')...")
        for idx in range(len(self)):
            if self.mode == 'online':
                # Salva dados brutos (NumPy) para o transform ser aplicado
                # de forma estocástica em cada __getitem__
                self._cache[idx] = self._load_raw(idx)
            else:
                # 'none' ou 'offline': salva resultado final (tensores)
                self._cache[idx] = self._load_item(idx)
        print("[VFSSImageDataset] Pré-carregamento concluído.")

    def clear_cache(self) -> None:
        """Libera o cache manualmente."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Carregamento bruto (NumPy) — usado pelo modo online com preload
    # ------------------------------------------------------------------

    def _load_raw(self, idx: int) -> tuple:
        """
        Carrega imagem e alvos como arrays NumPy, SEM aplicar transformações.
        Retorna (image_np, target_np_dict, meta).
        """
        row = self.video_frame_df.iloc[idx]
        root = get_project_root_directory()

        frame_path = self._resolve_path(root, row.frame_path)
        image = cv2.imread(str(frame_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # (H, W, 3) uint8
        original_dim = image.shape[:2]

        target_path = self._resolve_path(root, row.target_dir)
        target_np = self._load_targets_np(self.target_keys, target_path, original_dim)

        meta = self._build_meta(row, frame_path)
        return image, target_np, meta

    # ------------------------------------------------------------------
    # Carregamento completo de um item (com transform)
    # ------------------------------------------------------------------

    def _load_item(self, idx: int):
        """
        Carrega, transforma e retorna (tensor_image, dict_tensor_targets, meta).
        Ponto de entrada único para modes 'none' e 'offline'.
        """
        image_np, target_np, meta = self._load_raw(idx)

        if self.mode == 'none':
            return self._apply_none_transforms(image_np, target_np, meta)
        else:
            # offline: chamado uma vez no preload
            return self._apply_albumentations(image_np, target_np, meta)

    # ------------------------------------------------------------------
    # Pipeline de transformação — modo 'none'
    # ------------------------------------------------------------------

    def _apply_none_transforms(
        self,
        image_np: np.ndarray,
        target_np: dict,
        meta: dict,
    ) -> tuple:
        """
        Modo 'none': converte para tensor e aplica Resize(output_dim).
        Imagem e máscaras recebem interpolações distintas.
        """
        # Imagem: float32 tensor (C, H, W) + resize bilinear
        image_t = self._to_tensor_tv(image_np)          # (3, H, W) float [0,1]
        image_t = self._resize_none(image_t)

        target_t = {}
        for key, value in target_np.items():
            if key == 'points':
                target_t[key] = torch.tensor(value, dtype=torch.float32)
            else:
                # NumPy (H, W) uint8/float → PIL → tensor → resize nearest
                arr = value if isinstance(value, np.ndarray) else np.array(value)
                # Garante uint8 para PIL
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                t = self._to_tensor_tv(pil)              # (1, H, W)
                t = self._resize_mask_none(t)
                target_t[key] = t

        return image_t, target_t, meta

    # ------------------------------------------------------------------
    # Pipeline de transformação — modos 'online' e 'offline'
    # ------------------------------------------------------------------

    def _apply_albumentations(
        self,
        image_np: np.ndarray,
        target_np: dict,
        meta: dict,
    ) -> tuple:
        """
        Aplica a pipeline Albumentations em imagem e todos os alvos
        simultaneamente, garantindo consistência geométrica.

        Estratégia por tipo de alvo:
        - mask    → passa via 'mask' do A.Compose (uint8, interpolação NEAREST)
        - heatmap → (2, H, W): canal 0 = gaussiana C2, canal 1 = gaussiana C4.
                    Cada canal é replicado para (H, W, 3) e passado como
                    additional_targets={'heatmap_c0': 'image', 'heatmap_c1': 'image'},
                    garantindo interpolação BILINEAR e preservação de float32.
                    Na saída os dois canais são empilhados de volta em (2, H, W).
        - roi     → passa via additional_targets={'roi': 'mask'} (binário,
                    NEAREST correto)
        - points  → passa via keypoint_params; injetado como lista de (x, y)
        """
        aug_input = {'image': image_np}

        # --- mask e roi: tratados como 'mask' (uint8, NEAREST) ---
        for key in ('mask', 'roi'):
            if key in target_np:
                aug_input[key] = self._to_uint8_mask(target_np[key])

        # --- heatmap: (2, H, W) float32 — canal 0=C2, canal 1=C4 ---
        # Albumentations exige 3 canais para targets do tipo 'image'.
        # Cada canal é replicado para (H, W, 3) e passado como heatmap_c0/c1.
        if 'heatmap' in target_np:
            hm = target_np['heatmap']                  # (2, H, W) float32
            for _ch in range(hm.shape[0]):
                _ch_arr = hm[_ch]                      # (H, W)
                aug_input[f'heatmap_c{_ch}'] = np.stack(
                    [_ch_arr, _ch_arr, _ch_arr], axis=-1
                )                                      # (H, W, 3)

        # --- Keypoints (points) ---
        # O Albumentations exige que todo campo declarado em 'label_fields' do
        # KeypointParams esteja presente no aug_input com o mesmo nome e com
        # comprimento igual ao número de keypoints.
        # A classe detecta automaticamente os label_fields configurados no
        # A.Compose do usuário e injeta listas de strings para cada um,
        # evitando o ValueError independentemente de como o usuário configurou
        # o KeypointParams.
        has_points = 'points' in target_np
        if has_points:
            pts = target_np['points']                    # (4, 2) float
            aug_input['keypoints'] = [tuple(p) for p in pts]  # [(x,y), ...]

            # Injeta label_fields declarados pelo usuário no KeypointParams
            kp_processor = getattr(self.transform, 'processors', {}).get('keypoints')
            if kp_processor is not None:
                for field in getattr(kp_processor.params, 'label_fields', []):
                    aug_input[field] = [str(i) for i in range(len(pts))]

        # --- Aplica a pipeline ---
        result = self.transform(**aug_input)

        # --- Extrai resultados ---
        image_out = result['image']  # np.ndarray (H,W,3) uint8 OU tensor (3,H,W)

        # Converte imagem para tensor float [0, 1].
        # Compatível com ToTensorV2 dentro do A.Compose (já retorna tensor)
        # e com pipelines sem ToTensorV2 (retorna ndarray uint8).
        if isinstance(image_out, torch.Tensor):
            # ToTensorV2 já produziu (C, H, W) uint8 ou float
            image_t = image_out.float()
            if image_t.max() > 1.0:
                image_t = image_t / 255.0
        else:
            image_t = torch.from_numpy(
                image_out.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        target_t = {}
        for key in self.target_keys:
            if key == 'points':
                kps = result.get('keypoints', [])
                pts_out = np.array([[p[0], p[1]] for p in kps], dtype=np.float32)
                target_t[key] = torch.tensor(pts_out, dtype=torch.float32)

            elif key == 'mask':
                arr = result.get('mask', aug_input.get('mask'))
                target_t[key] = torch.from_numpy(
                    arr.astype(np.float32) / 255.0
                ).unsqueeze(0)

            elif key == 'roi':
                # roi é uint8 — normaliza para [0, 1]
                arr = result.get('roi')
                target_t[key] = torch.from_numpy(
                    arr.astype(np.float32) / 255.0
                ).unsqueeze(0)

            elif key == 'heatmap':
                # Reconstrói (2, H, W): extrai canal 0 de cada heatmap_cN.
                # Sem /255 — valores já estão na escala original (float32).
                _channels = []
                for _ch in range(2):
                    _hm_out = result[f'heatmap_c{_ch}']  # (H, W, 3)
                    _channels.append(_hm_out[:, :, 0])   # (H, W)
                target_t[key] = torch.from_numpy(
                    np.stack(_channels, axis=0).copy()   # (2, H, W)
                )

        return image_t, target_t, meta

    # ------------------------------------------------------------------
    # Carregamento dos alvos como NumPy (sem transformação)
    # ------------------------------------------------------------------

    def _load_targets_np(
        self,
        target_keys: list,
        path: str,
        original_dim: tuple,
    ) -> dict:
        """
        Carrega todos os alvos como arrays NumPy, na resolução original.
        Pontos são carregados uma única vez mesmo que usados por múltiplos alvos.
        """
        output = {}
        _points_cache = None

        for key in target_keys:
            if key == 'mask':
                pil = self._load_mask(path)
                output[key] = np.array(pil)              # (H, W) uint8

            elif key == 'points':
                if _points_cache is None:
                    _points_cache = self._load_points(path)
                output[key] = _points_cache              # (4, 2) float64

            elif key == 'roi':
                if _points_cache is None:
                    _points_cache = self._load_points(path)
                h, w = original_dim
                roi_pil = generate_roi_from_points(_points_cache, h, w)
                output[key] = np.array(roi_pil)

            elif key == 'heatmap':
                if _points_cache is None:
                    _points_cache = self._load_points(path)
                hm = generate_heatmap_from_points(
                    _points_cache, original_dim, self.sigma
                )
                # Normaliza para [0, 255] uint8 para compatibilidade com
                # Albumentations (que espera uint8 para targets do tipo mask)
                # Preserva como (2, H, W) float32: canal 0 = gaussiana C2,
                # canal 1 = gaussiana C4. Cada canal será replicado para
                # (H, W, 3) em _apply_albumentations (additional_targets='image')
                # garantindo bilinear e sem quantização.
                hm_np = np.array(hm) if not isinstance(hm, np.ndarray) else hm
                output[key] = hm_np.astype(np.float32)  # (2, H, W)

        return output

    # ------------------------------------------------------------------
    # Loaders de arquivo individuais
    # ------------------------------------------------------------------

    def _load_mask(self, path: str, filename: str = 'Mask.tif') -> Image.Image:
        """Carrega a máscara em escala de cinza."""
        full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Máscara não encontrada: {full_path}")
        return Image.open(full_path).convert("L")

    def _load_points(self, path: str, filename: str = 'Results.csv') -> np.ndarray:
        """Carrega e converte os pontos do arquivo CSV."""
        full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Arquivo de pontos não encontrado: {full_path}"
            )
        df = pd.read_csv(full_path)
        if df.empty:
            raise ValueError(f"Arquivo de pontos vazio: {full_path}")

        row = df.iloc[0]
        return get_corners_from_angle(
            row['BX'], row['BY'], row['Width'], row['Height'], row['Angle']
        )

    # ------------------------------------------------------------------
    # Utilitário: leitura de frame diretamente de vídeo
    # ------------------------------------------------------------------

    def load_frame_from_video(self, path: str, frame: int) -> Image.Image:
        """
        Carrega um frame específico de um arquivo de vídeo.

        ⚠️  Este método abre e fecha o vídeo a cada chamada — use apenas para
        utilitários pontuais (inspeção, extração offline), nunca dentro de um
        loop de treinamento. Para treino, prefira frames pré-extraídos em disco.

        Args:
            path (str): Caminho relativo ao arquivo de vídeo.
            frame (int): Índice do frame (1-based).

        Returns:
            PIL.Image: Frame no formato RGB.
        """
        import cv2 as cv

        root = get_project_root_directory()
        full_path = self._resolve_path(root, path)

        cap = cv.VideoCapture(str(full_path))
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {full_path}")

        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if frame <= 0 or frame > total:
            cap.release()
            raise IndexError(
                f"Frame {frame} fora do intervalo [1, {total}] para: {full_path}"
            )

        cap.set(cv.CAP_PROP_POS_FRAMES, frame - 1)
        ok, raw = cap.read()
        cap.release()

        if not ok:
            raise IOError(f"Erro ao ler frame {frame} de: {full_path}")

        return Image.fromarray(cv.cvtColor(raw, cv.COLOR_BGR2RGB))

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta(row, frame_path) -> dict:
        return {
            'frame_id':         int(row.frame_id),
            'video_id':         int(row.video_id),
            'paciente_id':      row.paciente_id,
            'momento':          row.momento,
            'procedimento':     row.procedimento,
            'selected_labeler': row.selected_labeler,
            'frame_path':       str(frame_path),
        }

    @staticmethod
    def _to_uint8_mask(arr) -> np.ndarray:
        """
        Converte um alvo (PIL.Image ou np.ndarray, qualquer dtype) para
        np.ndarray uint8 (H, W), compatível com Albumentations.
        """
        if isinstance(arr, Image.Image):
            arr = np.array(arr)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr

    @staticmethod
    def _resolve_path(root: str, path: str) -> str:
        """
        Normaliza e une o caminho raiz com um caminho relativo,
        removendo prefixos de caminho relativo legados (../ e ..\).
        """
        clean = path.replace("..\\", "").replace("../", "")
        return os.path.join(root, clean)