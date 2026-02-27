from src.utils import get_corners_from_angle
from src.target.heatmap import generate_heatmap_from_points
from src.target.roi import generate_roi_from_points
from src.utils import get_script_relative_path, get_project_root_directory

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

import os
import pandas as pd
from PIL import Image
import numpy as np


class VFSSImageDataset(Dataset):

    __valid_targets = frozenset({'mask', 'points', 'roi', 'heatmap'})

    def __init__(
        self,
        video_frame_df: pd.DataFrame,
        target: str = 'mask',
        output_dim: tuple = (256, 256),
        transform=None,
        target_transform=None,
        sigma: float = 10,
        preload: bool = True,
    ):
        """
        Args:
            video_frame_df (pd.DataFrame): DataFrame com colunas 'frame_path',
                'frame_id' e 'target_dir'.
            target (str): Tipo(s) de alvo separados por '+'.
                Valores válidos: 'mask', 'points', 'roi', 'heatmap'.
                Exemplo: 'mask+heatmap'.
            output_dim (tuple): Dimensão (H, W) de saída das imagens e alvos.
            transform: Transformação aplicada às imagens. Se None, aplica
                T.Resize(output_dim).
            target_transform: Transformação aplicada aos alvos. Se None, aplica
                T.Resize(output_dim).
            sigma (float): Desvio-padrão da gaussiana usada para gerar heatmaps.
            preload (bool): Se True, carrega todos os itens na RAM durante o
                __init__. Recomendado para datasets pequenos (< 1k frames).
        """
        self.video_frame_df = video_frame_df.reset_index(drop=True).copy()
        self.target = self._validate_target(target)
        self.target_keys = target.split('+')
        self.output_dim = output_dim
        self.sigma = sigma

        # --- Transformações instanciadas uma única vez ---
        self._to_tensor = T.ToTensor()

        self.transform = T.Compose([
            self._to_tensor,
            transform if transform is not None else T.Resize(output_dim),
        ])

        self.target_transform = (
            target_transform if target_transform is not None
            else T.Resize(output_dim, interpolation=T.InterpolationMode.NEAREST)
        )

        # --- Pre-loading em memória ---
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
            f"output_dim={self.output_dim}, "
            f"preloaded={bool(self._cache)})"
        )

    def _repr_html_(self) -> str:
        return self.video_frame_df._repr_html_()

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        return self._load_item(idx)

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    def _validate_target(self, target: str) -> str:
        """Valida se todos os tipos de alvo no target são suportados."""
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
        """Carrega todos os itens do dataset na RAM."""
        print(f"[VFSSImageDataset] Pré-carregando {len(self)} itens em memória...")
        for idx in range(len(self)):
            self._cache[idx] = self._load_item(idx)
        print("[VFSSImageDataset] Pré-carregamento concluído.")

    def clear_cache(self) -> None:
        """Libera a memória do cache manualmente, se necessário."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Carregamento de um item
    # ------------------------------------------------------------------

    def _load_item(self, idx: int):
        """
        Carrega e transforma um único item (imagem + alvo + metadados).
        Separado do __getitem__ para ser reutilizado no _preload_all.
        """
        row = self.video_frame_df.iloc[idx]
        root = get_project_root_directory()

        # --- Imagem ---
        frame_path = self._resolve_path(root, row.frame_path)
        image = Image.open(frame_path).convert("RGB")
        original_dim = (image.height, image.width)  # (H, W)

        # --- Alvo ---
        target_path = self._resolve_path(root, row.target_dir)
        target = self._load_targets(self.target_keys, target_path, original_dim)

        # --- Transformações ---
        image = self.transform(image)
        target = self._apply_target_transforms(target)

        # --- Metadados ---
        meta = {
            'frame_id':        int(row.frame_id),
            'video_id':        int(row.video_id),
            'paciente_id':     row.paciente_id,
            'momento':         row.momento,
            'procedimento':    row.procedimento,
            'selected_labeler': row.selected_labeler,
            'frame_path':      str(frame_path),
        }

        return image, target, meta

    # ------------------------------------------------------------------
    # Transformações de alvo
    # ------------------------------------------------------------------

    def _apply_target_transforms(self, target: dict) -> dict:
        """Converte alvos para tensor e aplica target_transform onde aplicável."""
        for key, value in target.items():
            if key == 'points':
                # Pontos são coordenadas numéricas — não aplicamos Resize
                if not isinstance(value, torch.Tensor):
                    target[key] = torch.tensor(value, dtype=torch.float32)
            else:
                target[key] = self._to_tensor(value)
                if isinstance(self.target_transform, T.Resize):
                    target[key] = self.target_transform(target[key])
        return target

    # ------------------------------------------------------------------
    # Loaders individuais
    # ------------------------------------------------------------------

    def _load_targets(
        self,
        target_keys: list,
        path: str,
        original_dim: tuple,
    ) -> dict:
        """
        Carrega todos os alvos requisitados a partir do diretório `path`.

        Args:
            target_keys (list): Lista de strings com os tipos de alvo.
            path (str): Caminho absoluto para o diretório de alvos.
            original_dim (tuple): Dimensão (H, W) da imagem original, usada
                para gerar ROI e heatmap na escala correta.

        Returns:
            dict: Mapeamento {tipo_alvo: dado_carregado}.
        """
        output = {}
        # Carrega pontos uma única vez se necessário para múltiplos alvos
        _points_cache = None

        for key in target_keys:
            if key == 'mask':
                output[key] = self._load_mask(path)

            elif key == 'points':
                if _points_cache is None:
                    _points_cache = self._load_points(path)
                output[key] = _points_cache

            elif key == 'roi':
                if _points_cache is None:
                    _points_cache = self._load_points(path)
                h, w = original_dim
                output[key] = generate_roi_from_points(_points_cache, h, w)

            elif key == 'heatmap':
                if _points_cache is None:
                    _points_cache = self._load_points(path)
                output[key] = generate_heatmap_from_points(
                    _points_cache, original_dim, self.sigma
                )

        return output

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
            raise FileNotFoundError(f"Arquivo de pontos não encontrado: {full_path}")

        df = pd.read_csv(full_path)
        if df.empty:
            raise ValueError(f"Arquivo de pontos vazio: {full_path}")

        row = df.iloc[0]
        return get_corners_from_angle(
            row['BX'], row['BY'], row['Width'], row['Height'], row['Angle']
        )

    # ------------------------------------------------------------------
    # Utilitário: leitura de frame diretamente de vídeo
    # (mantido para compatibilidade, mas não usado no caminho crítico)
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
    def _resolve_path(root: str, path: str) -> str:
        """
        Normaliza e une o caminho raiz com um caminho relativo,
        removendo prefixos de caminho relativo legados (../ e ..\).
        """
        clean = path.replace("..\\", "").replace("../", "")
        return os.path.join(root, clean)