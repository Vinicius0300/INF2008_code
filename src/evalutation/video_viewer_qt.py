"""
Visualizador de vídeo em janela separada (PySide6) com inferência em tempo real.

- Cada modelo/fold pode ser ligado ou desligado por checkbox (vários ao mesmo tempo).
- Só os modelos ativos rodam em cada frame.
- Play/pause, slider de frames, passo a passo e tabela de tempos ao vivo.

Uso no notebook (não importa Qt/torch no kernel):
    from src.evalutation.video_viewer_launcher import launch_viewer
    proc = launch_viewer(video_path, list_models_single_stage, list_models_two_stage)

Uso direto (a partir da raiz do projeto):
    python -m src.evalutation.video_viewer_qt --video data\\videos\\x.mp4 ^
        --single data\\model_weights\\UNet_ImageNet\\FocalMSELoss_1ep\\1783093205 ^
        --two-stage "data\\...\\GlobalStageWrapper\\...::data\\...\\LocalStageWrapper\\..."

Atalhos: Espaço = play/pause | ← / → = frame anterior / próximo
"""

import argparse
import faulthandler
import os
import pickle
import re
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.models.models_artigos.resnet_point.two_stage_resnet import compute_crop_box

_FOLD_PAT  = re.compile(r'fold_([1-9])_best\.pth$')
_COLORS    = ['#E63946', '#457B9D', '#2DC653', '#FF9F1C', '#9B5DE5',
              '#F15BB5', '#00BBF9', '#FEE440', '#00F5D4', '#FF6B6B']
_KP_LABELS = ['C2', 'C4']


# ---------------------------------------------------------------------------
# Pré-processamento (igual ao VFSSImageDataset / TestEvaluator)
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
# Inferência
# ---------------------------------------------------------------------------

def _output_dim(config, fallback=(448, 448)):
    od = getattr(config, 'output_dim', None)
    if isinstance(od, (tuple, list)) and len(od) >= 2:
        return int(od[0]), int(od[1])
    if isinstance(od, int):
        return od, od
    return fallback


def _load_model(config, ckpt_path, device):
    model = config.model_class(**getattr(config, 'model_kwargs', {}))
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state)
    return model.to(device).eval()


def _preprocess(frame_bgr, config, device):
    """Igual ao TestEvaluator: cinza [0,1] -> modify_input_fn (ou unsqueeze)."""
    H, W = _output_dim(config)
    t = _frame_to_gray_tensor(frame_bgr, H, W, getattr(config, 'transform_validation', None))
    fn = getattr(config, 'modify_input_fn', None)
    t = fn(t) if fn is not None else t.unsqueeze(0)
    return t.float().to(device)


class ModelRunner:
    """Um checkpoint (single-stage) ou par de checkpoints (two-stage) com carregamento sob demanda."""

    def __init__(self, label, color, config, ckpt, device, config_s2=None, ckpt_s2=None):
        self.label, self.color = label, color
        self.bgr = tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))
        self.config, self.ckpt = config, ckpt
        self.config_s2, self.ckpt_s2 = config_s2, ckpt_s2
        self.device = device
        self.models = None
        self.times = []

    @property
    def loaded(self):
        return self.models is not None

    def load(self):
        if self.models is None:
            m1 = _load_model(self.config, self.ckpt, self.device)
            m2 = _load_model(self.config_s2, self.ckpt_s2, self.device) if self.config_s2 else None
            self.models = (m1, m2)

    def _sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    def predict(self, frame_bgr):
        self.load()
        self._sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            pts = self._two_stage(frame_bgr) if self.config_s2 else self._single(frame_bgr)
        self._sync()
        ms = (time.perf_counter() - t0) * 1000
        self.times.append(ms)
        return pts, ms

    def _single(self, frame):
        model = self.models[0]
        fh, fw = frame.shape[:2]
        H, W = _output_dim(self.config)
        out = model(_preprocess(frame, self.config, self.device))
        if isinstance(out, (tuple, list)) and len(out) == 3:
            pred_roi, pred_hm, pred_kp = out
        else:
            pred_roi, pred_hm, pred_kp = None, out, None
        if pred_kp is not None:
            kp = pred_kp.squeeze(0).cpu()
            pts = [(float(kp[k, 0]), float(kp[k, 1])) for k in range(kp.shape[0])]
        else:
            pts = _extract_kp_from_heatmap(pred_hm, pred_roi)
        return _scale_pts_to_frame(pts, W, H, fw, fh)

    def _two_stage(self, frame):
        m1, m2 = self.models
        fh, fw = frame.shape[:2]
        H1, W1 = _output_dim(self.config)
        H2, W2 = _output_dim(self.config_s2, fallback=(224, 224))

        inp1 = _preprocess(frame, self.config, self.device)
        _, _, pts_s1 = m1(inp1)                                   # (1, K, 2) em W1×H1
        cbox = compute_crop_box(pts_s1, W1).cpu()

        img = inp1.squeeze(0).cpu()
        x1 = max(int(cbox[0][0].item()), 0)
        y1 = max(int(cbox[0][1].item()), 0)
        x2 = max(min(int(cbox[0][2].item()), img.shape[-1]), x1 + 1)
        y2 = max(min(int(cbox[0][3].item()), img.shape[-2]), y1 + 1)
        crop = F.interpolate(img[:, y1:y2, x1:x2].unsqueeze(0).float(), size=(H2, W2),
                             mode='bilinear', align_corners=False).to(self.device)

        _, _, pts_s2 = m2(crop)                                   # (1, K, 2) em W2×H2
        p = pts_s2.squeeze(0).float().cpu()
        pts = [(float(p[k, 0]) / W2 * (x2 - x1) + x1,
                float(p[k, 1]) / H2 * (y2 - y1) + y1) for k in range(p.shape[0])]
        return _scale_pts_to_frame(pts, W1, H1, fw, fh)


def _norm(p):
    """Normaliza separadores de caminho (configs salvos no Windows usam '\\')."""
    return os.path.normpath(str(p).replace('\\', os.sep).replace('/', os.sep))


def _short_name(model_dir):
    parts = _norm(model_dir).split(os.sep)
    if 'model_weights' in parts:
        parts = parts[parts.index('model_weights') + 1:]
    arch    = parts[0] if parts else model_dir
    details = parts[1] if len(parts) > 1 else ''
    return arch, details


def build_runners(single_dirs, two_stage_pairs, device):
    runners, i = [], 0

    for d in single_dirs:
        with open(os.path.join(_norm(d), 'config.pkl'), 'rb') as f:
            cfg = pickle.load(f)
        root = _norm(cfg.checkpoint_dir)
        if not os.path.isdir(root):
            print(f"[AVISO] Diretório não encontrado: {root}")
            continue
        arch, details = _short_name(d)
        for fname in sorted(os.listdir(root)):
            m = _FOLD_PAT.search(fname)
            if m:
                label = f"{arch} | {details} | Fold {m.group(1)}"
                runners.append(ModelRunner(label, _COLORS[i % len(_COLORS)], cfg,
                                           os.path.join(root, fname), device))
                i += 1

    for s1, s2 in two_stage_pairs:
        with open(os.path.join(_norm(s1), 'config.pkl'), 'rb') as f:
            cfg1 = pickle.load(f)
        with open(os.path.join(_norm(s2), 'config.pkl'), 'rb') as f:
            cfg2 = pickle.load(f)
        root1, root2 = _norm(cfg1.checkpoint_dir), _norm(cfg2.checkpoint_dir)
        if not os.path.isdir(root1):
            print(f"[AVISO] Diretório não encontrado: {root1}")
            continue
        _, details = _short_name(s1)
        for fname in sorted(os.listdir(root1)):
            m = _FOLD_PAT.search(fname)
            if not m:
                continue
            ck2 = os.path.join(root2, f"fold_{m.group(1)}_best.pth")
            if not os.path.exists(ck2):
                print(f"[AVISO] Stage 2 não encontrado: {ck2}")
                continue
            label = f"Two-Stage ResNet | {details} | Fold {m.group(1)}"
            runners.append(ModelRunner(label, _COLORS[i % len(_COLORS)], cfg1,
                                       os.path.join(root1, fname), device, cfg2, ck2))
            i += 1
    return runners


# ---------------------------------------------------------------------------
# Worker (thread separada): lê frame e roda os modelos ativos
# ---------------------------------------------------------------------------

class InferenceWorker(QObject):
    resultReady = Signal(int, object, object)    # idx, frame_bgr | None, {i: (pts, ms)}
    status      = Signal(str)

    def __init__(self, video_path, runners):
        super().__init__()
        self.video_path = video_path
        self.runners = runners
        self.cap = None
        self._last = -2

    @Slot(int, list)
    def process(self, idx, active):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        if idx != self._last + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            self._last = -2
            self.resultReady.emit(idx, None, {})
            return
        self._last = idx

        preds = {}
        for i in active:
            r = self.runners[i]
            try:
                if not r.loaded:
                    self.status.emit(f"Carregando {r.label}…")
                    r.load()
                    self.status.emit(f"{r.label} carregado.")
                preds[i] = r.predict(frame)
            except Exception as e:
                traceback.print_exc()
                self.status.emit(f"Erro em {r.label}: {e}")
        self.resultReady.emit(idx, frame, preds)

    @Slot()
    def release(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# Janela principal
# ---------------------------------------------------------------------------

class ViewerWindow(QMainWindow):
    requestFrame = Signal(int, list)

    def __init__(self, video_path, runners):
        super().__init__()
        self.runners = runners
        cap = cv2.VideoCapture(video_path)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self.current, self.playing, self.busy, self.pending = 0, False, False, None
        self._last_frame, self._last_preds, self._pix = None, {}, None
        self._t_last_result = None

        self.setWindowTitle(f"Comparação de modelos — {os.path.basename(video_path)}")
        self.resize(1400, 850)
        self._build_ui()

        # Thread de inferência
        self._worker_thread = QThread(self)
        self.worker = InferenceWorker(video_path, runners)
        self.worker.moveToThread(self._worker_thread)
        self.requestFrame.connect(self.worker.process)
        self.worker.resultReady.connect(self._on_result)
        self.worker.status.connect(self.statusBar().showMessage)
        self._worker_thread.start()

        self.timer = QTimer(self)
        self.timer.setInterval(max(1, int(1000 / self.fps)))
        self.timer.timeout.connect(self._tick)

        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self._step(-1))

        self._request(0)

    # ---------------- UI ----------------
    def _build_ui(self):
        # Vídeo + controles
        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background:#111;")

        self.btn_prev = QPushButton("⏮"); self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_play = QPushButton("▶ Play"); self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next = QPushButton("⏭"); self.btn_next.clicked.connect(lambda: self._step(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_frame = QLabel()
        self.lbl_frame.setMinimumWidth(110)

        ctrl = QHBoxLayout()
        for w in (self.btn_prev, self.btn_play, self.btn_next):
            ctrl.addWidget(w)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.lbl_frame)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.video_label, 1)
        lv.addLayout(ctrl)

        # Painel de modelos
        models_box = QGroupBox("Modelos (ligar/desligar)")
        mv = QVBoxLayout(models_box)
        self.checks, self.ms_labels = [], []
        for i, r in enumerate(self.runners):
            row = QHBoxLayout()
            sw = QLabel(); sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background:{r.color}; border-radius:7px;")
            cb = QCheckBox(r.label)
            cb.toggled.connect(self._on_toggle)
            ms = QLabel("—"); ms.setMinimumWidth(70)
            ms.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(sw); row.addWidget(cb, 1); row.addWidget(ms)
            mv.addLayout(row)
            self.checks.append(cb); self.ms_labels.append(ms)
        if self.checks:   # sem disparar sinal (worker ainda não existe)
            self.checks[0].blockSignals(True)
            self.checks[0].setChecked(True)
            self.checks[0].blockSignals(False)
        mv.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(models_box)

        btns = QHBoxLayout()
        b_all = QPushButton("Todos");  b_all.clicked.connect(lambda: self._set_all(True))
        b_none = QPushButton("Nenhum"); b_none.clicked.connect(lambda: self._set_all(False))
        btns.addWidget(b_all); btns.addWidget(b_none)

        self.chk_labels = QCheckBox("Mostrar rótulos C2/C4"); self.chk_labels.setChecked(True)
        self.chk_labels.toggled.connect(self._redraw)

        # Tabela de tempos
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Modelo", "Média (ms)", "Mediana (ms)", "FPS"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(180)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideMiddle)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(scroll, 2)
        rv.addLayout(btns)
        rv.addWidget(self.chk_labels)
        rv.addWidget(QLabel("Tempo de inferência (frames processados até agora):"))
        rv.addWidget(self.table, 1)

        split = QSplitter()
        split.addWidget(left); split.addWidget(right)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        split.setSizes([1000, 400])
        self.setCentralWidget(split)

    # ---------------- Controle de reprodução ----------------
    def _active(self):
        return [i for i, cb in enumerate(self.checks) if cb.isChecked()]

    def _request(self, idx):
        idx = max(0, min(idx, self.n_frames - 1))
        if self.busy:
            self.pending = idx
            return
        self.busy = True
        self.requestFrame.emit(idx, self._active())

    def _tick(self):
        if not self.playing or self.busy:
            return
        nxt = self.current + 1
        if nxt >= self.n_frames:
            self._set_playing(False)
        else:
            self._request(nxt)

    def _set_playing(self, on):
        self.playing = on
        self.btn_play.setText("⏸ Pause" if on else "▶ Play")
        (self.timer.start if on else self.timer.stop)()

    def _toggle_play(self):
        if not self.playing and self.current >= self.n_frames - 1:
            self.current = -1
        self._set_playing(not self.playing)

    def _step(self, d):
        self._set_playing(False)
        self._request(self.current + d)

    def _on_slider(self, v):
        if v != self.current:
            self._request(v)

    def _on_toggle(self, _=None):
        if not self.playing:
            self._request(self.current)

    def _set_all(self, on):
        for cb in self.checks:
            cb.blockSignals(True); cb.setChecked(on); cb.blockSignals(False)
        self._on_toggle()

    # ---------------- Resultado ----------------
    @Slot(int, object, object)
    def _on_result(self, idx, frame, preds):
        self.busy = False
        if frame is None:
            self._set_playing(False)
        else:
            self.current = idx
            self._last_frame, self._last_preds = frame, preds
            self.slider.blockSignals(True); self.slider.setValue(idx); self.slider.blockSignals(False)
            self.lbl_frame.setText(f"{idx} / {self.n_frames - 1}")

            for i, lbl in enumerate(self.ms_labels):
                lbl.setText(f"{preds[i][1]:.1f} ms" if i in preds else "—")
            self._redraw()
            self._update_table()

            now = time.perf_counter()
            if self.playing and self._t_last_result is not None:
                disp_fps = 1.0 / max(now - self._t_last_result, 1e-6)
                total = sum(ms for _, ms in preds.values())
                self.statusBar().showMessage(
                    f"Exibição: {disp_fps:.1f} FPS | inferência total no frame: {total:.1f} ms "
                    f"| modelos ativos: {len(preds)}")
            self._t_last_result = now

        if self.pending is not None:
            p, self.pending = self.pending, None
            self._request(p)

    def _redraw(self, *_):
        if self._last_frame is None:
            return
        img = self._last_frame.copy()
        h, w = img.shape[:2]
        r  = max(4, int(min(h, w) * 0.008))
        fs = max(0.45, min(h, w) / 1100)
        th = max(1, int(round(fs * 2)))
        for i, (pts, _) in self._last_preds.items():
            color = self.runners[i].bgr
            for j, (x, y) in enumerate(pts):
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
                if self.chk_labels.isChecked():
                    lbl = _KP_LABELS[j] if j < len(_KP_LABELS) else str(j)
                    org = (x + r + 4, y - r - 4)
                    cv2.putText(img, lbl, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
                    cv2.putText(img, lbl, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._show_pix()

    def _show_pix(self):
        if self._pix is not None:
            self.video_label.setPixmap(self._pix.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show_pix()

    def _update_table(self):
        rows = [r for r in self.runners if r.times]
        self.table.setRowCount(len(rows))
        for row, r in enumerate(sorted(rows, key=lambda r: np.mean(r.times))):
            t = np.asarray(r.times)
            vals = [r.label, f"{t.mean():.1f}", f"{np.median(t):.1f}", f"{1000 / t.mean():.1f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setToolTip(r.label)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, c, it)

    def closeEvent(self, e):
        self.timer.stop()
        self._worker_thread.quit()
        self._worker_thread.wait(3000)
        super().closeEvent(e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--single', action='append', default=[])
    ap.add_argument('--two-stage', action='append', default=[], help="DIR_STAGE1::DIR_STAGE2")
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    # Se algo travar, o traceback de todas as threads vai para o log a cada 60 s
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    print(f"[viewer] Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"cuda={torch.cuda.is_available()}")
    print(f"[viewer] vídeo: {args.video}")

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    pairs  = [tuple(p.split('::', 1)) for p in args.two_stage]

    app = QApplication(sys.argv)
    runners = build_runners(args.single, pairs, device)
    if not runners:
        QMessageBox.critical(None, "Erro", "Nenhum checkpoint encontrado para os modelos informados.")
        sys.exit(1)
    print(f"{len(runners)} checkpoint(s) disponíveis | device = {device}")

    win = ViewerWindow(args.video, runners)
    win.show()
    faulthandler.cancel_dump_traceback_later()
    print("[viewer] janela aberta.")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
