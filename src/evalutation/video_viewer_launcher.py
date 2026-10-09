"""
Lançador do visualizador em janela separada.

Este módulo usa SOMENTE a biblioteca padrão: não importa PySide6, torch nem cv2
no kernel do Jupyter. A janela roda em outro processo Python
(src.evalutation.video_viewer_qt), então o notebook fica livre.
"""
import importlib.util
import os
import subprocess
import sys
import time


def launch_viewer(video_path, list_models_single_stage=(), list_models_two_stage=(),
                  device=None, log_path=None, check_seconds=8):
    """
    Abre o visualizador e retorna o processo (subprocess.Popen).

    Aguarda `check_seconds` s: se o processo terminar nesse intervalo
    (erro de import, caminho errado etc.), mostra o log no notebook.
    """
    if importlib.util.find_spec('PySide6') is None:
        print("PySide6 não está instalado neste ambiente. Rode:  %pip install PySide6==6.11.2")
        return None
    if not os.path.exists(video_path):
        print(f"Vídeo não encontrado: {video_path}")
        return None

    cmd = [sys.executable, '-u', '-X', 'faulthandler',
           '-m', 'src.evalutation.video_viewer_qt', '--video', video_path]
    for d in list_models_single_stage:
        cmd += ['--single', d]
    for s1, s2 in list_models_two_stage:
        cmd += ['--two-stage', f'{s1}::{s2}']
    if device:
        cmd += ['--device', device]

    log_path = os.path.abspath(log_path)
    log = open(log_path, 'w', encoding='utf-8')
    log.write('Comando: ' + ' '.join(f'"{c}"' for c in cmd) + '\n\n')
    log.flush()

    proc = subprocess.Popen(
        cmd, cwd=os.getcwd(),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"Visualizador iniciado (PID {proc.pid}). Log: {log_path}")
    print("Carregando modelos/vídeo — a janela pode levar alguns segundos para aparecer…")

    t0 = time.time()
    while time.time() - t0 < check_seconds:
        if proc.poll() is not None:
            log.close()
            print(f"\nO visualizador terminou com código {proc.returncode}. Log:\n")
            with open(log_path, encoding='utf-8', errors='replace') as f:
                print(f.read())
            return proc
        time.sleep(0.5)
    return proc
