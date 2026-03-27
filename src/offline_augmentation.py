import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.utils import (
    get_project_root_directory,
    resolve_path,
    load_points
)

def point_in_image(point, img_shape):
    if point[0] >= 0 and point[0] <= img_shape[0]-1:
        if point[1] >= 0 and point[1] <= img_shape[1]-1:
            return True
    return False

def valid_points(points, img_shape):
    for point in points:
        if not point_in_image(point, img_shape):
            return False
    return True

def debug_incorrect_transformed_image(row, image, img_aug, keypoints, kp_aug):
    print(f"\n[DEBUG] Pontos inválidos em: v{row.video_id}_f{row.frame_id} - batch {row.batch}")
    print(f"ROTULADOR: {row.selected_labeler}")
    print(f"Keypoints - Real: {keypoints} | Image Shape - Real: {image.shape}")
    print(f"Keypoints: {kp_aug} | Image Shape: {img_aug.shape}")
    
    # Prepara a imagem para plot (garante que seja 2D para o imshow)
    temp_img = np.squeeze(img_aug)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(temp_img, cmap='gray')
    
    # Plotar os pontos
    # Se houver pontos, separa x e y. Se estiver vazio, não plota nada.
    if len(kp_aug) > 0:
        xs, ys = zip(*kp_aug)
        plt.scatter(xs, ys, c='red', marker='x', s=100, label='Keypoints Aug')
    
    # Desenha bordas da imagem para ver o "off-limits"
    h, w = temp_img.shape
    plt.axvline(0, color='blue', linestyle='--')
    plt.axvline(w, color='blue', linestyle='--')
    plt.axhline(0, color='blue', linestyle='--')
    plt.axhline(h, color='blue', linestyle='--')
    
    plt.title(f"DEBUG: Pontos Fora da Imagem (v{row.video_id}_f{row.frame_id})")
    plt.legend()
    plt.show() # O loop vai pausar aqui até você fechar a janela do gráfico

def augmentate_single_frame(row, transform, abs_dir, n_aug, root, save_dir, debug):
    
    new_rows = []

    frame_path = resolve_path(root, row.frame_path)
    image = np.array(Image.open(frame_path).convert("L"))
    
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    keypoints_path = resolve_path(root, row.target_dir)
    keypoints = load_points(keypoints_path)

    for i in tqdm(range(n_aug), desc="Augmentations", leave=False):

        # Garante que os pontos gerados estejam dentro da imagem
        redo_transformation = True
        while redo_transformation:
            transformed = transform(image=image, keypoints=keypoints)
            img_aug = transformed["image"]
            kp_aug = transformed["keypoints"]
            if valid_points(kp_aug, img_aug.shape):
                redo_transformation = False
            else:
                if debug == True:
                    debug_incorrect_transformed_image(row, image, img_aug, keypoints, kp_aug)

        img_aug_name = f"v{row.video_id}_f{row.frame_id}_aug{i+1}.png"
        img_aug_path = os.path.join(save_dir, img_aug_name)
        img_aug_path_abs = os.path.join(abs_dir, img_aug_name)

        # garante numpy
        if not isinstance(img_aug, np.ndarray):
            img_aug = img_aug.cpu().numpy()

        # garante formato (H, W)
        img_aug = np.squeeze(img_aug)

        # salvar com PIL se ainda não estiver 
        if not os.path.exists(img_aug_path_abs):
            Image.fromarray(img_aug.astype("uint8")).save(img_aug_path_abs)

        new_rows.append({
            "frame_path": img_aug_path,
            "keypoints": kp_aug
        })
    
    return new_rows

def generate_offline_dataset(df, transform, save_dir, n_aug=5, debug = False):

    root = get_project_root_directory()
    abs_dir = os.path.join(root, save_dir)
    os.makedirs(abs_dir, exist_ok=True)

    existing_files = set(os.listdir(abs_dir))
    new_rows = []
    
    if debug:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processando frames"):
            new_rows = augmentate_single_frame(row, transform, abs_dir, n_aug, root, save_dir, debug)
        
    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                 executor.submit(augmentate_single_frame, row, transform, abs_dir, n_aug, root, save_dir, False) 
                 for _, row in df.iterrows()
            ]
            for f in tqdm(futures, desc="Processando em Paralelo"):
                new_rows.extend(f.result())
        
    dfAug = pd.DataFrame(new_rows)
    
    return dfAug