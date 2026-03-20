import os
import cv2
import pandas as pd
from tqdm import tqdm

from src.utils import (
    get_project_root_directory,
    resolve_path,
    load_points
)

def generate_offline_dataset(df, transform, save_dir, n_aug=5):

    root = get_project_root_directory()
    abs_dir = os.path.join(root, save_dir)
    os.makedirs(abs_dir, exist_ok=True)

    existing_files = set(os.listdir(abs_dir))
    new_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processando frames"):

        frame_path = resolve_path(root, row.frame_path)
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        keypoints_path = resolve_path(root, row.target_dir)
        keypoints = load_points(keypoints_path)

        for i in tqdm(range(n_aug), desc="Augmentations", leave=False):

            transformed = transform(image=image, keypoints=keypoints)

            img_aug = transformed["image"]
            kp_aug = transformed["keypoints"]

            img_name = f"v{row.video_id}_f{row.frame_id}_aug{i+1}.png"
            img_path = os.path.join(save_dir, img_name)
            img_path_abs = os.path.join(abs_dir, img_name)

            # evita sobrescrever
            if img_name not in existing_files:
                cv2.imwrite(img_path_abs, img_aug)
                existing_files.add(img_name)

            new_rows.append({
                "frame_path": img_path,
                "keypoints": kp_aug
            })

    return pd.DataFrame(new_rows)