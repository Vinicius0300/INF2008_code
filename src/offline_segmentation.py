import os
import cv2
import pandas as pd
from src.utils import (get_project_root_directory,
                       resolve_path,
                       load_points)

def generate_offline_dataset(df, transform, save_dir, n_aug=5):
    
    root = get_project_root_directory()
    abs_dir = os.path.join(root, save_dir)
    os.makedirs(abs_dir, exist_ok=True)
    new_rows = []

    for idx, row in df.iterrows():

        frame_path = resolve_path(root, row.frame_path)
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        keypoints_path = resolve_path(root, row.target_dir)
        keypoints = load_points(keypoints_path)

        for i in range(n_aug):

            transformed = transform(image=image, keypoints=keypoints)

            img_aug = transformed["image"]
            kp_aug = transformed["keypoints"]

            img_name = f"{idx}_aug_{i}.png"
            img_path = os.path.join(save_dir, img_name)

            cv2.imwrite(img_path, img_aug)

            new_rows.append({
                "frame_path": img_path,
                "keypoints": kp_aug
            })
    return pd.DataFrame(new_rows)