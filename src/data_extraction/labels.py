from typing import Optional
import argparse
import os
import pandas as pd
import re

def __is_results_csv_valid(path_to_csv: str) -> pd.DataFrame:
    """
    Validate the results CSV file to ensure it contains the required columns.

    Parameters:
        path_to_csv (str): Path to the results CSV file.

    Returns:
        bool: True if the CSV file is valid, False and raises Warning otherwise.
    """
    
    required_columns = {'BX','BY','Width','Height','Angle'}
    df_results = pd.read_csv(path_to_csv)

    results_columns = set(df_results.columns)
    have_all_columns = required_columns.issubset(results_columns)
    
    if not have_all_columns:
        missing_columns = required_columns - results_columns
        print(f"The results CSV is missing the following required columns: {missing_columns}. Check the file at {path_to_csv}.")

    return have_all_columns


def get_label_metadata(
    path_to_labels, batch: Optional[list] = None, labeler: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a DataFrame containing metadata for labels. We assume the following directory structure:
    /batch-<batch_id>/<labeler_id>/v<video_id>_f<frame_id>/

    Inside each frame directory, there can be mask files (.tif or .tiff) and points files (.csv). This function checks for the presence of these files.

    Parameters:
        path_to_labels (str): Path to the directory containing label files.
        batch (Optional[None, list]): List of batch identifiers to filter labels. Default is None and includes all batches.
        labeler (Optional[None, list]): List of labeler identifiers to filter labels. Default is None and includes all labelers.

    Returns:
        pd.DataFrame: DataFrame containing label metadata.
    """
    label_metadata = {
        "batch": [],
        "labeler": [],
        "video_id": [],
        "frame_id": [],
        "file_path": [],
        "has_mask": [],
        "has_points": [],
    }

    i = 0
    for root, dirs, files in os.walk(path_to_labels):
        # /batch_1/VR/v45_f11
        pattern = rf"{re.escape(os.sep)}batch-(\d+){re.escape(os.sep)}([A-Z]{{2}}){re.escape(os.sep)}v(\d+)_f(\d+)"
        metadata_str = re.search(pattern, root)

        if metadata_str is not None:
            batch_id = metadata_str.group(1)
            labeler_id = metadata_str.group(2)
            video_id = metadata_str.group(3)
            frame_id = metadata_str.group(4)

            if (batch is None or batch_id in batch) and (
                labeler is None or labeler_id in labeler
            ):
                has_points = False
                has_mask = False

                # For each file in the directory, check for mask and points files
                count_mask_files = 0
                count_points_files = 0

                for file in files:
                    if file.endswith(".tif") or file.endswith(".tiff"):
                        has_mask = True
                        count_mask_files += 1

                    if file.endswith(".csv"):
                        path = os.path.join(root, file)
                        has_points = __is_results_csv_valid(path)
                        count_points_files += 1

                if count_mask_files > 1 or count_points_files > 1:
                    print(f"Warning: More than one mask or points file found in {root}")

                label_metadata["batch"].append(batch_id)
                label_metadata["labeler"].append(labeler_id)
                label_metadata["video_id"].append(video_id)
                label_metadata["frame_id"].append(frame_id)
                label_metadata["file_path"].append(root)
                label_metadata["has_mask"].append(has_mask)
                label_metadata["has_points"].append(has_points)

    metadata_df = pd.DataFrame(label_metadata)
    return metadata_df
