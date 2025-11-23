from typing import Optional
import argparse
import os
import pandas as pd
import re

from src.utils import get_script_relative_path, get_project_root_directory


def get_attributions_metadata(
    path_to_attributions: str, batch: Optional[list] = None, labeler: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a DataFrame containing metadata for attributions. We assume the following directory structure:

    /atribuicoes-<batch-id>.xlsx

    Inside each atribution file, columns represent labelers and rows represent video frames. 
    This function reads all atribution files in the given directory and extracts metadata.

    Parameters:
        path_to_attributions (str): Path to the directory containing atribution files.

    Returns:
        pd.DataFrame: DataFrame containing atribution metadata.
    """

    files = os.listdir(path_to_attributions)
    attributions_df_list = []

    # For each file in the directory, check if it matches the pattern 'atribuicoes-<batch-id>.xlsx'
    for file in files:
        if file.startswith("atribuicoes-") and file.endswith(".xlsx"):
            # Read the attiribution file into a DataFrame, add a 'batch' column, and append to the list
            batch_id = re.search(r"atribuicoes-(\d+)\.xlsx", file).group(1)
            df_attributions_batch = pd.read_excel(
                os.path.join(path_to_attributions, file)
            )
            df_attributions_batch["batch"] = batch_id
            attributions_df_list.append(df_attributions_batch)

    # Concatenate all batch DataFrames into a single DataFrame
    attributions_df_pivot = pd.concat(attributions_df_list, ignore_index=True)

    # Melt the DataFrame to have columns: 'batch', 'labeler', 'video_frame'
    attributions_metadata_df = attributions_df_pivot.melt(
        id_vars=["batch"], var_name="labeler", value_name="video_frame"
    )
    attributions_metadata_df = attributions_metadata_df.dropna(subset=["video_frame"])

    attributions_metadata_df["video_id"] = (
        attributions_metadata_df["video_frame"].str.split("_").str[0].str[1:]
    )

    attributions_metadata_df["frame_id"] = (
        attributions_metadata_df["video_frame"].str.split("_").str[1].str[1:]
    )

    return attributions_metadata_df
