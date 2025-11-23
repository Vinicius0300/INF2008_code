from src.data_extraction.patients import load_patients_metadata_from_csv
from src.data_extraction.labels import get_label_metadata
from src.data_extraction.attributions import get_attributions_metadata
from src.utils import get_script_relative_path, get_project_root_directory

import os
from typing import Optional
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set a random seed for reproducibility
np.random.seed(42)


def __check_labeled_wo_assignment(video_frame_df: pd.DataFrame) -> None:
    """
    Check for labeled frames without corresponding attributions metadata and print a warning if any are found.

    Parameters:
        video_frame_df (pd.DataFrame): DataFrame containing labeled frames metadata.
    """
    labeled_wo_assigment = video_frame_df[video_frame_df._merge == "right_only"]

    qtt_labeled_wo_assignment = labeled_wo_assigment.shape[0]
    any_labeled_wo_assignment = qtt_labeled_wo_assignment > 0
    if any_labeled_wo_assignment:
        print(
            f"Warning: There are {qtt_labeled_wo_assignment} labeled frames without corresponding attributions metadata."
        )
        print(labeled_wo_assigment)

    return not any_labeled_wo_assignment

def __select_eligible_labelers(row):
    possible_labels = []
    for i in range(2):
        if row[f'l{i+1}_exist_and_labeled']:
            possible_labels.append(row['labelers'][i])
    return possible_labels

def __get_target_path(row):
        if row['selected_labeler'] == row['labelers'][0]:
            return row['l1_file_path']
        elif row['selected_labeler'] == row['labelers'][1]:
            return row['l2_file_path']
        else:
            return None

def __sample_labeler(video_frame_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Sample one labeler randomly from the eligible labelers for each video frame.

    Parameters:
        video_frame_df (pd.DataFrame): Video frame DataFrame containing eligible labelers.
    
    Returns:
        pd.DataFrame: Video frame DataFrame with an additional column 'selected_labeler' containing the sampled labeler.
    '''
    video_frame_df['eligible_labelers'] = video_frame_df.apply(__select_eligible_labelers, axis=1)
    video_frame_df['selected_labeler'] = video_frame_df['eligible_labelers'].apply(lambda x: np.random.choice(x) if len(x) > 0 else None)    
    
    return video_frame_df


def create_video_frame_metadata_from_label_and_attributions(
    label_metadata: pd.DataFrame, attributions_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a DataFrame containing metadata for labeled frames by merging label and atribution metadata.

    Parameters:
        label_metadata (pd.DataFrame): DataFrame containing label metadata.
        attributions_metadata (pd.DataFrame): DataFrame containing atribution metadata.

    Returns:
        pd.DataFrame: DataFrame containing labeled frames metadata.
    """

    print(f"Shape of attributions metadata df: {attributions_metadata.shape}")
    print(f"Shape of label metadata df: {label_metadata.shape}")

    video_frame_metadata_df = attributions_metadata.merge(
        label_metadata,
        on=["batch", "labeler", "video_id", "frame_id"],
        how="outer",
        indicator=True,
    )
    print(f"Shape of labeled frames df: {video_frame_metadata_df.shape}")

    # Check for labeled frames without corresponding attributions
    __check_labeled_wo_assignment(video_frame_metadata_df)
    del video_frame_metadata_df["_merge"]

    video_frame_metadata_df["has_mask"] = video_frame_metadata_df["has_mask"].fillna(False)
    video_frame_metadata_df["has_points"] = video_frame_metadata_df["has_points"].fillna(False)

    return video_frame_metadata_df


def aggregate_video_frame_metadata(video_frame_metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate video frames metadata by video frame.

    Parameters:
        video_frame_metadata_df (pd.DataFrame): DataFrame containing labeled frames metadata.

    Returns:
        pd.DataFrame: DataFrame containing statistics about labeled frames.
    """
    video_frame_df = video_frame_metadata_df.groupby(["video_frame"]).agg(
        video_id=("video_id", lambda x: x.iloc[0]),
        frame_id=("frame_id", lambda x: x.iloc[0]),
        batch=("batch", "max"),
        labelers=("labeler", "unique"),
        l1=("labeler", lambda x: x.iloc[0]),
        l2=("labeler", lambda x: x.iloc[1] if len(x) > 1 else None), 
        l1_has_mask=("has_mask", lambda x: x.iloc[0]),
        l1_has_points=("has_points", lambda x: x.iloc[0]),
        l1_file_path=("file_path", lambda x: x.iloc[0]),
        l2_has_mask=("has_mask", lambda x: x.iloc[1] if len(x) > 1 else None),
        l2_has_points=("has_points", lambda x: x.iloc[1] if len(x) > 1 else None),
        l2_file_path=("file_path", lambda x: x.iloc[1] if len(x) > 1 else None),
        qtt_labelers=("labeler", "nunique"),
        has_mask=("has_mask", "min"),
        has_points=("has_points", "min"),
    )

    return video_frame_df.reset_index()


def filter_video_frame(video_frame_df: pd.DataFrame, target: str='mask', labeler_filter_criteria='union', batch: list=None, labeler: list=None) -> pd.DataFrame:
    ''''
    Filter the video frame DataFrame based on the specified target ('mask' or 'points'), batches, labelers, and labeler filter criteria.

    Parameters:

        video_frame_df (pd.DataFrame): Video frame DataFrame containing labelling status.
        target (str): The labeling target. It can be either 'mask' or 'points'. Default to 'mask'.
        bathces (list): List of batches to consider for the split. If None, all batches are considered. For example, ['1', '2', '4'] will consider only the batches 1, 2, and 4.
        labeler (list): List of labelers to consider for the split. If None, all labelers are considered. For example, ['VC', 'CS'] will consider only the labelers VC and CS.
        labeler_filter_criteria (str): Criteria to filter labelers. It can be 'union' or 'intersection'. Default to 'intersection'. 
                For 'intersection', only frames labeled by all specified labelers are considered. 
                    Eg. if labeler=['VC', 'CS'] and labeler_filter_criteria='intersection', only frames labeled by both VC and CS will be considered. 
                
                For 'union', frames labeled by at least one of the specified labelers are considered
                    Eg. if labeler=['VC', 'CS'] and labeler_filter_criteria='union', only frames labeled by either VC or CS will be considered.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered labelling status.
    '''
    # Constraints checks
    if target not in ['mask', 'points']:
        raise ValueError("Target must be either 'mask' or 'points'.")

    if batch is not None and not isinstance(batch, list):
        raise ValueError("Batches must be a list or None.")
    
    if labeler is not None and not isinstance(labeler, list):
        raise ValueError("Labelers must be a list or None.")

    if labeler_filter_criteria not in ['union', 'intersection']:
        raise ValueError("Labeler filter criteria must be either 'union' or 'intersection'.")


    # Default to all batches and labeler if not provided
    if batch is None:
        batch = video_frame_df['batch'].unique().tolist()
    
    if labeler is None:
        unique_l1 = set(video_frame_df['l1'].unique().tolist())
        unique_l2 = set(video_frame_df['l2'].unique().tolist())
        labeler = list(unique_l1.union(unique_l2))
    
    # Filter by batch
    video_frame_df_filtered = video_frame_df[
        (video_frame_df['batch'].isin(batch))
    ].copy()

    for i in range(2):
        video_frame_df_filtered[f'l{i+1}_exist_and_labeled'] = (
            (video_frame_df_filtered['labelers'].str[i].apply(lambda li: li in labeler))
            & (video_frame_df_filtered[f'l{i+1}_has_{target}'])
        )

    if labeler_filter_criteria == 'intersection':
        video_frame_df_filtered = video_frame_df_filtered[
            video_frame_df_filtered['l1_exist_and_labeled'] & video_frame_df_filtered['l2_exist_and_labeled']
        ]

    elif labeler_filter_criteria == 'union':
        # Para 'union', nos desejamos verificar se qualquer um dos labelers selecionados
        # rotulou o frame com o target especificado.
        # Se um dos labelers rotulou, mantemos o frame.

        video_frame_df_filtered = video_frame_df_filtered[
            video_frame_df_filtered['l1_exist_and_labeled'] | video_frame_df_filtered['l2_exist_and_labeled']
        ]

    video_frame_df_filtered.reset_index(drop=True, inplace=True)

    return video_frame_df_filtered


def add_target_dir(video_frame_df: pd.DataFrame, target: str) -> pd.DataFrame:
    '''
    Add the target path to the video frame DataFrame.

    Parameters:
        video_frame_df (pd.DataFrame): Video frame DataFrame containing selected labeler and file paths.
        target (str): The labeling target. It can be either 'mask' or 'points'.

    Returns:
        pd.DataFrame: Video frame DataFrame with an additional column 'target_dir' containing the target file path.
    '''
    if not 'selected_labeler' in video_frame_df.columns:
        raise ValueError("The 'selected_labeler' column is missing in the video_frame_df DataFrame.")

    video_frame_df['target_dir'] = video_frame_df.apply(__get_target_path, axis=1)
    return video_frame_df


def add_video_path(video_frame_df: pd.DataFrame, videos_dir: str) -> pd.DataFrame:
    '''
    Add the full video path to the video frame DataFrame.

    Parameters:
        video_frame_df (pd.DataFrame): Video frame DataFrame containing video IDs.
        videos_dir (str): Directory containing the video files.

    Returns:
        pd.DataFrame: Video frame DataFrame with an additional column 'selected_video_path' containing the full video path.
    '''
    video_frame_df['video_path'] = video_frame_df['video_id'].apply(
        lambda vid: os.path.join(videos_dir, f"{vid}.avi")
    )
    return video_frame_df


def are_all_paths_valid(video_frame_df: pd.DataFrame) -> None:
    '''
    For each row in the video frame DataFrame, validate if the target path and video path exist.
    Parameters:
        video_frame_df (pd.DataFrame): Video frame DataFrame containing target and video paths.
    
    Returns:
        bool: True if all paths exist, False otherwise.
    '''
    all_exist = True
    for index, row in video_frame_df.iterrows():
        target_dir = row['target_dir']
        video_path = row['video_path']

        if not os.path.exists(target_dir):
            print(f"Warning: Target path does not exist: {target_dir}")
            all_exist = False

        if not os.path.exists(video_path):
            print(f"Warning: Video path does not exist: {video_path}")
            all_exist = False
    return all_exist

def create_video_frame_df(
    videos_dir : str,
    labels_dir: str,
    frames_dir: str,
    attribtutions_dir: str = None,
    patients_metadata_path : str = '../data/metadados/patients_metadata.csv',
    batch : Optional[list[str]] = None,
    labeler: Optional[list[str]] = None,
    labeler_filter_criteria: str = 'union',
    target: str = 'mask'
) -> pd.DataFrame:
    """
    Create labeled frames metadata DataFrame from the given path by extracting label and atribution metadata.

    Parameters:
        videos_dir (str): 
            Path to the directory containing video files. Video files are expected to be named as '{video_id}.avi'.

        labels_dir (str): 
            Path to the directory containing label folders (and atribution files for convention). Expect 2 type of artifacts:
                1. batch folders (e.g., batch-1, batch-2, etc) containing labeler folders with label excels inside.
                2. attributions excels named as 'attributions-{batch}.xlsx' containing all attributes video_frame for the respective batch.
        
        attributions_dir (Optional[str]):
            Path to the directory containing attributions files. If None, it defaults to labels_dir.
        
        patients_metadata_path (str):
            Path to the patients metadata CSV file. If None, it defaults to '../data/metadados/patients_metadata.csv'.
            If the file does not exist, please generate it using the 'patients.py' script.
        
        batch (Optional[list[str]]): 
            List of batches to consider for the split. If None, all batches are considered. For example, ['1', '2', '4'] will consider only the batches 1, 2, and 4.
        
        labeler (Optional[list[str]]): 
            List of labelers to consider for the split. If None, all labelers are considered. For example, ['VC', 'CS'] will consider only the labelers VC and CS.
        
        labeler_filter_criteria (str): 
            Criteria to filter labelers. It can be 'union' or 'intersection'. Default to 'intersection'. 
                For 'intersection', only frames labeled by all specified labelers are considered. 
                    Eg. if labeler=['VC', 'CS'] and labeler_filter_criteria='intersection', only frames labeled by both VC and CS will be considered.
                For 'union', frames labeled by at least one of the specified labelers are considered
                    Eg. if labeler=['VC', 'CS'] and labeler_filter_criteria='union', only frames labeled by either VC or CS will be considered.
        
        target (str):
            The labeling target. It can be either 'mask' or 'points'. Default to 'mask'.


    Returns:
        video_frame_df (pd.DataFrame)
            DataFrame containing labeled frames metadata.
    """
    final_columns = ['video_frame', 'video_id', 'frame_id', 'batch', 'fonte_dados', 'paciente_id', 'momento', 'procedimento', 'selected_labeler', 'video_path', 'frame_path', 'target_dir']

    # Load label and attributions metadata
    label_metadata_df = get_label_metadata(labels_dir)
    attributions_metadata_df = get_attributions_metadata(labels_dir)

    # Create video frame metadata
    video_frame_metadata_df = create_video_frame_metadata_from_label_and_attributions(
            label_metadata_df, attributions_metadata_df
        )

    # Aggregate video frame metadata
    video_frame_df = aggregate_video_frame_metadata(video_frame_metadata_df)

    # Filter video frame metadata
    video_frame_df = filter_video_frame(
        video_frame_df, 
        labeler=labeler,
        labeler_filter_criteria=labeler_filter_criteria, 
        batch=batch,
        target=target
    )

    # Sample labeler from labelers with valid target for each frame
    video_frame_df = __sample_labeler(video_frame_df)

    video_frame_df = add_target_dir(video_frame_df, target)
    video_frame_df = add_video_path(video_frame_df, videos_dir)

    if not are_all_paths_valid(video_frame_df):
        raise FileNotFoundError("Some target or video paths do not exist. Please check the warnings above.")
    
    patients_metadata_df = load_patients_metadata_from_csv()
    
    video_frame_df = video_frame_df.merge(patients_metadata_df, on='video_id', how='left')
    video_frame_df["frame_path"] = video_frame_df.apply(
        lambda row: os.path.join(frames_dir, f"v{row['video_id']}_f{row['frame_id']}.png"),
        axis=1
    )
    return video_frame_df[final_columns]


def save_video_frame_metadata_to_csv(video_frame_df: pd.DataFrame, filename: str = 'video_frame_metadata.csv', output_dir: str=None) -> None:
    '''Save video frame DataFrame to a CSV file.'''
    relative_metadata_dir = 'data/metadados/'
    
    if output_dir is None:
        root_dir = get_project_root_directory()
        relative_metadata_dir = relative_metadata_dir.replace("..\\", "").replace("../", "")
        output_dir = os.path.join(root_dir, relative_metadata_dir)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to CSV
    output_path =  os.path.join(output_dir, filename)
    video_frame_df.to_csv(output_path, index=False)
    print(f"Video frame Metadata saved to {output_path}")



def load_video_frame_metadata_from_csv(path=None, filename: str = 'video_frame_metadata.csv', input_dir: str= None) -> pd.DataFrame:
    '''Load video frame metadata DataFrame from a CSV file.'''
    relative_metadata_dir = 'data/metadados/'

    if path is None:
        if input_dir is None:
            root_dir = get_project_root_directory()
            relative_metadata_dir = relative_metadata_dir.replace("..\\", "").replace("../", "")
            input_dir = os.path.join(root_dir, relative_metadata_dir)

        # Construct the full path to the CSV file
        path = os.path.join(input_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    video_frame_df = pd.read_csv(path)

    return video_frame_df

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Create video frame metadata DataFrame and save it to a CSV file.")
    parser.add_argument('--videos-dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument('--labels-dir', type=str, required=True, help='Path to the directory containing label folders and attributions files.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save the video frame metadata CSV file. Defaults to data/metadados/.')
    parser.add_argument('--filename', type=str, default='video_frame_metadata.csv', help='Name of the output CSV file. Default is video_frame_metadata.csv.')
    parser.add_argument('--batch', type=str, nargs='*', default=None, help='List of batches to consider for the split. If not provided, all batches are considered.')
    parser.add_argument('--labeler', type=str, nargs='*', default=None, help='List of labelers to consider for the split. If not provided, all labelers are considered.')
    parser.add_argument('--labeler-filter-criteria', type=str, choices=['union', 'intersection'], default='union', help="Criteria to filter labelers. Can be 'union' or 'intersection'. Default is 'union'.")
    parser.add_argument('--target', type=str, choices=['mask', 'points'], default='mask', help="The labeling target. Can be either 'mask' or 'points'. Default is 'mask'.")
    parser.add_argument('--dry_run', action='store_true', help='If set, the script will only create the DataFrame and print its shape without saving it to CSV.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    video_frame_df = create_video_frame_df(
        videos_dir=args.videos_dir,                            # Path to videos directory
        labels_dir=args.labels_dir,                           # Path to labels directory
        frames_dir = args.frames_dir,
        batch=args.batch,                                     # List of batches to consider
        labeler=args.labeler,                                 # List of labelers to consider
        labeler_filter_criteria=args.labeler_filter_criteria, # Labeler filter criteria
        target=args.target                                    # Target: 'mask' or 'points'
    )
    print(f'There are {video_frame_df.shape[0]} labeled frames in the dataset.')
    if args.dry_run:
        print("Dry run mode: DataFrame created but not saved to CSV.")
    else:
        save_video_frame_metadata_to_csv(video_frame_df, filename=args.filename, output_dir=args.output_dir)