from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from IPython.display import display

def print_split_summary(name, video_frame_df, initial_frames_qtt, add_new_line=False):
    n_videos = video_frame_df['video_id'].nunique()
    n_patients = video_frame_df['paciente_id'].nunique()
    n_frames = video_frame_df.shape[0]

    print(f'{name} - Videos: {n_videos} | Patients: {n_patients} | Frames: {n_frames} ({n_frames/initial_frames_qtt:.1%})')
    if add_new_line:
        print()

def create_split_summary(video_frame_split):
    ''''
    Create a summary of the data splits including number of videos, patients, frames, and videos_id in each split.

    Parameters:
        video_frame_split (dict): Dictionary containing the data splits.

    Returns:
        pd.DataFrame: DataFrame summarizing the data splits.
    '''
    split_summary = {
        'split_name': [],
        'n_videos': [],
        'n_patients': [],
        'n_frames': [],
        'percentage_of_initial_frames': [],
        'video_ids': []
    }

    splits = video_frame_split['folds'] + [video_frame_split['test']]
    split_name_folds = [f'fold_{i+1}' for i in range(len(video_frame_split['folds']))]
    split_name = split_name_folds + ['test_set']

    initial_frames_qtt = video_frame_split['video_frame_df'].shape[0]

    for split, name in zip(splits, split_name):
        n_videos = split['video_id'].nunique()
        n_patients = split['paciente_id'].nunique()
        n_frames = split.shape[0]
        percentage_of_initial_frames = n_frames / initial_frames_qtt
        video_ids = split['video_id'].unique().tolist()

        split_summary['split_name'].append(name)
        split_summary['n_videos'].append(n_videos)
        split_summary['n_patients'].append(n_patients)
        split_summary['n_frames'].append(n_frames)
        split_summary['percentage_of_initial_frames'].append(percentage_of_initial_frames)
        split_summary['video_ids'].append(video_ids)

    split_summary_df = pd.DataFrame(split_summary)
    return split_summary_df


def split_data_k_fold(video_frame_df, test_size=0.2, n_folds=5, random_state=42):
    '''Split the labeled frames into training, validation, and test sets based on the specified target ('mask' or 'points').

    Parameters:
        video_frame_df (pd.DataFrame): Video frame DataFrame containing labelling status.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        n_folds (int): Number of folds for cross-validation. Default is 5.

    Returns:
        folds (list of pd.DataFrame): List containing DataFrames for each validation fold.
        video_frame_test (pd.DataFrame): DataFrame containing the test set.
    '''

    desired_columns = [
        'video_frame',
        'video_id',
        'frame_id',
        'selected_labeler',
        'paciente_id',
        'momento',
        'procedimento',
        'frame_path',
        'video_path',
        'target_dir'
    ]
    video_frame_split = {}

    video_frame_split['video_frame_df'] = video_frame_df

    # There is only one patient per video naturally, so when we split by patient_id we are also splitting by video_id
    patient_ids = video_frame_df['paciente_id'].unique()

    # Split off the test set with stratification
    patient_id_folds, patient_id_test = train_test_split(
        patient_ids, 
        test_size=test_size, 
        random_state=random_state
    )

    # Creating folds and test set
    for split_name in ['folds_all', 'test']:
        patients_ids = patient_id_folds if split_name == 'folds_all' else patient_id_test

        video_frame_split[split_name] = (
            video_frame_df[
                video_frame_df
                .paciente_id
                .isin(patients_ids)
            ]
            .reset_index(drop=True)
        )
    
    print('Folds all:', video_frame_split['folds_all'].paciente_id.nunique())
    print('Test:', video_frame_split['test'].paciente_id.nunique())

    # Print dataset sizes
    initial_frames_qtt = video_frame_df.shape[0]
    print_split_summary('Overall Dataset', video_frame_df, initial_frames_qtt)
    print_split_summary('Training + Validation Set', video_frame_split['folds_all'], initial_frames_qtt)
    print_split_summary('Test Set', video_frame_split['test'], initial_frames_qtt, add_new_line=True)
    
    # Create K-Folds
    video_frame_split['folds'] = []
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    kfold_split = kfold.split(patient_id_folds)

    # Generate each fold
    for fold_index, (_, val_indexes) in enumerate(kfold_split):
        patient_id_val = patient_id_folds[val_indexes]
        fold = (
            video_frame_split['folds_all'][
                video_frame_split['folds_all']
                .paciente_id
                .isin(patient_id_val)
            ]
            [desired_columns]
            .reset_index(drop=True)
        )
        video_frame_split['folds'].append(fold)
        print_split_summary(f'Fold {fold_index+1}', fold, initial_frames_qtt)
    
    split_summary = create_split_summary(video_frame_split)
    display(split_summary)

    return video_frame_split['folds'], video_frame_split['test']