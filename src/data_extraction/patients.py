from src.utils import get_project_root_directory
import pandas as pd
import os

def read_video_reindex_file(pacientes_csv_path: str) -> pd.DataFrame:
    '''Load patient information from a CSV file.'''
    pacientes_df = pd.read_csv(pacientes_csv_path)
    return pacientes_df

def extract_patients_w_name_metadata(patients_w_name_metadata_df: pd.DataFrame) -> pd.DataFrame:
    '''Extracts patient video metadata from the file path.'''
    
    # Extract paciente_id and momento_procedimento from the path
    patients_w_name_metadata_df['paciente_id'] = patients_w_name_metadata_df['path'].str.extract(r'Pacientes\\(\d+)')[0].astype(int)
    patients_w_name_metadata_df['momento_procedimento'] = patients_w_name_metadata_df['path'].str.extract(r'\\([^\\]+)\\[^\\]+\.avi')[0]

    # Standardize 'momento_procedimento' values
    patients_w_name_metadata_df['momento_procedimento'] = (
        patients_w_name_metadata_df['momento_procedimento']
        .str.lower()
        .str.replace('_', '')
        .str.replace('-', '_')
        .str.replace(' ', '_')
        .str.replace('ó', 'o')
        .str.replace('é', 'e')
    )

    # Extract 'momento' and 'procedimento'
    patients_w_name_metadata_df['momento'] = patients_w_name_metadata_df['momento_procedimento'].str.extract(r'(pre|pos)')[0]

    patients_w_name_metadata_df['procedimento'] = (
        patients_w_name_metadata_df['momento_procedimento']
        .str.replace('pre', '')
        .str.replace('pos', '')
        .str.strip('_')
        .str.replace('é', 'e')
        .str.replace('micirurgia', 'microcirurgia')
    )
    patients_w_name_metadata_df['procedimento'] = patients_w_name_metadata_df.procedimento.replace({'': 'desconhecido'})

    del patients_w_name_metadata_df['momento_procedimento']

    return patients_w_name_metadata_df

def get_patients_metadata_from_reindex_file(video_reindex_csv_path: str) -> pd.DataFrame:
    '''
    Load video metadata from a CSV file.

    Parameters:
        video_dictionary_csv_path (str): Path to the CSV file containing video metadata.
    Returns:
        pd.DataFrame: DataFrame containing video metadata.
    '''
    
    video_reindex_df = read_video_reindex_file(video_reindex_csv_path)

    video_reindex_df['fonte_dados'] = video_reindex_df['path'].apply(
        lambda x: 'Pacientes' if 'Pacientes\\' in x else 'video100'
    )

    # Process patients with name metadata
    patients_w_name_metadata_df = video_reindex_df[video_reindex_df['fonte_dados'] == 'Pacientes'].copy()
    patients_w_name_metadata_df = extract_patients_w_name_metadata(patients_w_name_metadata_df)

    # Process video100 metadata
    patients_videos100_metadata_df = video_reindex_df[video_reindex_df['fonte_dados'] == 'video100'].copy()
    patients_videos100_metadata_df['momento'] = 'pos'
    patients_videos100_metadata_df['procedimento'] = 'total'

    max_paciente_id = patients_w_name_metadata_df['paciente_id'].max()
    patients_videos100_metadata_df['paciente_id'] = range(max_paciente_id + 1, max_paciente_id + len(patients_videos100_metadata_df) + 1)

    # Combine both metadata DataFrames
    patients_metadata_df = pd.concat([patients_w_name_metadata_df, patients_videos100_metadata_df], ignore_index=True)
    del patients_metadata_df['path']

    # Reindex by video_id
    patients_metadata_df = patients_metadata_df.rename(columns={'idx': 'video_id'})
    patients_metadata_df = patients_metadata_df.set_index('video_id', drop=False)
    patients_metadata_df = patients_metadata_df.sort_index()
    patients_metadata_df.index.name = None

    patients_metadata_df['video_id'] = patients_metadata_df.video_id.astype(str)

    return patients_metadata_df

def save_patients_metadata_to_csv(patients_metadata_df: pd.DataFrame, filename: str = 'patients_metadata.csv', output_dir: str= None):
    '''Save the patients metadata DataFrame to a CSV file.'''
    
    relative_metadata_dir = 'data/metadados/'
    
    if output_dir is None:
        root_dir = get_project_root_directory()
        output_dir = os.path.join(root_dir, relative_metadata_dir)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to CSV
    output_path = os.path.join(output_dir, filename)
    patients_metadata_df.to_csv(output_path, index=False)
    print(f"Patients metadata saved to: {output_path}")


def load_patients_metadata_from_csv(path=None, filename: str = 'patients_metadata.csv', input_dir: str= None) -> pd.DataFrame:
    '''Load the patients metadata DataFrame from a CSV file. If no path is provided, it loads from a default location (data/metadados/patients_metadata.csv).
    
    Parameters:
        path (str): Full path to the CSV file. If provided, this is used directly
        filename (str): Name of the CSV file to load.
        input_dir (str): Directory where the CSV file is located.

    Returns:
        pd.DataFrame: DataFrame containing patients metadata.
    '''
    relative_metadata_dir = 'data/metadados/'

    if path is None:
        if input_dir is None:
            root_dir = get_project_root_directory()
            input_dir = os.path.join(root_dir, relative_metadata_dir)

        # Construct the full path to the CSV file
        path = os.path.join(input_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patients metadata file not found: {path}. Please generate it using the 'patients.py' script with the appropriate reindex file (also called ''reindex dict'').")
    
    patients_metadata_df = pd.read_csv(path, dtype={'video_id': str})
    return patients_metadata_df
    

if __name__ == '__main__':
    video_reindex_csv_path = 'data/metadados/video_dictionary.csv'
    patients_metadata_path = 'data/metadados/patients_metadata.csv'

    if os.path.exists(patients_metadata_path):
        print(f"Patients metadata file already exists at: {patients_metadata_path}. No need to regenerate.")
        exit(0)
    
    if not os.path.exists(video_reindex_csv_path):
        raise FileNotFoundError(f"Video reindex file not found: {video_reindex_csv_path}. Please provide the correct path to generate patients metadata.")
    
    patients_metadata_df = get_patients_metadata_from_reindex_file(video_reindex_csv_path)
    save_patients_metadata_to_csv(patients_metadata_df)
        