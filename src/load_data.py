import pandas as pd
import torch
import numpy as np
import os

from vfss_dataset import VFSSImageDataset
from data_extraction.video_frame import create_video_frame_df

def load_vfss_dataset(videos_dir: str, labels_dir: str, target='mask', transform=None, target_transform=None):
    '''Load VFSS dataset from video and mask directories.'''

    video_frame_df = create_video_frame_df(
        videos_dir=videos_dir,
        labels_dir=labels_dir,
        batch=None, # Get all batches
        labeler=None, # Get all labelers
        labeler_filter_criteria="union", 
        target=target
    )
    dataset = VFSSImageDataset(video_frame_df, target=target, transform=transform, target_transform=target_transform)
    return dataset

if __name__ == '__main__':
    video_dir = '../data/videos/'
    mask_dir = '../data/rotulos/anotacoes-tecgraf/all/'
    reindex_file = '../data/metadados/video_dicionary.csv'
    
    vfss_dataset = load_vfss_dataset(video_dir, reindex_file, mask_dir, target='mask')
    print(vfss_dataset) 