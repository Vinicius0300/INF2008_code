import pandas as pd

def calculate_labelling_stats(video_frame_metadata_df: pd.DataFrame) -> pd.DataFrame:
    """'
    Calculate statistics about labeled frames per labeler.

    Parameters:
        video_frame_metadata_df (pd.DataFrame): DataFrame containing labeled frames metadata.

    Returns:
        pd.DataFrame: DataFrame containing statistics about labeled frames per labeler.

    """
    labelling_stats = video_frame_metadata_df.groupby("labeler").agg(
        total_labeled_frames=("frame_id", "count"),
        frames_with_masks=("has_mask", "sum"),
        frames_with_points=("has_points", "sum"),
    )
    labelling_stats["mask_labeled_percentage"] = (
        labelling_stats["frames_with_masks"] / labelling_stats["total_labeled_frames"]
    ) * 100
    labelling_stats["points_labeled_percentage"] = (
        labelling_stats["frames_with_masks"] / labelling_stats["total_labeled_frames"]
    ) * 100
    return labelling_stats