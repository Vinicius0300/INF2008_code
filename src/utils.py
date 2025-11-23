import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_NAME = "vfss-data-split"

def get_script_directory() -> str:
    '''Get the directory of the current script.'''
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return script_dir

def get_project_root_directory() -> str:
    '''Get the root directory of the project (assumed to be the parent of the script directory).'''
    script_dir = get_script_directory()
    
    if not PROJECT_NAME in script_dir:
        raise RuntimeError(f"Cannot determine project root directory. '{PROJECT_NAME}' not found in script path.")

    project_root = script_dir.split(PROJECT_NAME)[0]
    project_root = os.path.join(project_root, PROJECT_NAME)
    return project_root

def get_script_relative_path(relative_path: str) -> str:
    '''Get the absolute path to a file relative to the script's directory.'''
    script_dir = get_script_directory()
    return os.path.join(script_dir, relative_path)

def get_corners_from_angle(x: float, y: float, w: float, h: float, angle_degrees: float):
    '''Get two opposite corners of a rotated rectangle based on its angle.'''
    corners = {
        'top_left': (x, y),
        'top_right': (x + w, y),
        'bottom_right': (x + w, y + h),
        'bottom_left': (x, y + h)
    }
    q1 = angle_degrees > 0 and angle_degrees < 90
    q3 = angle_degrees > -180 and angle_degrees < -90
    if q1 or q3:
        p1 = corners['top_right']
        p2 = corners['bottom_left']
    else:
        p1 = corners['top_left']
        p2 = corners['bottom_right']
    
    points = np.array([p1, p2])
    return points

def draw_mask(image, mask, alpha_mask=0.3, color=(0,255,0)):
    '''Overlay a mask on an image with specified color and transparency.'''
    # Normalize color
    color = np.array(color, dtype='float32')
    color = color/color.max()

    masked_image = image.copy()

    masked_image = np.where(
        mask.astype(int),
        color,
        masked_image
    )

    masked_image = masked_image.astype(np.float32)
    weight_image = 1 - alpha_mask

    return cv.addWeighted(image, weight_image, masked_image, alpha_mask, 0)

def plot_image_with_mask(frame, mask, plot_mask=True, size_inches=8, alpha_mask=0.3, color=(0, 255, 0)):
    '''Plot image with mask overlaid. '''

    # Change CxHxW to HxWxC
    frame_np = frame.permute(1, 2, 0).numpy()

    if plot_mask and mask != None:
        mask_np = mask.permute(1, 2, 0).numpy().astype(bool)
        masked_image = draw_mask(frame_np, mask_np, color=color, alpha_mask=alpha_mask)
    else: 
        masked_image = frame_np

    # Return a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(masked_image, vmin=0, vmax=1)
    fig.set_size_inches(size_inches, size_inches)
    ax = ax.axis('off')
    return ax