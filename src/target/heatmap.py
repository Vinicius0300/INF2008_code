import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

def generate_heatmap_from_points(points, orig_size, sigma=10):

    H, W = orig_size
    points = np.array(points)

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = xx[None, :, :]
    yy = yy[None, :, :]

    x = points[:, 0][:, None, None]
    y = points[:, 1][:, None, None]

    heatmaps = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    heatmaps = torch.from_numpy(heatmaps.astype(np.float32))
    heatmaps = heatmaps / (heatmaps.amax(dim=(1,2), keepdim=True) + 1e-8)

    return heatmaps
    

