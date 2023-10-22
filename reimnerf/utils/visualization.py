import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def visualize_depth_err(prediction, ref, cmap=cv2.COLORMAP_HOT):
    error = torch.abs(prediction-ref)
    error[torch.isnan(error)]=0
    return visualize_depth(error, cmap=cmap)

def visualize_normals(normals):
    """
    https://computergraphics.stackexchange.com/questions/10387/how-to-draw-surface-normals-from-surface-normal-maps
    normals: (H, W, 3)
    """
    x = normals.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    x = 0.5*x + 0.5*np.ones_like(x)
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(x)
    x_ = T.ToTensor()(x_).permute(1,2,0) # (H, W,3)
    return x_