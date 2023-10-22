import torch
from kornia import create_meshgrid
import numpy as np
import cv2


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)# i corresponds to x, j corresponds to y
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([((i)-W/2)/focal, -((j)-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_ray_directions_ocv(calib, pix_offset=0.5, coordinates='opengl'):
    #TODO check what happens after resizing an image.
    h,w = calib['h'], calib['w']
    fx, fy = calib['fx'], calib['fy']
    cx, cy = calib['cx'], calib['cy']
    K= np.array([[fx,0,cx],
                [0, fy, cy],
                [0,0,1]])
    
    pixel_loc = np.mgrid[0:w, 0:h].transpose(2, 1, 0).astype(np.float32)+pix_offset
    pixel_loc = pixel_loc.reshape(1, -1, 2)
    
    if 'FISHEYE' not in calib['model']:
        #perspective models
        D = np.array([calib['k1'],
                        calib['k2'],
                        calib['p1'],
                        calib['p2']])
        ray_dirs = cv2.undistortPoints(pixel_loc, K, D).squeeze()

    else:
        # fisheye models

        D = np.array([calib['k1'],
                    calib['k2'],
                    calib['k3'],
                    calib['k4']])     
        ray_dirs = cv2.fisheye.undistortPoints(pixel_loc.reshape(1, -1, 2), K, D).squeeze()
    
    if coordinates=='opencv':
        ray_dirs = np.hstack((ray_dirs,np.ones((ray_dirs.shape[0],1))))
    elif coordinates =='opengl':
        ray_dirs = np.hstack((ray_dirs[:,:1], -ray_dirs[:,-1:], -np.ones((ray_dirs.shape[0],1))))
    else:
        raise NotImplementedError
    return torch.FloatTensor(ray_dirs.astype(np.float32).reshape(calib['h'], calib['w'],-1))

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d