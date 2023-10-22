import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms as T
import cv2
from reimnerf.datasets.ray_utils import *
from pathlib import Path
import json



class REIMNeRFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(256, 256), val_num=1, **kwargs):
        """
        split: training, eval or test split.
        img_wh: dimention to resize the images
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_wh = img_wh
        self.val_num = max(1, val_num) # TODO this should change to eval every ... check why it is not yet coded like this
        self.white_back=False # maintain compatibility with the rendering functions
        self.define_transforms()
        self.depth_ratio = kwargs.get('depth_ratio', 1)
        self.read_meta()

    def read_meta(self):
        with open(self.root_dir/f"transforms_{self.split}.json", 'r') as f:
            self.meta = json.load(f)
        frames = self.meta['frames']

        # construct calib_dict()
        self.image_paths=[]
        self.distmap_paths=[]
        self.near_bounds=[]
        self.far_bounds=[]
        self.poses = []

        # construct calib dict 
        calib={
            'h':self.meta['h'],
            'w':self.meta['w'],
            'fx':self.meta['fx'],
            'fy':self.meta['fy'],
            'cx':self.meta['cx'],
            'cy':self.meta['cy'],
            'k1':self.meta['k1'],
            'k2':self.meta['k2'],
            'k3':self.meta['k3'],
            'k4':self.meta['k4'],
            'p1':self.meta['p1'],
            'p2':self.meta['p2'],
            'model':self.meta['camera_model']
        }

        # read addtional data relevant to some dataset
        try:
            rgb_mask = cv2.imread(str(self.root_dir/self.meta['rgb_mask']),-1)

        except KeyError:
            # construct another black mask
            rgb_mask = np.zeros(self.img_wh[::-1], dtype=np.uint8)
        if len(rgb_mask.shape)==3:
            rgb_mask =rgb_mask[...,0]

        rgb_mask = cv2.resize(rgb_mask, self.img_wh, interpolation=cv2.INTER_NEAREST).reshape(-1)




        invalid_idxs = np.where(rgb_mask==255)# (h*w,1)


        self._rescale_calib(calib, self.img_wh[1], self.img_wh[0])
        assert calib['w'] == self.img_wh[0]
        assert calib['h'] == self.img_wh[1]
        
        self.directions = get_ray_directions_ocv(calib)

        
        self.all_rays = []
        self.all_rgbs = []
        self.all_depths = []

        for frame_data in frames:
            # poses and ray generation
            pose = np.array(frame_data['transform_matrix']).astype(np.float32)[:3, :4]
            self.poses+=[pose]
            c2w = torch.FloatTensor(pose)
            rays_o, rays_d = get_rays(self.directions, c2w)

            # bounds
            self.near_bounds.append(frame_data['near'])
            self.far_bounds.append(frame_data['far'])

            # image
            self.image_paths.append(self.root_dir/frame_data['file_path'])
            img = Image.open(self.image_paths[-1]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3)
            img[invalid_idxs] = torch.nan#fix this

            self.all_rgbs += [img]

            self.all_rays += [torch.cat([rays_o, rays_d, 
                              torch.tensor(self.near_bounds[-1]).repeat(rays_o.size(0)).reshape(-1,1),
                              torch.tensor(self.far_bounds[-1]).repeat(rays_o.size(0)).reshape(-1,1),
                              torch.ones_like(rays_o)*c2w[:,-1]],# ray origin added
                              1)]

            #distance maps
            if 'distmap_path' in frame_data.keys() and self.depth_ratio!=0:
                self.distmap_paths.append(self.root_dir/frame_data['distmap_path'])
                distmap = self.read_depthmap(self.distmap_paths[-1],
                                             resize_hw= self.img_wh[::-1],
                                             dense = (self.meta.get('distmap','dense')=='dense'))# hxw
                assert self.near_bounds[-1]<=np.nanmin(distmap)
                assert np.nanmax(distmap)<=self.far_bounds[-1]
                distmap = self.transform(distmap) # (1, h, w)? check, it may be just (h,w)
                distmap = distmap.view(1, -1).permute(1, 0)
                distmap[invalid_idxs]=torch.nan
                # randomly subsample depth indexes. This works great depthmaps span the whole image
                idx = torch.randperm(distmap.size(0))[:int(distmap.size(0)*(1-self.depth_ratio))]
                distmap[idx]=torch.nan # delete gt samples based on depth ratio choosen
                self.all_depths += [distmap]

        self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        if self.all_depths:
            self.all_depths = torch.cat(self.all_depths, 0) # ((N_images-1)*h*w, 1)
            assert self.all_depths.shape[0] == self.all_rgbs.shape[0]
       
        # TODO: refactor below
        if self.split == 'val' or self.split=='test':
            samples = len(self.poses)
            # we need to reshape the samples such that they can be pulled 1 per image
            self.all_rays = self.all_rays.reshape(samples,
                                                  self.img_wh[0]* self.img_wh[1],
                                                  -1)
            self.all_rgbs = self.all_rgbs.reshape(samples,
                                                  self.img_wh[0]* self.img_wh[1],
                                                  -1)
            if self.distmap_paths:
                self.all_depths = self.all_depths.reshape(samples,
                                                  self.img_wh[0]* self.img_wh[1],
                                                  -1)

    def define_transforms(self):
        self.transform = T.ToTensor()
    
    

    def read_depthmap(self, path, dense=True, resize_hw=None):
        # images are stored as 16bit uint pngs
        # values between 0-(2**16-1) should map to 0-20 cm
        # WARINGING originally we read from 0-20 now we read from 0 to pi
        depth = np.array(Image.open(path)).astype(np.float32)
        depth = ((depth)/(2**16-1))*np.pi
        depth[depth==0]=np.nan

        if resize_hw is not None:
            new_h, new_w = resize_hw
            if dense:
                depth = cv2.resize(depth, 
                                  (int(new_w),int(new_h)),
                                   interpolation=cv2.INTER_NEAREST)
            else:
                #in order to keep those points when resizing, instead ore returning
                # the depthmap, we are going to find the indexes of the non zero values
                rescaled_depth=np.full((new_h, new_w), np.nan, dtype=np.float32)
                scale_y = new_h/depth.shape[0]
                scale_x = new_w/depth.shape[1]
                valid_info_idxs = np.where(~np.isnan(depth)) # (array, array), first array contain row, second col
                new_idxs_y = (valid_info_idxs[0]*scale_y).astype(np.uint)
                new_idxs_x = (valid_info_idxs[1]*scale_x).astype(np.uint)
                rescaled_depth[new_idxs_y, new_idxs_x] = depth[valid_info_idxs[0], valid_info_idxs[1]]
                depth = rescaled_depth
        return depth.astype(np.float32)

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            # return np.min([8, len(self.poses)])  # only validate 8 images (to support <=8 gpus)
            return len(self.poses)  # only validate 8 images (to support <=8 gpus)

        return len(self.poses)
    
    def _rescale_calib(self, calib, h_new, w_new):
        h_old, w_old = calib['h'], calib['w']
        scale_h = h_new/h_old
        scale_w = w_new/w_old
        calib['w'] = w_new
        calib['h'] = h_new
        calib['fx']*= scale_w
        calib['fy']*= scale_h
        calib['cx']*= scale_w
        calib['cy']*= scale_h

    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                    'rgbs': self.all_rgbs[idx]}
        if self.distmap_paths:
            sample['depths']=self.all_depths[idx]
        return sample
