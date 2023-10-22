import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms as T
import cv2
from reimnerf.datasets.ray_utils import *
from pathlib import Path
import json


#main reason to do that is to support different camera models and trajectory offsets 


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
        self.spiral_poses = kwargs.get('spiral_poses',False)# generate spiral poses
        self.spiral_r1 = kwargs.get('spiral_radius_1', 1) # generate a spiral trajectory offsetting the camera by spiral_radius
        self.spiral_r2 = kwargs.get('spiral_radius_2', self.spiral_r1) # generate a spiral trajectory offsetting the camera by spiral_radius
        self.spiral_period = kwargs.get('spiral_period', 30) # generate a spiral trajectory offsetting the camera by spiral_radius
        self.spiral_frequency_f1 = kwargs.get('spiral_frequency_1', 1) # frequency to use when generating spiral
        self.spiral_frequency_f2 = kwargs.get('spiral_frequency_2', self.spiral_frequency_f1) # frequency to use when generating spiral

        self.T_offset = kwargs.get('T_offset', (0,0,0)) # offset the camera path by a constant fixed amount 
        
        self.R_offset = kwargs.get('R_offset', (0,0,0)) # rotate the camera by a fixed amount, in angle axis
        
        
        self.decay_distortions = kwargs.get('decay_distortions',False) # render a path but linearly scale the distoriton parameters

        self.remove_distortions = kwargs.get('remove_distortions',False) # render a path but linearly scale the distoriton parameters
        
        self.interpolate_poses = kwargs.get('interpolate_poses', False) # create a custom path between the first and the last pose

        self.read_meta()

    def read_meta(self):
        with open(self.root_dir/f"transforms_{self.split}.json", 'r') as f:
            self.meta = json.load(f)
        frames = self.meta['frames']

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

        self._rescale_calib(calib, self.img_wh[1], self.img_wh[0])
        assert calib['w'] == self.img_wh[0]
        assert calib['h'] == self.img_wh[1]


        # generate pose offsets 
        pose_offsets= np.full((len(frames),4,4), np.eye(4))

        self.spiral_t = np.array(range(len(frames)))/30.0

        poses = np.array([np.array(f['transform_matrix']) for f in frames ])


        if self.interpolate_poses:
            # grab the first and last pose, and create a custom path. 
            # then compute the difference between the word and then compute 
            # the difference between the new and old poses and store it as an offset

            start_pose = np.array(frames[0]['transform_matrix'])[:3,-1]
            last_pose =  np.array(frames[-1]['transform_matrix'])[:3,-1]
            # start_pose = np.eye(4)[:3,-1]
            # last_pose =  (np.linalg.inv(np.array(frames[0]['transform_matrix']))@np.array(frames[-1]['transform_matrix']))[:3,-1]
            print(last_pose.shape)

            interp_segments = np.linspace(0,1, len(frames))
            pose_diff = last_pose-start_pose
            translations = np.array([start_pose + step * pose_diff for step in interp_segments])

            poses[:,:3,-1] = translations


        if self.T_offset != (0,0,0):
            poses[:,:3,-1] += np.array(self.T_offset)
        if self.R_offset != (0,0,0):
            transform = np.eye(4)
            R_mat = cv2.Rodrigues(np.array(self.R_offset))[0]
            transform[:3,:3]=R_mat
            poses = poses@transform
        if self.spiral_poses:
            # compute x,y as a function of two periodic functions
            # dx = self.spiral_r1* np.cos(self.spiral_frequency_1*self.spiral_t)
            # dy = self.spiral_r2* np.sin(self.spiralfrequecy_f2*self.spiral_t)
            # poses[:,0,-1] +=dx
            # poses[:,1,-1] +=dy
            ts = np.linspace(0, len(frames)/30, len(frames))
            r1 = self.spiral_r1
            r2 = self.spiral_r2
            f1 = self.spiral_frequency_f1
            f2 = self.spiral_frequency_f2

            for i, (t,p) in enumerate(zip(ts,poses)):
                poses[i,:3,:3] = p[:3,:3].copy()
                poses[i,0,3] = p[0,3].copy()+(r1*np.cos(f1*t))
                poses[i,1,3] = p[1,3].copy()+(r2*np.sin(f2*t))
                poses[i,2,3] = p[2,3].copy()



        if not self.remove_distortions and self.decay_distortions:
            k1_interp = (np.linspace(1,0, len(frames)))* calib['k1']
            k2_interp = (np.linspace(1,0, len(frames)))* calib['k2']
            k3_interp = (np.linspace(1,0, len(frames)))* calib['k3']
            k4_interp = (np.linspace(1,0, len(frames)))* calib['k4']
            p1_interp = (np.linspace(1,0, len(frames)))* calib['p1']
            p2_interp = (np.linspace(1,0, len(frames)))* calib['p2']
            mean_f = (calib['fx']+calib['fy'])/2
            fx_interp = calib['fx'] - (np.linspace(0,1, len(frames)))* (calib['fx']-mean_f)
            fy_interp = calib['fy'] - (np.linspace(0,1, len(frames)))* (calib['fy']-mean_f)
            centre_x = calib['w']/2
            centre_y = calib['h']/2
            cx_interp = calib['cx'] - (np.linspace(0,1, len(frames)))* (calib['cx']-centre_x)
            cy_interp = calib['cy'] - (np.linspace(0,1, len(frames)))* (calib['cy']-centre_y)


        if self.remove_distortions:
            mean_f = (calib['fx']+calib['fy'])/2
            centre_x = calib['w']/2
            centre_y = calib['h']/2
            calib={
            'h':calib['h'],
            'w':calib['w'],
            'fx':calib['fx'],
            'fy':calib['fy'],
            'cx':calib['cx'],
            'cy':calib['cy'],
            'k1':0,
            'k2':0,
            'k3':0,
            'k4':0,
            'p1':0,
            'p2':0,
            'model':'PINHOLE'
        }

        

        
        ray_directions = get_ray_directions_ocv(calib)

        self.all_rays = []
        self.all_rgbs = []
        self.all_depths = []


        for i, frame_data in enumerate(frames):
            # poses and ray generation

            if not self.remove_distortions and self.decay_distortions:
                calib = {
                'h':calib['h'],
                'w':calib['w'],
                'fx':fx_interp[i],
                'fy':fy_interp[i],
                'cx':cx_interp[i],
                'cy':cy_interp[i],
                'k1':k1_interp[i],
                'k2':k2_interp[i],
                'k3':k3_interp[i],
                'k4':k4_interp[i],
                'p1':p1_interp[i],
                'p2':p2_interp[i],
                'model':calib['model']
                }

                ray_directions = get_ray_directions_ocv(calib)
            pose = poses[i,:3]



            self.poses+=[pose]
            c2w = torch.FloatTensor(pose)
            rays_o, rays_d = get_rays(ray_directions, c2w)

            # bounds
            self.near_bounds.append(frame_data['near']*0.9)
            self.far_bounds.append(frame_data['far']*1.3)



            self.all_rays += [torch.cat([rays_o, rays_d, 
                              torch.tensor(self.near_bounds[-1]).repeat(rays_o.size(0)).reshape(-1,1),
                              torch.tensor(self.far_bounds[-1]).repeat(rays_o.size(0)).reshape(-1,1),
                              torch.ones_like(rays_o)*c2w[:,-1]],# ray origin added
                              1)]



        self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)

        # TODO: refactor below
        if self.split == 'val' or self.split=='test':
            samples = len(self.poses)
            # we need to reshape the samples such that they can be pulled 1 per image
            self.all_rays = self.all_rays.reshape(samples,
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
            return np.min([8, len(self.poses)])  # only validate 8 images (to support <=8 gpus)
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
        # sample = {'rays': self.all_rays[idx],
        #             'rgbs': self.all_rgbs[idx]}
        sample = {'rays': self.all_rays[idx]}
        # if self.distmap_paths:
        #     sample['depths']=self.all_depths[idx]
        return sample
