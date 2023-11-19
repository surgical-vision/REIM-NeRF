import numpy as np
import reimnerf.datasets.preprocessing.transforms_3d as transforms_3d
from pathlib import Path
import cv2
import open3d as o3d
import json
import shutil
from tqdm import tqdm 

# Calibration files were generated using opencv and the provided calibration images from the C3VD dataset



class ReimNeRFDataset():

    def load_dataset(self):
        """this functions loads , poses, images and pointclouds"""
        raise NotImplementedError

    def _combine_pointclouds(self):
        combined = []
        for ptc, pose in zip(self.frame_pointclouds, self.poses):
            
            pcd = o3d.geometry.PointCloud()
            # express the pointcloud in the world frame of reference
            pcd.points = o3d.utility.Vector3dVector(transforms_3d.transform_left_ptcloud(ptc, pose))
            # limit duplicate points. Most defiantly this needs to be done after combining 
            # the current pointcloud with the rest but we are living it like this because 
            # this is how data were generated during the paper's preparation.
            pcd =pcd.voxel_down_sample(voxel_size=0.005)
            
            combined.append(np.array(pcd.points))
        
        combined = np.concatenate(combined, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined)
        pcd =pcd.voxel_down_sample(voxel_size=0.005) # can be removed if the previous voxel down_sample is fixed
        self.pointcloud = np.array(pcd.points)


    def _get_ray_directions(self, pix_offset=0.5, coordinates='opengl', overwrite_calib = None):
        """ returns ray ray directions based on the calibration parameters and the camera model 
        specified. The calibration parameters are assumed to be in the opencv format. 
        coordinate parameters specify the coordinate system in which the ray directions will be expressed.
        opencv: x right, y down, z forward; opengl: x right, y up, z backward.
        pix_offset: offset to add to the pixel coordinates. This is used to specify if that the rays
        are casted from the center of the pixels."""

        if overwrite_calib is not None:
            calib = overwrite_calib
        else:
            calib = self.calib

        h,w = calib['h'], calib['w']
        fx, fy = calib['fx'], calib['fy']
        cx, cy = calib['cx'], calib['cy']
        K= np.array([[fx,0,cx],
                    [0, fy, cy],
                    [0,0,1]])
        
        pixel_loc = np.mgrid[0:w, 0:h].transpose(2, 1, 0).astype(np.float32)+pix_offset
        pixel_loc = pixel_loc.reshape(-1, 1, 2) # h*w x 1 x 2 (x,y)
        
        
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
            ray_dirs = cv2.fisheye.undistortPoints(pixel_loc, K, D).squeeze()
        
        # convert to homogeneous coordinates and change coordinate system
        if coordinates=='opencv':
            ray_dirs = np.hstack((ray_dirs,np.ones((ray_dirs.shape[0],1))))
        elif coordinates =='opengl':
            ray_dirs = np.hstack((ray_dirs[:,:1], -ray_dirs[:,-1:], -np.ones((ray_dirs.shape[0],1))))
        else:
            raise NotImplementedError
        return ray_dirs
        
    def _rescale_calib(self, h_new, w_new):
        """this function is used to adjust the original calibration parameters to work with resized
        images."""
        h_old, w_old = self.calib['h'], self.calib['w']
        scale_h = h_new/h_old
        scale_w = w_new/w_old
        self.calib['w'] = w_new
        self.calib['h'] = h_new
        self.calib['fx']*= scale_w
        self.calib['fy']*= scale_h
        self.calib['cx']*= scale_w
        self.calib['cy']*= scale_h

    def _normalize_dataset(self, cube_len=2):
        """computes the transformation by which the pointcloud needs to be manipulated 
        in order to fit inside a (cube_len x cube_len x cube_len cube), positioned in the origin.
        This transformation is important because the positional encoding used will not work properly for points
        outside the range (-pi,pi). The default cube_len ensures that all known points are expressed
        in range (-1,1)"""
        if not self.frame_pointclouds:
            raise ValueError
        # center the pointcloud
        points_mean = np.nanmean(self.pointcloud,axis=0)
        self.center_geom_T = np.eye(4)
        self.center_geom_T[:-1,-1]=-points_mean
        self.pointcloud = transforms_3d.transform_left_ptcloud(self.pointcloud, self.center_geom_T)

        # scale the pointcloud to fit inside a cube with dimensions cube_len^3
        self.scale_factor = (0.5*cube_len)/np.nanmax(np.abs(self.pointcloud))
        self.pointcloud *= self.scale_factor

        # downsample the pointcloud again in the new scale 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pointcloud)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        self.pointcloud= np.array(pcd.points)

        # scale depthmaps, bounds and frame_pointclouds.
        # no need to apply T because the pointclouds themselves are are expressed in the image frame of reference 
        for i in range(len(self.frame_pointclouds)):
            self.frame_pointclouds[i] *= self.scale_factor
            self.near_bounds[i] *= self.scale_factor
            self.far_bounds[i] *= self.scale_factor
            self.depthmaps[i] *= self.scale_factor
            self.distmaps[i] *= self.scale_factor

        # transform poses based on scale factor and center transformation 
        self.poses = [self.center_geom_T@pose for pose in self.poses]
        for i in range(len(self.poses)):
            self.poses[i][:3,-1] *= self.scale_factor 

    def export_reim(self, dst_dir, start=0, stop=-1, step=1, save_images=True, suffix=''):
        '''
        reim-nerf format data convention. The dataset follows a json format similar to 
        the one used in blender, instant-ngp etc. 
        it includes fields related to camera calibration and then a field containing frame
        information. the frames field will include the relative filepath of an image, the pose 
        of an image expressed in c2w coordinates following the opengl convention
        optionally a path to distmap information.
        This function assumes that the following lists and arrays are populated:
        image_paths
        poses
        calib
        bounds
        scene_scale
        depthmap_paths(optional)
        depthmaps(optional)
        depthmap_scale(optional)
        '''
        #TODO: implement sanity check to make sure all rays and bounds end up within an cube with len pi
        if stop==-1:
            stop=len(self.image_paths)
        if stop<=start:
            raise ValueError

        # create the output file structure
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_img_dir = dst_dir/'images'
        out_img_dir.mkdir(exist_ok=True, parents=True)
        if save_images and self.distmaps:
            out_dist_dir = dst_dir/'distmaps'
            out_dist_dir.mkdir(exist_ok=True, parents=True)


        schema={
            'camera_model': self.calib['model'],
            'fx' : self.calib['fx'],
            'fy' : self.calib['fy'],
            'cx' : self.calib['cx'],
            'cy' : self.calib['cy'],
            'k1': self.calib['k1'],
            'k2': self.calib['k2'],
            'k3': self.calib['k3'],
            'k4': self.calib['k4'],
            'p1': self.calib['p1'],
            'p2': self.calib['p2'],
            'w' : self.calib['w'],
            'h' : self.calib['h'],
            'camera_angle_x' : 2 * np.arctan(self.calib['w'] / (2 * self.calib['fx'])),
            'camera_angle_y' : 2 * np.arctan(self.calib['h'] / (2 * self.calib['fy'])),
            'scene_scale': self.scale_factor,
        }
        # additional information for the dataset, for instance sparse depth etc. 
        if hasattr(self, 'metainfo'):
            for k, v in self.metainfo.items():
                schema[k]=v
                if k =='rgb_mask':
                    cv2.imwrite(str(dst_dir/v), self.rgb_mask)
        
        # save the transformation we did to the scene before scaling it. This is 
        # useful in case we need to reconstruct in the original scene
        
        if self.center_geom_T is not None:
            schema['scene_T'] = self.center_geom_T.tolist()

        # this is the scale factor to multiply the depthmap.png values
        # it is set to np.pi as in a normalized scene we cannot have depth more than np.pi
        if save_images:
            # scale factor used to save distmaps as pngs.
            schema['distmap_scale'] = np.pi

        frames = []
        # generate information for every frame
        for sample_i in range(start, stop, step):
            src_img_path = Path(self.image_paths[sample_i])

            dst_img_path = out_img_dir/(src_img_path.name)
            frame={'file_path':str(dst_img_path.relative_to(dst_dir))}
            dst_dist_path = out_dist_dir/f"{src_img_path.stem}.png"
            if self.distmaps:
                frame['distmap_path']=str(dst_dist_path.relative_to(dst_dir))
            if save_images:
                shutil.copy(src_img_path, dst_img_path)
                if self.distmaps:
                    self._save_distmap(dst_dist_path, self.distmaps[sample_i], distmap_scale=schema['distmap_scale'])
            frame['near']=self.near_bounds[sample_i]
            frame['far']=self.far_bounds[sample_i]
            # no need to change pose format because the canonical format we use 
            # is opencv c2w
            frame['transform_matrix']=self.poses[sample_i].tolist()
            frames.append(frame)
            
            
        schema['frames']=frames

        json_name = 'transforms'
        if suffix:
            json_name+='_'+suffix
        
        with open(dst_dir/f'{json_name}.json', "w") as outfile:
            json.dump(schema, outfile, indent=4)



    def _save_distmap(self, path, distmap, distmap_scale=np.pi):
        """saves distance maps as png images. To maintain good precision with this format
        we normalize the distance maps with the distmap_scale parameter and then multiply 
        expressing them in a range(0,1). Then we multiply 2**16-1 maintain decimal points
        while saving as uint16. This is a common technique using by other depth datasets
        such as KITTI and allows for easy preview"""
        distmap = ((distmap/distmap_scale)*(2**16-1)).astype(np.uint16)
        cv2.imwrite(str(path), distmap)


class C3VD(ReimNeRFDataset):
    def __init__(self, data_path, start=0, stop=-1,step=1, undistort=False, old_format=False):
        self.dataset_dir = Path(data_path) 
        self.start=start
        self.stop=stop
        self.step=step
        self.old_format = old_format

        self.data_dir = Path(data_path)
        self.poses_path = self.data_dir/'pose.txt'
        self.calib_path = Path(__file__).parents[3]/'resources'/'c3vd_calib.json'
        assert self.calib_path.exists()

        self.far_bounds_scaling = 1.1
        self.near_bounds_scaling = 0.9

        # data provided with the dataset
        self.image_paths=[]
        self.poses=[]
        self.calib=dict()


        # data we compute
        self.depthmaps=[]
        self.distmaps=[]
        self.frame_pointclouds=[]
        self.near_bounds=[]
        self.far_bounds=[]
        self.pointcloud=None
        self.scale_factor=1.0
        self.center_geom_T=np.eye(4)
        self.undistort = undistort
        if self.undistort: 
            self.img_tmp_dir = Path('./tmp_endomapper_dir')
            self.img_tmp_dir.mkdir(exist_ok=True, parents=True)
        self.load_dataset()
        
        # an rgb mask is used to ignore rgb and pixel information 
        # in the periphery of the fisheye frames, where calibration is
        # expected to be less accurate.
        self.metainfo={'dist_ftype':'png',
                       'distmap':'dense',
                       'rgb_mask':'rgb_mask.png'}
        
    def cleanup(self):
        # cleanup temp files if needed
        if self.undistort:
            for img_p in self.image_paths:
                if img_p.parents[0].name == 'tmp_endomapper_dir':
                    img_p.unlink()
            self.undistort = None


    def load_dataset(self):        
        # make a list of all the keys in images to be used from the loading functions
        # in order to load data in the correct order
        self.load_data() # depthmaps have np.nan values in place of zero depth pixels
        self._compute_bounds() # np.nan values are ignored
        
        self._combine_pointclouds()

    def _construct_rgb_masks(self):
        """construct mask to ignore rgb values during optimization and evaluation"""
        # img = cv2.imread(str(self.image_paths[0]))
        # self.rgb_mask = np.all(img<=5, axis=-1).astype(np.uint8)*255
        # if not self.undistort:
        #     # remove the white box on the bottom right
        #     self.rgb_mask[1000:,1250:]=255
        centre_xy = (int(self.calib['cx']), int(self.calib['cy']))

        min_x_dist = np.min([self.calib['cx'],self.calib['w']-self.calib['cx']])
        min_y_dist = np.min([self.calib['cy'],self.calib['h']-self.calib['cy']])
        max_radius = np.max([min_x_dist,min_y_dist])
        
        self.rgb_mask = np.ones((self.calib['h'], self.calib['w'],3), dtype=np.uint8)*255
        cv2.circle(self.rgb_mask, centre_xy, radius = int(max_radius-20), color=0, thickness=cv2.FILLED)

        
    def colmap_pose_to_T(self, img_data):
        transform = np.eye(4)
        transform[:3,:3] = img_data.qvec2rotmat()
        transform[:3,-1] = img_data.tvec.reshape(-1)
        return transform
    

    def load_data(self):
        # enlist path from images and depthmaps
        self.image_paths = list([p for p in self.data_dir.iterdir() if p.name.endswith('color.png')])
        self.image_paths = sorted(self.image_paths, key= lambda x: int(x.name.split('_')[0]))

        self.depth_paths = list([p for p in self.data_dir.iterdir() if p.name.endswith('depth.tiff')])
        self.depth_paths = sorted(self.depth_paths, key= lambda x: int(x.name.split('_')[0]))


        # read calibration and construct undistort maps if needed
        with open(self.calib_path, 'r') as f:
            self.calib = json.load(f)
            self.calib['p1']=0
            self.calib['p2']=0
        self._construct_rgb_masks()
        if self.undistort:
            original_K = np.array([[self.calib['fx'], 0, self.calib['cx']],
                            [0, self.calib['fy'], self.calib['cy']],
                            [0,0,1]])
            h,w = self.calib['h'], self.calib['w']

            if 'FISHEYE' in self.calib['model']:
                original_D = np.array([self.calib['k1'],
                            self.calib['k2'],
                            self.calib['k3'],
                            self.calib['k4']])
                balance = 0
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(original_K, original_D, (w,h), np.eye(3), balance=balance)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(original_K, original_D, np.eye(3), K_new, (w,h), cv2.CV_32FC1)
            else:
                # not tested yet
                original_D = np.array([self.calib['k1'],
                            self.calib['k2'],
                            self.calib['p1'],
                            self.calib['p2']])
                alpha = 0
                K_new = cv2.getOptimalNewCameraMatrix(original_K, original_D, (w,h), np.eye(3), alpha=alpha)[0]
                map1, map2 = cv2.initUndistortRectifyMap(original_K, original_D, np.eye(3), K_new, (w,h), cv2.CV_32FC1)
            undistostorted_calib = {
                            'k1':0,
                            'k2':0,
                            'k3':0,
                            'k4':0,
                            'p1':0,
                            'p2':0,
                            'fx': K_new[0,0],
                            'fy': K_new[1,1],
                            'cx': K_new[0,2],
                            'cy': K_new[1,2],
                            'h': self.calib['h'],
                            'w': self.calib['w'],
                            'model' :'PINHOLE'}
            self.rgb_mask=np.zeros((self.calib['h'], self.calib['w'],3))

        # generate ray direction which are going to be used to unproject the depthmaps to pointclouds.
        if not self.undistort:
            ray_dirs = self._get_ray_directions(coordinates='opencv', pix_offset=0.5) # (hw x 3)
        else:
            ray_dirs = self._get_ray_directions(coordinates='opencv', pix_offset=0.5, overwrite_calib=undistostorted_calib) 

        for i in tqdm(range(len(self.image_paths))):

            if self.undistort:
                # save a copy of the rgb image
                image = cv2.imread(str(self.image_paths[i]))
                undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cv2.imwrite(str(self.img_tmp_dir/self.image_paths[i].name), undistorted_image)
                self.image_paths[i] = self.img_tmp_dir/self.image_paths[i].name

            # 
            depthmap = self._read_depthmap(self.depth_paths[i])# (h xw)
            if self.undistort:# this hsould work if depthmaps express z depth.
                depthmap = cv2.remap(depthmap, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.cv2.INTER_LINEAR)

            
            depthmap[self.rgb_mask[...,0]!=0]=np.nan
            
            # unproject depthmap to pointcloud
            frame_ptcloud = depthmap.reshape(-1,1)*ray_dirs #(hw x 3)
            
            distmap = np.linalg.norm(frame_ptcloud, axis=-1).reshape(self.calib['h'], self.calib['w'])
            
            frame_ptcloud = frame_ptcloud[~np.any(np.isnan(frame_ptcloud), axis=1)]# remove nan points
            frame_ptcloud = frame_ptcloud@ np.diag((1,-1,-1))# convert to opengl

            self.frame_pointclouds.append(frame_ptcloud[::1000])# reduce the resolution to manage
            # pointcloud size. 
            self.depthmaps.append(depthmap)
            self.distmaps.append(distmap)

        # the loaded poses seem to be in opencv c2w format
        poses_c2w = np.loadtxt(self.poses_path, delimiter=',').reshape(-1,4,4) # poses seem to express c2w transforms
        if not self.old_format:
            poses_c2w = poses_c2w.transpose(0,2,1) # c3vd has changed the format of their pose file. loading poses from txt, now need to be transposed. 
        
        self.poses = poses_c2w@np.diag((1,-1,-1,1)) #convert to opengl

        if self.undistort:
            # update calibration 
            self.calib = undistostorted_calib

    def _compute_bounds(self):
        for dist in self.distmaps:
            # we need to bounds based on cartesian distance and not z distance
            # which depthmaps typically encode. this is because the bounds will
            # be used to configure nerf's render distance
            self.far_bounds.append(np.nanmax(dist)*self.far_bounds_scaling)
            self.near_bounds.append(np.nanmin(dist)*self.near_bounds_scaling)


    def _read_depthmap(self, file_path):
        # the c3vd dataset follows the format discribed bellow
        # depthmaps are stored as uint_16 .tiff
        # values between 0-(2**16-1) are map to 0-100 mm
        depth = np.array(cv2.imread(str(file_path),-1))
        
       
        depth = depth.astype(np.float32)
        depth = ((depth)/(2**16-1))*100
        #handle cases where depth is missing
        # replace zero value with nan
        depth[depth==0]=np.nan
        assert depth.shape==(self.calib['h'], self.calib['w'])
        return depth






# helper functions to manipulate the json files generated by the first class and create 
# eval, train, test splits. 
def remove_json_fames(p, pop_step=10, skip_first=False):
    with open(p, 'r') as f:
        meta = json.load(f)
    frames = meta['frames']
    f_new = []
    for i, f in enumerate(frames):
        if (i % pop_step==0) and not (skip_first and i==0):
            continue
        f_new.append(f)
    meta['frames'] = f_new
    with open(p, 'w') as f:
        json.dump(meta, f, indent=4)


def keep_only_json_fames(p, keep_step=10, start_from=0):
    with open(p, 'r') as f:
        meta = json.load(f)
    meta['frames'] = meta['frames'][start_from::keep_step]
    with open(p, 'w') as f:
        json.dump(meta, f, indent=4)