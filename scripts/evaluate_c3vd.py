import cv2
from tqdm import tqdm

import numpy as np

import kornia
import torch
from pathlib import Path
from reimnerf.datasets.ray_utils import get_ray_directions_ocv
import json
import argparse



def rescale_calib( calib, h_new, w_new):
    h_old, w_old = calib['h'], calib['w']
    scale_h = h_new/h_old
    scale_w = w_new/w_old
    calib['w'] = w_new
    calib['h'] = h_new
    calib['fx']*= scale_w
    calib['fy']*= scale_h
    calib['cx']*= scale_w
    calib['cy']*= scale_h


def make_parser()->argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root_dir', required=True, type=str)
    p.add_argument('--predictions_root_dir', required=True, type=str)
    return p


def main(args):
    # you have your results dataset
    dataset_root_dir = sorted(list(Path('dataset_root_dir').iterdir()))
    results_root_dir = sorted(list(Path('predictions_root_dir').iterdir()))

    eval_h, eval_w = (216, 270)

    agg_dict=dict()

    for dataset in tqdm(dataset_root_dir, desc='datasets:'):
        transforms_path = dataset/'transforms_true_test.json'

        with open(transforms_path, 'r') as f:
            frame_data = json.load(f)


        frame_data['model']=frame_data['camera_model']


        ref_img_paths = [dataset/f['file_path'] for f in frame_data['frames']]
        ref_distmap_paths = [dataset/f['distmap_path'] for f in frame_data['frames']]

        # you need to load images and distmaps 

        ref_images = [cv2.imread(str(p)) for p in ref_img_paths]
        ref_distmaps = [cv2.imread(str(p),-1).astype(np.float32)/(2**16-1)*np.pi for p in ref_distmap_paths]


        # resize them

        ref_images = np.array([cv2.resize(img, (eval_w, eval_h),interpolation=cv2.INTER_LINEAR) for img in ref_images])
        ref_distmaps = np.array([cv2.resize(img, (eval_w, eval_h),interpolation=cv2.INTER_NEAREST) for img in ref_distmaps])

        # mask out pixels

        # resize calib
        rescale_calib(frame_data, eval_h, eval_w)

        # compute normals

        rays = get_ray_directions_ocv(frame_data)
        rays_len = np.linalg.norm(rays, axis=2)



        # convert reference images to tensors 

        ref_images = np.array(ref_images).astype(np.float32)/255 # BHWC
        ref_images = torch.FloatTensor(ref_images).permute(0,3,1,2)# BCHW [0-1]

        ref_depthmaps = ref_distmaps / rays_len # normals need to be of the same dimentions as the last to of ref_depthmaps
        ref_depthmaps = torch.FloatTensor(ref_depthmaps).reshape(len(ref_img_paths),1,-1)


        # metrics 
        # this accepts bchw
        metric_ssim = kornia.metrics.SSIM(window_size=11)


        mask = cv2.imread(str(dataset/frame_data['rgb_mask']), -1).astype(np.uint8)

        mask  = cv2.resize(mask, (eval_w, eval_h), interpolation=cv2.INTER_NEAREST)
        if len(mask.shape)==3:
            mask=mask[...,0]

        mask = (mask==0).reshape(-1)

        mask2 = (ref_depthmaps[0]!=0).numpy().reshape(-1)

        assert mask.shape == mask2.shape

        mask =mask*mask2


        ds_root_dir = results_root_dir/dataset.name


        predicted_img_paths = [ds_root_dir/f"{int(p.name.split('_')[0]):03d}.png" for p in ref_img_paths]
        predicted_distmap_paths = [ds_root_dir/f"depth_{int(p.name.split('_')[0]):03d}.pfm" for p in ref_img_paths]

        assert len(predicted_img_paths) == ref_images.shape[0]
        assert len(predicted_distmap_paths) == ref_distmaps.shape[0]

        # read_images and distmaps

        pred_images = torch.FloatTensor(np.array([cv2.imread(str(p)) for p in predicted_img_paths]).astype(np.float32)/255.0).permute(0,3,1,2) 
        
        
        pred_distmap = np.array([cv2.imread(str(p),-1) for p in predicted_distmap_paths])

        pred_depthmaps = pred_distmap / rays_len
        pred_depthmaps = torch.FloatTensor(pred_depthmaps).reshape(len(predicted_img_paths),1,-1)

        with torch.no_grad():
            tmp_ssim = metric_ssim(pred_images, ref_images)
            tmp_ssim = tmp_ssim.reshape(-1,3,eval_w*eval_h)
            tmp_mssim = torch.mean(torch.mean(tmp_ssim[..., mask], dim=-1), dim=1)# first average pixels and then average channels


            img_se = (pred_images-ref_images)**2 # BxCxHW

            img_se = img_se.reshape(-1,3,eval_w*eval_h)
            img_se = torch.mean(img_se, dim=1)
            img_mse = torch.mean(img_se[..., mask], dim=-1) # Bx1

            img_psnr = 10*torch.log10(1/img_mse) # Bx1

            psnr_std, psnr_avg = torch.std_mean(img_psnr)
            ssim_std, ssim_avg = torch.std_mean(tmp_mssim)




        with torch.no_grad():
            depth_se = (pred_depthmaps-ref_depthmaps)**2 # Bx1xHW

            depth_mse = torch.mean(depth_se[...,mask], dim=-1)/frame_data['scene_scale'] # Bx1

            depth_mse_std, depth_mse_avg = torch.std_mean(depth_mse)


        results = np.array(np.hstack((img_psnr.numpy().reshape(-1,1),
                                        tmp_mssim.numpy().reshape(-1,1),
                                        depth_mse.numpy().reshape(-1,1))))
        np.savetxt(ds_root_dir/'per_frame_psnr_ssim_dmse.csv', results, delimiter=",")


        agg = np.array([[psnr_avg.numpy(), ssim_avg.numpy(), depth_mse_avg.numpy()],
                        [psnr_std.numpy(), ssim_std.numpy(), depth_mse_std.numpy()]])
        

        np.savetxt(ds_root_dir/'aggregated_psnr_ssim_dmse_mean_std.csv', agg, delimiter=",")
        
        agg_dict[Path('predictions_root_dir').name].append(agg)

    print('-'*20)
    print('final_results')
    print('-'*20)
    np.set_printoptions(precision=3)
    print('model \t psnr\t ssim\tdmse')
    for k,v in agg_dict.items():
        # v should a be a list of 2d matrices of aggs
        method_results  = np.array(v) # dataset x stats x metric
        # print(method_results.shape)

        method_avg = np.mean(method_results, axis=0)
        # print(method_avg)
        print(f'{k} | {method_avg[0,0]:.03f}+-({method_avg[1,0]:.03f}) |\t {method_avg[0,1]:.03f} +- ({method_avg[1,1]:.03f}) |\t{method_avg[0,2]:.03f}+-({method_avg[1,2]:.03f})')

        np.savetxt(results_root_dir/f'{k}_psnr_ssim_dmse.csv', method_avg, delimiter=',')



if __name__ == "__main__":
    p = make_parser()
    main(p.parse_args())
