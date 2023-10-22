from collections import defaultdict
import torch
from torch.utils.data import DataLoader

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from reimnerf.opt import get_opts
from reimnerf.datasets import dataset_dict
from reimnerf.models.nerf import *
from reimnerf.models.rendering import *
from reimnerf.utils import *
from reimnerf.losses import loss_dict
from reimnerf.metrics import *
from collections import defaultdict



class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['color'](coef=1, loss_type=self.hparams.rgb_loss)

        self.compute_normals=False
        if self.hparams.supervise_normals:
            self.compute_normals=True
            self.geom_loss = loss_dict['normal']()

        if self.hparams.supervise_depth:
            self.depth_loss = loss_dict['depth'](levels=self.hparams.depth_loss_levels, loss_type=self.hparams.depth_loss)

        self.embedding_xyz = Embedding(hparams.N_emb_xyz)
        self.embedding_dir = Embedding(hparams.N_emb_dir)
        self.embeddings = defaultdict(lambda : None)
        self.embeddings['xyz'] = self.embedding_xyz
        self.embeddings['dir']= self.embedding_dir

        color_mlp_in = 6*hparams.N_emb_dir+3
        if self.hparams.variant == 'ls_loc':
            self.embedding_light_xyz = Embedding(hparams.N_emb_light_xyz)
            self.embeddings['light_loc'] = self.embedding_light_xyz
            color_mlp_in += (6*hparams.N_emb_light_xyz)+3

        

        self.nerf_coarse = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=color_mlp_in, init_type=hparams.init_type)
        self.models = {'coarse': self.nerf_coarse}

        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=color_mlp_in, init_type=hparams.init_type)
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            compute_normals=self.hparams.visualize_normals,
                            normal_pertrube=self.hparams.normal_perturb)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name  in ['llff', 'reim_json']:
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['depth_ratio'] = self.hparams.depth_ratio
        # TODO check how we can use more than one image for testing
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        
        rays, rgbs = batch['rays'], batch['rgbs']
            
        results = self(rays)
        loss = self.loss(results, rgbs)
        if self.hparams.supervise_depth:
            depths = batch['depths'].squeeze()
            depth_loss = self.depth_loss(results, depths)
            if self.hparams.depth_ratio==0:
                assert depth_loss == 0 
        if self.hparams.supervise_normals:
            normal_loss = self.norm_loss(results)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/rgb_loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        if self.hparams.supervise_depth:
            self.log('train/depth_loss', depth_loss)
            loss += depth_loss
        if self.hparams.supervise_normals:
            self.log('train/normal_loss', normal_loss)
            loss += normal_loss




        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        if self.hparams.supervise_depth:
            depths = batch['depths'].squeeze() # (H*W)

        results = self(rays)
        loss = self.loss(results, rgbs)
        log = {'val_rgb_loss': loss}

        if self.hparams.supervise_depth:
            loss_d = self.depth_loss(results, depths)
            log['val_depth_loss'] =  loss_d
            loss+=loss_d
        if self.hparams.supervise_normals:
            loss_n = self.norm_loss(results)
            log['val_normal_loss'] = loss_n
            loss += loss_n

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            #TODO clean this and separate all fields. 
            vis_lst=[]
            W, H = self.hparams.img_wh
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            vis_lst.append(img_gt)
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            vis_lst.append(img)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W).cpu()) # (3, H, W)
            vis_lst.append(depth)
            if self.hparams.supervise_depth:
                depth_err = visualize_depth_err(results[f'depth_{typ}'].view(H, W).cpu(), depths.view(H, W).cpu())
                vis_lst.append(depth_err)
                depth_gt = visualize_depth(depths.view(H, W))
                vis_lst.append(depth_gt)


            if self.hparams.visualize_normals:
                normals = visualize_normals(results[f'normals_{typ}'].view(H, W, 3)).permute(2,0,1)
                vis_lst.append(normals)
            if self.hparams.visualize_opacity:
                opacity = results[f'opacity_{typ}'].view(H, W, 1).cpu() # (H, W, 1)
                opacity = np.transpose(np.repeat(opacity,3,axis=2),(2,0,1))
                vis_lst.append(opacity)
            stack = torch.stack(vis_lst) # (5, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_rgb_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

        if self.hparams.supervise_depth:
            self.log('val/depth_loss',
            torch.stack([x['val_depth_loss'] for x in outputs]).mean())
        if self.hparams.supervise_normals:
            self.log('val/normal_loss',
            torch.stack([x['val_normal_loss'] for x in outputs]).mean())


def main(hparams):
    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
