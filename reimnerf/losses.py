from torch import nn
import torch
class ColorLoss(nn.Module):
    def __init__(self, coef=1, loss_type='L2'):
        super().__init__()
        self.coef = coef
        if loss_type=='L1':
            self.loss = nn.L1Loss(reduction='mean')
        elif loss_type =='L2':
            self.loss = nn.MSELoss(reduction='mean')
        elif loss_type =='Huber':
            self.loss = nn.HuberLoss(reduction='mean')
        else:
            raise NotImplementedError
        


    def forward(self, inputs, targets):

        valid_rgb_idx = ~torch.isnan(targets)
        # check if there is ground truth of at least one pixel
        if torch.sum(valid_rgb_idx)==0:
            # print('NO depth samples found but depth loss is used')
            return torch.tensor(0.0)
        loss =0.0
        loss = self.loss(inputs['rgb_coarse'][valid_rgb_idx], targets[valid_rgb_idx])
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'][valid_rgb_idx], targets[valid_rgb_idx])
        return self.coef * loss

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        loss = (1-torch.sqrt(torch.abs(torch.sum(inputs['normals_fine']*inputs['normals_fine_pertrube'],axis=1)))).mean()
        # loss = torch.sum(torch.abs(inputs['normals_fine']-inputs['normals_fine_pertrube']),axis=1).mean()

        return loss.sum()


class DepthLoss(nn.Module):
    def __init__(self, coef=1, levels='coarse', loss_type='L1'):
        super().__init__()
        self.coef = coef
        self.levels=levels
        if loss_type=='L1':
            self.loss = nn.L1Loss(reduction='mean')
        elif loss_type =='L2':
            self.loss = nn.MSELoss(reduction='mean')
        elif loss_type =='Huber':
            self.loss = nn.HuberLoss(reduction='mean')
        else:
            raise NotImplementedError

    def forward(self, inputs, targets):
        # we need to filter out the the depth target values for which we do have depth information
        # assume that unknown values are set to np.nan
        # so find the valid indexes and compute the loss for only those indexes
        known_depth_idx = ~torch.isnan(targets)
        # check if there is ground truth of at least one pixel
        if torch.sum(known_depth_idx)==0:
            # print('NO depth samples found but depth loss is used')
            return torch.tensor(0.0)
        loss =0.0
        if self.levels=='coarse' or self.levels=='all':
            loss += self.loss(inputs['depth_coarse'][known_depth_idx], targets[known_depth_idx])
        if self.levels=='fine' or self.levels=='all':
            loss += self.loss(inputs['depth_fine'][known_depth_idx], targets[known_depth_idx])

        return self.coef * loss
               

loss_dict = {'color': ColorLoss, 'normal': NormalLoss, 'depth': DepthLoss}