
import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from .gradRib import GradImRib, makeGaussEdgeFltr, cmptGradIm
from . import gradImSnake

class MSELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = (pred-target).pow(2)
        if weights is not None:
            loss *= weights

        if self.ignore_index is not None:
            loss = loss[target!=self.ignore_index]

        return loss.mean()
    
class MAELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = torch.abs(pred-target)
        if weights is not None:
            loss *= weights

        if self.ignore_index is not None:
            loss = loss[target!=self.ignore_index]

        return loss.mean()

class SnakeFastLoss(nn.Module):
    def __init__(self, stepsz, alpha, beta, fltrstdev, ndims, nsteps,
                 cropsz, dmax, maxedgelen, extgradfac):
        super(SnakeFastLoss, self).__init__()
        self.stepsz = stepsz
        self.alpha = alpha
        self.beta = beta
        self.fltrstdev = fltrstdev
        self.ndims = ndims
        self.cropsz = cropsz
        self.dmax = dmax
        self.maxedgelen = maxedgelen
        self.extgradfac = extgradfac
        self.nsteps = nsteps

        self.fltr = makeGaussEdgeFltr(self.fltrstdev, self.ndims)
        self.fltrt = torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda = False

    def cuda(self):
        super(SnakeFastLoss, self).cuda()
        self.fltrt = self.fltrt.cuda()
        self.iscuda = True
        return self

    def forward(self, pred_dmap, lbl_graphs, crops=None, masks=None, original_shapes=None):
        # pred_dmap is the predicted distance map from the UNet
        # lbl_graphs contains graphs each represent a label as a snake
        # crops is a list of slices, each represents the crop area of the corresponding snake

        pred_ = pred_dmap
        dmapW = torch.abs(pred_).clone()
        gimgW = cmptGradIm(dmapW.detach(), self.fltrt)
        gimg = cmptGradIm(pred_.detach(), self.fltrt)
        gimg *= self.extgradfac
        gimgW *= self.extgradfac
        snake_dmap = []

        # Get the output dimensions from pred_dmap
        output_size = pred_dmap.shape[2:]
        device = pred_dmap.device

        for i, (full_graph, grad_img, grad_width_img) in enumerate(zip(lbl_graphs, gimg, gimgW)):
            if crops:
                crop_slices = crops[i]
                original_shape = original_shapes[i] # Get the full image shape for this item
            else:
                # If no crops, assume pred_dmap is the full image
                crop_slices = [slice(0, s) for s in grad_img.shape[1:]]
                original_shape = pred_dmap.shape[2:]

            s = GradImRib(graph=full_graph, crop=crop_slices, stepsz=self.stepsz, alpha=self.alpha,
                        beta=self.beta, dim=self.ndims, gimgV=grad_img, gimgW=grad_width_img)

            if self.iscuda: 
                s.cuda()

            s.optim(self.nsteps)
            full_dmap = s.render_distance_map_with_widths(original_shape)
            cropped_rendered_dmap = full_dmap[crop_slices]
            if masks is not None:
                current_mask = masks[i] if masks.shape[0] > 1 else masks
                cropped_rendered_dmap = cropped_rendered_dmap * (current_mask == 0).squeeze(0)
            if cropped_rendered_dmap.shape != output_size:
                cropped_rendered_dmap = torch.nn.functional.interpolate(
                    cropped_rendered_dmap.unsqueeze(0).unsqueeze(0),
                    size=output_size,
                    mode='nearest' # or appropriate mode
                ).squeeze(0).squeeze(0)
                
            cropped_rendered_dmap = cropped_rendered_dmap.to(device)
            snake_dmap.append(cropped_rendered_dmap)

        snake_dm = torch.stack(snake_dmap, 0).unsqueeze(1) 
        snake_dm = snake_dm.to(device)     
        loss = ((pred_dmap - snake_dm)**2).mean()
        self.snake = s
        return loss


    
class SnakeSimpleLoss(nn.Module):
    def __init__(self, stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                       cropsz,dmax,maxedgelen,extgradfac):
        super(SnakeSimpleLoss,self).__init__()
        self.stepsz=stepsz
        self.alpha=alpha
        self.beta=beta
        self.fltrstdev=fltrstdev
        self.ndims=ndims
        self.cropsz=cropsz
        self.dmax=dmax
        self.maxedgelen=maxedgelen
        self.extgradfac=extgradfac
        self.nsteps=nsteps

        self.fltr =gradImSnake.makeGaussEdgeFltr(self.fltrstdev,self.ndims)
        self.fltrt=torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda=False

    def cuda(self):
        super(SnakeSimpleLoss,self).cuda()
        self.fltrt=self.fltrt.cuda()
        self.iscuda=True
        return self

    def forward(self, pred_dmap, lbl_graphs, crops=None, masks=None, original_shapes=None):

        pred_ = pred_dmap.detach()
        # Gradient for position (signed distance map)
        gimg = cmptGradIm(pred_, self.fltrt)
        gimg *= self.extgradfac
        # Gradient for width (absolute distance map)
        dmapW = torch.abs(pred_).clone()
        gimgW = cmptGradIm(dmapW, self.fltrt)
        gimgW *= self.extgradfac
        snake_dmap = []
        snake_dmap_initial = []  # Store initial state before optimization

        for i, (l, g, gw) in enumerate(zip(lbl_graphs, gimg, gimgW)):
            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            # Use RibbonSnake for width-aware distance maps
            s = GradImRib(l, crop, self.stepsz, self.alpha, self.beta, self.ndims, g, gw)
            if self.iscuda: s.cuda()

            # Render initial distance map (before optimization)
            dmap_initial = s.render_distance_map_with_widths(g.shape[1:], max_dist=self.dmax)
            dmap_initial = dmap_initial.cpu().numpy() if self.iscuda else dmap_initial.numpy()
            
            # Enhance negative values 2x to match GT
            enhancement_factor = 2.0
            dmap_initial = np.where(dmap_initial < 0, dmap_initial * enhancement_factor, dmap_initial)
            dmap_initial = np.clip(dmap_initial, -self.dmax, self.dmax)
            
            snake_dmap_initial.append(torch.Tensor(dmap_initial).type(torch.float32).cuda())

            # Optimize snake (adjust position and width)
            s.optim(self.nsteps)

            # Render final distance map (after optimization)
            dmap_final = s.render_distance_map_with_widths(g.shape[1:], max_dist=self.dmax)
            dmap_final = dmap_final.cpu().numpy() if self.iscuda else dmap_final.numpy()
            
            # Enhance negative values 2x to match GT
            dmap_final = np.where(dmap_final < 0, dmap_final * enhancement_factor, dmap_final)
            dmap_final = np.clip(dmap_final, -self.dmax, self.dmax)
            
            snake_dmap.append(torch.Tensor(dmap_final).type(torch.float32).cuda())

        snake_dm_initial = torch.stack(snake_dmap_initial,0).unsqueeze(1)
        snake_dm = torch.stack(snake_dmap,0).unsqueeze(1)
        
        # Simple MSE loss (using final optimized snake)
        loss = torch.pow(pred_dmap - snake_dm, 2).mean()
                  
        self.snake=s
        self.gimg=gimg
        self.snake_dm_initial = snake_dm_initial  # Store initial for visualization
        self.snake_dm = snake_dm  # Store final for visualization
        
        return loss
    
    
    