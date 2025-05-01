
import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from .gradRib import GradImRib, makeGaussEdgeFltr, cmptGradIm

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

    def forward(self, pred_dmap, lbl_graphs, crops=None, mask= None, original_shapes=None):
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
            if mask is not None:
                current_mask = mask[i] if mask.shape[0] > 1 else mask
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

    def forward(self,pred_dmap,lbl_graphs,crops=None):
    
        pred_=pred_dmap.detach()
        gimg=gradImSnake.cmptGradIm(pred_,self.fltrt)
        gimg*=self.extgradfac
        snake_dmap=[]

        for i,lg in enumerate(zip(lbl_graphs,gimg)):
            l = lg[0]
            g = lg[1]
            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            s=gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
                                      self.beta,self.ndims,g)
            if self.iscuda: s.cuda()

            s.optim(self.nsteps)

            lbl = np.zeros(g.shape[1:])
            lbl = s.renderSnakeWithLines(lbl)
            if np.sum(lbl) == 0:
                dmap = self.dmax * np.ones(lbl.shape)
            else:
                # the distance map is calculated here from the probability map
                dmap = dist(1-lbl)
                dmap[dmap > self.dmax] = self.dmax
                
            snake_dmap.append(torch.Tensor(dmap).type(torch.float32).cuda())

        snake_dm=torch.stack(snake_dmap,0).unsqueeze(1)
        loss=torch.pow(pred_dmap-snake_dm,2).mean()
                  
        self.snake=s
        self.gimg=gimg
        
        return loss
    
    
