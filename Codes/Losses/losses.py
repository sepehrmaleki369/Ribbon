
import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from gradRib import GradImRib, makeGaussEdgeFltr, cmptGradIm

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

# class SnakeFastLoss(nn.Module):
#     def __init__(self, stepsz,alpha,beta,fltrstdev,ndims,nsteps,
#                        cropsz,dmax,maxedgelen,extgradfac):
#         super(SnakeFastLoss,self).__init__()
#         self.stepsz = stepsz
#         self.alpha = alpha
#         self.beta = beta
#         self.fltrstdev = fltrstdev
#         self.ndims = ndims
#         self.cropsz = cropsz
#         self.dmax = dmax
#         self.maxedgelen = maxedgelen
#         self.extgradfac = extgradfac
#         self.nsteps = nsteps

#         self.fltr = gradImSnake.makeGaussEdgeFltr(self.fltrstdev,self.ndims)
#         self.fltrt = torch.from_numpy(self.fltr).type(torch.float32)

#         self.iscuda = False

#     def cuda(self):
#         super(SnakeFastLoss,self).cuda()
#         self.fltrt = self.fltrt.cuda()
#         self.iscuda = True
#         return self

    # def forward(self,pred_dmap,lbl_graphs,crops=None):
    # # pred_dmap is the predicted distance map from the UNet (why isn't it a probability map???)
    # # lbl_graphs contains graphs each represent a label as a snake (not exactly a snake but a graph which represents a snake) / not snake class
    # # crops is a list of slices, each represents the crop area of the corresponding snake

    #     pred_ = pred_dmap
    #     gimg = gradImSnake.cmptGradIm(pred_,self.fltrt)
    #     gimg *= self.extgradfac
    #     snake_dmap = []

    #     # if index == 0:
    #     #     show(gimg[0][0].detach().numpy(),"gradient of image calculated from cmptGradIm",lbl_graphs[0])
    #     #     plt.colorbar()
    #     #     print(len(lbl_graphs), type(lbl_graphs))
    #     #     print(len(gimg), type(gimg))

    #     for i,lg in enumerate(zip(lbl_graphs,gimg)):
    #         # i is index num
    #         # lg is a tuple of a graph and a gradient image
    #         l = lg[0] # graph
    #         g = lg[1] # gradient image

    #         if crops:
    #             crop = crops[i]
    #         else:
    #             crop=[slice(0,s) for s in g.shape[1:]]
    #         #s = gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,self.beta,self.ndims,g)
    #         gimgW = torch.abs(g).clone()
    #         # s = ribbon.RibbonSnake(graph=l,crop=crop,stepsz=self.stepsz,alpha=self.alpha,
    #         #                          beta=self.beta,gimgN=g,gimgW=gimgW,step_type="original")
    #         # In losses.py, when you create the RibbonSnake:
    #         s = ribbon.RibbonSnake(graph=l, crop=crop, stepsz=self.stepsz, alpha=self.alpha,
    #                    beta=self.beta, gimgN=g, gimgW=gimgW, step_type="original", 
    #                    ndims=3)  # set to 3
    #         if self.iscuda: s.cuda()

    #         s.optim(self.nsteps)
    #         dmap = s.render_distance_map(self.cropsz)
    #         #dmap = s.renderDistanceMap(g.shape[1:],self.cropsz,self.dmax,self.maxedgelen)
    #         snake_dmap.append(dmap)

    #     snake_dm = torch.stack(snake_dmap,0).unsqueeze(1)
    #     loss = torch.pow(pred_dmap-snake_dm,2).mean()
                  
    #     self.snake = s
    #     self.gimg = gimg
        
    #     return loss

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

    def forward(self, pred_dmap, lbl_graphs, crops=None):
        # pred_dmap is the predicted distance map from the UNet
        # lbl_graphs contains graphs each represent a label as a snake
        # crops is a list of slices, each represents the crop area of the corresponding snake

        pred_ = pred_dmap
        gimg = cmptGradIm(pred_, self.fltrt)
        gimg *= self.extgradfac
        snake_dmap = []

        # Get the output dimensions from pred_dmap
        output_size = pred_dmap.shape[2:]
        device = pred_dmap.device

        for i, lg in enumerate(zip(lbl_graphs, gimg)):
            # i is index num
            # lg is a tuple of a graph and a gradient image
            l = lg[0]  # graph
            g = lg[1]  # gradient image

            if crops:
                crop = crops[i]
            else:
                crop = [slice(0, s) for s in g.shape[1:]]

            with torch.no_grad:    
                gimgW = torch.abs(g).clone()
                s = GradImRib(graph=l, crop=crop, stepsz=self.stepsz, alpha=self.alpha,
                            beta=self.beta,dim=self.ndims, gimgV=g, gimgW=gimgW)
                        
                if self.iscuda: 
                    s.cuda()

                s.optim(self.nsteps)
                dmap = s.render_distance_map_with_widths(g.shape)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Ensure the dmap has the right dimensions
                if dmap.shape != output_size:
                    dmap = torch.nn.functional.interpolate(
                        dmap.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                        size=output_size,
                        mode='nearest'
                    ).squeeze(0).squeeze(0)  # Remove the extra dimensions
                    
                # Make sure dmap is on the correct device
                dmap = dmap.to(device)
                snake_dmap.append(dmap)

        snake_dm = torch.stack(snake_dmap, 0).unsqueeze(1)
        if snake_dm.shape != pred_dmap.shape:
            snake_dm = torch.nn.functional.interpolate(
                snake_dm,
                size=pred_dmap.shape[2:],
                mode='nearest'
            )      
        snake_dm = snake_dm.to(device)     
        loss = torch.pow(pred_dmap - snake_dm, 2).mean()
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
    
    
