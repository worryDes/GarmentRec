import torch
import torch.nn as nn
from torch.nn import Module
import os.path as osp
import numpy as np
import openmesh as om
import pickle
from module.GCNs import ResidualAdd, MultiPerceptro, SpiralConv


class GarmentDisplacementNet(Module):
    def __init__(self, imgf_size, gar_latent_size, gartype, spiral_indices_folder, pca_dim=64, step_size=2):
        super(GarmentDisplacementNet,self).__init__()
        
        # spiral_np=np.load(osp.join(osp.dirname(__file__),'../../body_garment_dataset/tmps/%s/spiral_indices_%d.npy'%(gartype,step_size)))
        spiral_np = np.load(osp.join(spiral_indices_folder, gartype, 'spiral_indices_%d.npy' % step_size))
        
        # self.pointMLP=MultiPerceptro([3+3+9+3+imgf_size+gar_latent_size,512,256],False)
        # #final 1:
        # infeature_size=3+3+9+3+imgf_size+gar_latent_size
        infeature_size=3+3+3+9+3+10+pca_dim+imgf_size+gar_latent_size
        self.pointMLP=nn.Sequential(nn.Linear(infeature_size,256,False),nn.ReLU())
        # self.pointMLP=SpiralConv(3+3+9+3+imgf_size+gar_latent_size,256,spiral_np)
        self.res1=ResidualAdd(SpiralConv(256,256,spiral_np),SpiralConv(256,256,spiral_np),nn.ReLU(),nn.ReLU())
        self.midDown=nn.Linear(256,128,False)
        self.ress=nn.ModuleList([ResidualAdd(SpiralConv(128,128,spiral_np),SpiralConv(128,128,spiral_np),nn.ReLU(),nn.ReLU()) for i in range(3)])
        # self.outConv=SpiralConv(256+256,3,spiral_np[:,:,:7])
        # #final 1:
        # self.outMLP=MultiPerceptro([128+128,128,3])
        self.outMLP=MultiPerceptro([256+128+128,256,128,3])
    def forward(self,x):
        # pfs=self.pointMLP(x)
        assert(x.dim()==3)		
        x=self.pointMLP(x)		
        batch_num,vnum,in_size=x.shape
        pfs=torch.cat((x,x.new_zeros(batch_num,1,in_size)),dim=1)
        vnum+=1
        zero_padding=pfs.new_ones(1,vnum,1)
        zero_padding[0,-1,0] = 0.0
        fs=self.midDown(self.res1(pfs,zero_padding=zero_padding))*zero_padding
        gfs,_=torch.max(fs[:,:-1,:],1,keepdim=True)
        for res in self.ress:
            fs=res(fs,zero_padding=zero_padding)
        # out=self.outConv(torch.cat((fs,gfs.expand(batch_num,fs.shape[1],gfs.shape[-1]))*zero_padding,dim=-1),zero_padding=zero_padding)
        out=self.outMLP(torch.cat((pfs,fs,gfs.expand(batch_num,fs.shape[1],gfs.shape[-1])*zero_padding),dim=-1))
        return out[:,:-1,:]