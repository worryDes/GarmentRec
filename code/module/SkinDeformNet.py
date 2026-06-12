import torch
import torch.nn as nn
from torch.nn import Module
from smpl_pytorch.util import batch_rodrigues, batch_global_rigid_transformation
import torch_scatter
import torch.nn.functional as F
import numpy as np
from utils import getRt

class SkinDeformNet(Module):
    def __init__(self,smpl):
        super(SkinDeformNet,self).__init__()
        self.smpl=smpl
    def skeleton(self,shapes,require_body=False):
        return self.smpl.skeleton(shapes,require_body)
    def forward(self,ps,JsorShapes,ws,poses,batch,check_rotation=True,is_Rotation=False):
        batch_num=poses.shape[0]
        assert(batch_num==JsorShapes.shape[0])
        if JsorShapes.shape.numel()==batch_num*10:	#is shapes
            Js=self.smpl.skeleton(JsorShapes)
        else:
            Js=JsorShapes

        # Rs = batch_rodrigues(poses.view(-1, 3)).view(-1, 24, 3, 3)
        if poses.numel()==batch_num*24*3:
            Rs = batch_rodrigues(poses.view(-1, 3)).view(-1, 24, 3, 3)            
            Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
        elif poses.numel()==batch_num*24*9:
            #input poses are general matrix
            if not is_Rotation:
                ms=poses.reshape(-1,3,3)
                # use gram schmit regularization
                b1=F.normalize(ms[:,:,0],dim=1)
                dot_prod = torch.sum(b1 * ms[:, :, 1], dim=1, keepdim=True)
                b2 = F.normalize(ms[:, :, 1] - dot_prod * b1, dim=-1)
                b3 = torch.cross(b1,b2,dim=1)
                Rs=torch.stack([b1,b2,b3],dim=-1).reshape(batch_num,24,3,3)
            
                # # TEST: CHANGE POSE
                # Rs[0, 1] = getRt(*torch.tensor([20, -20, 20, 0, 0, 0]), device=Rs.device)[0]
                # Rs[0, 2] = getRt(*torch.tensor([-20, -20, -20, 0, 0, 0]), device=Rs.device)[0]
            
            else:
                Rs=poses.reshape(batch_num,24,3,3)
            Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
        elif poses.numel()==batch_num*24*16:
            A=poses.reshape(batch_num,24,4,4)
            Js_transformed=None
            Rs=None

        # Js_transformed, A = batch_global_rigid_transformation(Rs, Js, self.smpl.parents, rotate_base = False)
        splitl=torch_scatter.scatter(batch.new_ones(batch.numel(),1),batch,dim=0).cpu().numpy().reshape(-1).astype(np.int32).tolist()		
        ws=ws.split(splitl,0)
        T=torch.cat([weight.matmul(a.reshape(24,16)) for weight,a in zip(ws,A)],dim=0)
        T=T.reshape(-1,4,4)
        ps=torch.cat((ps,ps.new_ones(ps.shape[0],1)),dim=-1).unsqueeze(-1)
        ps=torch.matmul(T,ps).squeeze(-1)
        return ps[:,0:3],T,Rs,Js_transformed