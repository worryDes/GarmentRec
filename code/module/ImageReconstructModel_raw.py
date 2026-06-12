import os.path

import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from module.GCNs import ResidualAdd,MultiPerceptro,SpiralConv
from module.SkinWeightModel import SkinWeightNet
from smpl_pytorch.SMPL import SMPL
import numpy as np
import os.path as osp
import pickle
from renderer import Renderer
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.mesh.textures import Textures, TexturesUV
from pytorch3d.structures import Meshes
from render_mesh import load_textured_mesh
from time import sleep
import cv2
from pytorch3d.ops import SubdivideMeshes
from module.encoders import *
from module.decoders import *
from module.SkinDeformNet import SkinDeformNet
from module.GarmentDisplacementNet import GarmentDisplacementNet
from utils import *
from renderer_deca import Pytorch3dRasterizer
from torch.nn.functional import pad
from sklearn.preprocessing import MinMaxScaler
import torchvision
from torch_cluster import knn
from openmesh import read_trimesh
import matplotlib.pyplot as plt
from visualize import vis
from smpl_pytorch.util import batch_rodrigues


class ImageReconstructModel(Module):
    def __init__(self, 
                 SkinWeightNet, 
                 with_classification=False, 
                 check=False, 
                 bcnet_tran_mean=[0, 0, 0], 
                 tran_mean=[0, 0, 0], 
                 garments=[], 
                 garmentvnums=[], 
                 upper_type_num=0, 
                 pca_folder=None, 
                 pca_dim=64, 
                 tex_pca_dim=64, 
                 tex_witdh=256, 
                 tex_height=256, 
                 smpl_model_path=None, 
                 model_type='D3G', 
                 hard_stitch=False, 
                 midpairs=None, 
                 infer_camera=False, 
                 infer_tex=False, 
                 use_detail=True, 
                 inferring=False, 
                 vis_save_folder=None, 
                 mesh_save_folder=None, 
                 create_detail_meshes=False, 
                 subdivide_template_folder=None, 
                 subdivide_template_name=None, 
                 dense_template_folder=None, 
                 tensorboard_logging=False, 
                 writer=None, 
                 neptune_logging=False, 
                 light_instance_scale=1.0, 
                 displacement_scale=None, 
                 upsample_dismap=False, 
                 use_neighbor=False, 
                 device='cuda:0'):
        super(ImageReconstructModel,self).__init__()
        if displacement_scale is None:
            raise NotImplementedError
        self.device = device
        self.pca_dim = pca_dim
        self.pca_folder = pca_folder
        self.imgEncoder=ImageEncoder(tran_mean=tran_mean, infer_camera=infer_camera, light_instance_scale=light_instance_scale)
        self.use_detail = use_detail
        self.n_detail = 128
        if use_detail:
            self.detailEncoder = DetailEncoder(n_detail=self.n_detail)
        self.inferring = inferring
        self.vis_save_folder = vis_save_folder
        self.mesh_save_folder = mesh_save_folder
        # if infer_camera:
        #     self.cameraEncoder = CameraEncoder()
            
        # self.patchEncoder=PatchEncoder()
  
        # self.garments=['shirts','short_shirts','pants','short_pants','skirts','short_skirts']
        # self.garmentvnums=[4248,4258,5327,3721,5404,2818]
  
        # self.garments=['Shirt', 'T-shirt', 'Pants', 'Shorts']
        # self.garmentvnums=[2468,1954,1180,678,678,678]
        
        # self.garments = ['T-shirt', 'front_open_T-shirt', 'Shirt', 'front_open_Shirt', 'Shorts', 'Pants']
        # self.garmentvnums = [1954, 1954, 2468, 2468, 678, 1180]
        
        self.bcnet_tran_mean = bcnet_tran_mean
        self.tran_mean = tran_mean
        self.garments = garments
        self.garmentvnums = garmentvnums
        self.upper_type_num = upper_type_num
        self.lower_type_num = len(garments) - upper_type_num
        assert(self.upper_type_num > 0 and self.lower_type_num > 0)
        self.pca_dim = pca_dim
        self.tex_pca_dim = tex_pca_dim
        self.tex_width = tex_witdh
        self.tex_height = tex_height
        self.smpl_model_path = smpl_model_path
        self.model_type = model_type
        self.hard_stitch = hard_stitch
        self.midpairs = midpairs
        self.infer_camera = infer_camera
        self.infer_tex = infer_tex
        self.create_detail_meshes = create_detail_meshes
        self.upsample_dismap = upsample_dismap
        self.use_neighbor = use_neighbor
        
        # if create_detail_meshes:
        #     assert(subdivide_template_folder is not None and subdivide_template_name is not None)
        #     self.subdivide_templates = dict()
        #     for g in self.garments:
        #         subdivide_template_name = 'garment_tmp.obj'
        #         subdivide_template_path = os.path.join(subdivide_template_folder, g, subdivide_template_name)
        #         if not os.path.exists(subdivide_template_path):
        #             raise FileNotFoundError
        #         subdivide_times = 2
        #         subdivided_verts, subdivided_faces = subdivide_mesh(moved_cloth1_path, subdivide_times)
        #         pass
        
        self.tot = 0

        self.garPcaparamLayers=nn.ModuleList([GarmentPcaLayer(gtype, 10+self.imgEncoder.gar_latent_size, pca_dim) for gtype in self.garments])
        
        if infer_tex:         
            self.texPcaparamLayers = nn.ModuleList([TexturePcaLayer(gtype, 10+self.imgEncoder.gar_latent_size, tex_pca_dim) for gtype in self.garments])

        # self.detailDecoderLayers = nn.ModuleList([DetailDecoder(gtype, self.n_detail, self.tex_width, self.tex_height) for gtype in self.garments])

        self.detailDecoderLayers = nn.ModuleList([Generator(latent_dim=10+216+self.n_detail, out_channels=1, out_scale=displacement_scale, sample_mode = 'bilinear', gtype=gtype) for gtype in self.garments]).to(device)

        # self.garPcapsLayers=nn.ModuleList([GarmentPcaDecodeLayer(osp.join(osp.dirname(__file__),'../../body_garment_dataset/tmps/%s/pca_data.npz'%gtype)) for gtype in self.garments])
        self.garPcapsLayers=nn.ModuleList(
            [GarmentPcaDecodeLayer(
                osp.join(pca_folder, gtype, 'pca_data_%d.npz' % pca_dim), 
                osp.join(dense_template_folder, 'texture_data_%s_%d.npy' % (gtype, tex_height))) 
             for gtype in self.garments])
        
        if infer_tex:
            self.texPcapsLayers=nn.ModuleList(
                [TexturePcaDecodeLayer(osp.join(pca_folder, gtype, 'pca_data_%d.npz' % pca_dim), 
                                                osp.join(pca_folder, gtype, 'tex_uv.pkl'), osp.join(pca_folder, gtype, 'uv_mask.png'), gtype) 
                for gtype in self.garments])
        
        self.garDisplacementLayers=nn.ModuleList(
            [GarmentDisplacementNet(256, self.imgEncoder.gar_latent_size, gtype, pca_folder, pca_dim)
             for gtype in self.garments])
        
        # self.patchEncoder=MultiPerceptro([3*32*32,1024,524,256])
        self.skinWsNet=SkinWeightNet
        #use pretrained model, fix weights
        for param in self.skinWsNet.parameters():
            param.requires_grad=False
        self.smpl=SMPL(smpl_model_path, obj_saveable=True)
        
        self.skinDeformNet = SkinDeformNet(self.smpl)
        
        # classification Net
        self.gar_classification = with_classification
        if with_classification:
            self.up_classifier = nn.Linear(self.imgEncoder.gar_latent_size, self.upper_type_num)
            self.up_dropout=nn.Dropout(p=0.2)
            self.bottom_classifier = nn.Linear(self.imgEncoder.gar_latent_size, self.lower_type_num)
            self.bottom_dropout = nn.Dropout(p=0.2)
                
        self.check = check
        
        # test
        self.visited = set()
        
        self.subdivider = SubdivideMeshes()
        
        self.dense_template_folder = dense_template_folder
        
        pi = np.pi
        self.constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))], device=self.device).float()

        self.tensorboard_logging = tensorboard_logging
        if tensorboard_logging:
            assert(writer is not None)
            self.writer = writer
        
    # def reinit(self, infer_tex=True):
    #     # self.garPcapsLayers=nn.ModuleList([GarmentPcaDecodeLayer(osp.join(osp.dirname(__file__),'../../body_garment_dataset/tmps/%s/pca_data.npz'%gtype)) for gtype in self.garments])
    #     self.garPcapsLayers=nn.ModuleList(
    #         [GarmentPcaDecodeLayer(
    #             osp.join(self.pca_folder, gtype, 'pca_data_%d.npz' % self.pca_dim), 
    #             osp.join(self.dense_template_folder, gtype, 'texture_data_%s_%d.npy' % (gtype, self.tex_height))) 
    #          for gtype in self.garments])
        
    #     if infer_tex:
    #         self.texPcapsLayers=nn.ModuleList(
    #             [TexturePcaDecodeLayer(osp.join(self.pca_folder, gtype, 'pca_data_%d.npz' % self.pca_dim), 
    #                                             osp.join(self.pca_folder, gtype, 'tex_uv.pkl'), osp.join(self.pca_folder, gtype, 'uv_mask.png')) 
    #             for gtype in self.garments])

    def extract_tris_infos(self,tris_infos):
        edge_index=tris_infos['edge_index']
        garbatch=tris_infos['gar_batch']
        face_index=tris_infos['face_index']
        vf_vindex=tris_infos['vf_vindex']
        vf_findex=tris_infos['vf_findex']
        self.edge_index=edge_index
        self.face_index=face_index
        self.vf_vindex=vf_vindex
        self.vf_findex=vf_findex
        self.garbatch=garbatch
        return garbatch,edge_index,face_index,vf_vindex,vf_findex
    
    def pro_ps(self,ps,cam_k):
        if cam_k.shape[1] == 4:
            ones = torch.ones(ps.shape[0], device=ps.device).reshape(-1, 1)
            ps = torch.cat((ps, ones), dim=1)
        proPs=ps.matmul(cam_k.transpose(0,1))
        depth=proPs[:,2].reshape(-1,1)
        select=(depth>=-1.0e-4)*(depth<=1.0e-4)
        signs=depth.sign()
        signs[(signs>=-0.01)*(signs<=0.01)]=1.0
        depth[select]=signs[select]*1.0e-4
        proPs=torch.cat((proPs[:,0:2]/depth,proPs[:,2].reshape(-1,1)),dim=-1)
        return proPs
    
    # def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
    #     ''' Convert displacement map into detail normal map
    #     '''
    #     batch_size = uv_z.shape[0]
    #     uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
    #     uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

    #     uv_z = uv_z*self.uv_face_eye_mask
    #     uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
    #     dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
    #     uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
    #     uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
    #     uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1.-self.uv_face_eye_mask)
    #     return uv_detail_normals
    
    def forward(self, imgs, img_names=None, gtypes=None, Rs_T=None, cam_k=None, input_imgbatch=None, cam_R_gt=None, cam_T_gt=None, Js_2d_gt=None, pca_verts_gt=None, tex_pca_perg_gt=None, img_masks=None, imgs_perg=None, **kwargs):
        batch_size = imgs.shape[0]
        cam_Rs, cam_Ts = None, None
        if not self.infer_camera:
            shapes,poses,trans,garlatents, lights, _=self.imgEncoder(imgs)
        else:
            shapes,poses,trans,garlatents, cam_exts, lights, _=self.imgEncoder(imgs)
        poses_pre = poses.clone()
        
        # test virtual try-on
        # d = pickle.load(open('test.pkl', 'rb'))
        # shapes = d['shapes']
        # poses = d['poses']
        # trans = d['trans']
        # cam_exts = d['cam_exts']
        # lights = d['lights']
        
        # test motion driven
        # p = torch.as_tensor(pickle.load(open('pose3.pkl', 'rb')), device=poses.device)
        # poses = batch_rodrigues(p.view(-1, 3)).view(-1, 24 * 3 * 3)
    
        # lights_per_gar = torch.cat((lights.view(-1, 3+3), lights.view(-1, 3+3)), 1).view(-1, 3+3)
        lights_per_gar = torch.cat((lights, lights), 1).view(-1, 9, 3)
        
        if self.use_detail:
            detail_latents = self.detailEncoder(imgs)
        # if self.infer_camera:
        #     cam_exts = self.cameraEncoder(imgs)
        if 'garl' in kwargs:
            garlatents=kwargs['garl']
        self.shapes=shapes
        self.poses=poses
        self.trans=trans
        self.garl=garlatents
        if self.infer_camera:
            cam_Rs, cam_Ts = get_batch_RT(cam_exts, device=cam_exts.device)
        self.cam_Rs, self.cam_Ts = cam_Rs, cam_Ts
        batch_num=shapes.shape[0]
        if gtypes is None:
            assert(self.gar_classification)		
        if self.gar_classification:
            tmpfs=F.relu(garlatents)
            up_gar_prob=self.up_classifier(self.up_dropout(tmpfs))
            bottom_gar_prob=self.bottom_classifier(self.bottom_dropout(tmpfs))			
            up_index=up_gar_prob.max(1)[1]
            up_gtypes=up_gar_prob.new_zeros(up_gar_prob.shape).scatter(1,up_index.repeat(2,1).transpose(0,1),up_gar_prob.new_ones(up_gar_prob.shape))
            bottom_index=bottom_gar_prob.max(1)[1]
            bottom_gtypes=bottom_gar_prob.new_zeros(bottom_gar_prob.shape).scatter(1,bottom_index.repeat(2,1).transpose(0,1),bottom_gar_prob.new_ones(bottom_gar_prob.shape))
            if gtypes is not None:
                if type(gtypes) is not torch.Tensor:
                    gtypes=torch.tensor(gtypes,dtype=torch.long,device=imgs.device)
                # nonzero postions [r, c]
                rows,cols=torch.nonzero(gtypes>=0,as_tuple=False).transpose(0,1)
            # tgtypes:
            # line1: subject1: Shirt_prob, T-shirt_prob, Pants_prob, Shorts_prob
            # line2: subject2: Shirt_prob, T-shirt_prob, Pants_prob, Shorts_prob
            # ...
            tgtypes=torch.cat((up_gtypes,bottom_gtypes),dim=-1)
            # modify prob result by gt
            if gtypes is not None:
                for r,c in zip(rows,cols):
                    c=gtypes[r,c].item()
                    
                    if self.model_type == 'D3G':
                        # d3g
                        # upper
                        if c < self.upper_type_num:
                            tgtypes[r,0:4]=0
                            tgtypes[r,c]=1
                        # lower
                        elif c < len(self.garments):
                            tgtypes[r,4:]=0
                            tgtypes[r,c]=1
                    elif self.model_type == 'BCNET':
                        # bcnet
                        if c<2:
                            tgtypes[r,0:2]=0
                            tgtypes[r,c]=1
                        elif c<6:
                            tgtypes[r,2:]=0
                            tgtypes[r,c]=1
                    else:
                        raise NotImplementedError
            gtypes=tgtypes
        self.gtypes=gtypes
        # 同一类组合到一起
        if self.use_detail:
            ordered_datas,ordered_imgbids,ordered_gtypes=order_data_follow_gartypes([shapes,poses_pre,garlatents,detail_latents],batch_num,None,gtypes,self.garmentvnums,self.garments)
            pca_datas=[]
            for ind, (shapes_gtype,poses_gtype,latents_gtype, detail_latents_gtype) in zip(ordered_gtypes,ordered_datas):
                pca_params=self.garPcaparamLayers[ind](torch.cat((shapes_gtype,latents_gtype),dim=-1))
                tex_pca_params = self.texPcaparamLayers[ind](torch.cat((shapes_gtype,latents_gtype),dim=-1))
                pca_ps=self.garPcapsLayers[ind](pca_params).reshape(shapes_gtype.shape[0],self.garmentvnums[ind],3)
                texs = self.texPcapsLayers[ind](tex_pca_params).reshape(shapes_gtype.shape[0],self.tex_height*self.tex_width,3)
                # check Texture PCA Model
                if 1 == 0:
                    for d in range(0, 64, 1):
                        t = tex_pca_params[0].clone()
                        td = t[d].item()
                        s = range(-10, 11, 1)
                        if not os.path.exists('../texture_debug/s109_APose_2_0/btm/dim%d' % d):
                            os.mkdir('../texture_debug/s109_APose_2_0/btm/dim%d' % d)
                        for i in range(21):
                            t[d] = td + s[i] * 10
                            tex = self.texPcapsLayers[5](t).reshape(self.tex_height * self.tex_width, 3).view(self.tex_height,self.tex_width,3)
                            cv2.imwrite('../texture_debug/s109_APose_2_0/btm/dim%d/' % d + 'texture_range(%d)' % (s[i] * 10 + 100) + '.png', tex.detach().cpu().numpy() * 255)
                    # -----------------------
                dis_maps = self.detailDecoderLayers[ind](torch.cat((shapes_gtype,poses_gtype,detail_latents_gtype),dim=-1)).reshape(shapes_gtype.shape[0],self.tex_height*self.tex_width)
                pca_datas.append([pca_params,pca_ps,tex_pca_params,texs, dis_maps])
            [pcas_perg,gps_pca, tex_pcas_perg, textures, displacement_maps],tris_infos=unorder_data_follow_imgbatch(pca_datas,ordered_imgbids,ordered_gtypes,batch_num,self.garPcapsLayers,self.texPcapsLayers,True,True,True,True,True)
        else:
            ordered_datas,ordered_imgbids,ordered_gtypes=order_data_follow_gartypes([shapes,poses,garlatents],batch_num,None,gtypes,self.garmentvnums,self.garments)
            pca_datas=[]
            for ind, (shapes_gtype,_,latents_gtype) in zip(ordered_gtypes,ordered_datas):
                pca_params=self.garPcaparamLayers[ind](torch.cat((shapes_gtype,latents_gtype),dim=-1))
                tex_pca_params = self.texPcaparamLayers[ind](torch.cat((shapes_gtype,latents_gtype),dim=-1))
                pca_ps=self.garPcapsLayers[ind](pca_params).reshape(shapes_gtype.shape[0],self.garmentvnums[ind],3)
                texs = self.texPcapsLayers[ind](tex_pca_params).reshape(shapes_gtype.shape[0],self.tex_height*self.tex_width,3)
                pca_datas.append([pca_params,pca_ps,tex_pca_params,texs])
            [pcas_perg,gps_pca, tex_pcas_perg, textures],tris_infos=unorder_data_follow_imgbatch(pca_datas,ordered_imgbids,ordered_gtypes,batch_num,self.garPcapsLayers,self.texPcapsLayers,True,True,True,True,True)
        textures = textures.view(batch_num * 2, self.tex_height, self.tex_width, 3).permute(0, 3, 1, 2)

 
        # test textures
        # for i in range(len(textures)):
        #     cv2.imwrite('test%d.png' % i, textures[i].detach().cpu().numpy() * 255)
        
        # # repose garments
        # train_item_path = "/workspace/data/Datasets/D3G_and_SIZER/train_items_64.pkl"
        # with open(train_item_path, 'rb') as f:
        #     train_items = pickle.load(f)
        # val_item_path = "/workspace/data/Datasets/D3G_and_SIZER/val_items_64.pkl"
        # with open(val_item_path, 'rb') as f:
        #     val_items = pickle.load(f)
        # test_item_path = "/workspace/data/Datasets/D3G_and_SIZER/test_items_64.pkl"
        # with open(test_item_path, 'rb') as f:
        #     test_items = pickle.load(f)
            
        # items = {**train_items, **val_items, **test_items}
        
        # if len(self.visited) >= len(items):
        #     print('finish!')
        #     exit(0)
        
        # upper_type = 'T-shirt'
        # if gtypes[0][2] == 1 or gtypes[0][3] == 1:
        #     upper_type = 'Shirt'
        
        # lower_type = 'Shorts'
        # if gtypes[0][5] == 1:
        #     lower_type = 'Pants'
        
        # for sid in items.keys():
        #     if sid in self.visited:
        #         continue
        #     data = items[sid]
        #     if upper_type != data['upper_type'] or lower_type != data['lower_type']:
        #         continue
        #     print('%s, %s: %d' % (upper_type, lower_type, len(self.visited)))
        #     upper_t_verts = torch.tensor(data['upper_t'], device='cuda:0')
        #     lower_t_verts = torch.tensor(data['lower_t'], device='cuda:0')
            
        #     t_verts = torch.cat([upper_t_verts, lower_t_verts], dim=0)
            
        #     upper_faces = data['upper_faces']
        #     lower_faces = data['lower_faces']
            
        #     shapes = torch.unsqueeze(torch.tensor(data['shapes'], dtype=torch.float32, device='cuda:0'), dim=0)
        #     poses = data['poses']
        #     poses = torch.tensor(poses, dtype=torch.float32, device='cuda:0')
        #     poses = batch_rodrigues(poses.view(-1, 3)).view(-1, 24 * 3 * 3)
             
        #     garbatch,edge_index,face_index,vf_vindex,vf_findex=self.extract_tris_infos(tris_infos)
        #     imgbatch=imgBatchFromGarBatch(garbatch,gtypes)
        #     if input_imgbatch is not None:
        #         assert((imgbatch-input_imgbatch).sum()==0)
        #     self.imgbatch=imgbatch
        #     Js,body_ns=self.skinDeformNet.skeleton(shapes,True)
            
        #     diss=(t_verts.unsqueeze(1)-Js[imgbatch,:]).norm(dim=-1)
            
        #     if self.skinWsNet.use_normal:
        #         vnorms=compute_vnorms(t_verts,face_index,vf_vindex,vf_findex)
        #         ws=self.skinWsNet(torch.cat((t_verts,vnorms,diss),dim=-1),edge_index,garbatch)
        #     else:
        #         ws=self.skinWsNet(torch.cat((t_verts,diss),dim=-1),edge_index,garbatch)
            
        #     # no rotation
        #     poses[0, 0:9] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1])
            
        #     deform_rec,transforms,pose_Rs,Js_transformed=self.skinDeformNet(t_verts,Js,ws,poses,imgbatch)
            
        #     upper_verts = deform_rec[: len(upper_t_verts)]
        #     lower_verts = deform_rec[len(upper_t_verts): ]

        #     # fix mid line
        #     if data['detail_upper_type'] == 'T-shirt':
        #         local_ind_p1 = [395, 396, 399, 400, 564, 565, 593, 594, 923, 924, 926, 927, 928, 929, 964, 965, 966, 967, 977, 978, 979, 980, 981, 982, 984, 985, 987, 988, 990, 991]
        #         local_ind_p2 = [1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953]
        #         upper_verts[local_ind_p2, :] = upper_verts[local_ind_p1, :]
        #     elif data['detail_upper_type'] == 'Shirt':
        #         local_ind_p1 = [381, 382, 385, 386, 736, 737, 765, 766, 1191, 1192, 1194, 1195, 1196, 1197, 1221, 1222, 1223, 1224, 1233, 1234, 1235, 1236, 1237, 1238, 1240, 1241, 1243, 1244, 1246, 1247]
        #         local_ind_p2 = [2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467]
        #         upper_verts[local_ind_p2, :] = upper_verts[local_ind_p1, :]
            
        #     # save
        #     save_obj(upper_verts, upper_faces, '/workspace/data/Datasets/D3G_and_SIZER/garments/%s/algined_cloth1_repose.obj' % sid)
        #     save_obj(lower_verts, lower_faces, '/workspace/data/Datasets/D3G_and_SIZER/garments/%s/algined_cloth2_repose.obj' % sid)

        #     self.visited.add(sid)

        # check --------------------------
        if self.check:
            save_obj(gps_pca, tris_infos['face_index'], 'check/check_pca.obj')
        # check end ----------------------
        
        garbatch,edge_index,face_index,vf_vindex,vf_findex=self.extract_tris_infos(tris_infos)
        imgbatch=imgBatchFromGarBatch(garbatch,gtypes)
        if input_imgbatch is not None:
            assert((imgbatch-input_imgbatch).sum()==0)
        self.imgbatch=imgbatch
        Js,body_ns=self.skinDeformNet.skeleton(shapes,True)
        
        diss=(gps_pca.unsqueeze(1)-Js[imgbatch,:]).norm(dim=-1)
        
        if self.skinWsNet.use_normal:
            vnorms=compute_vnorms(gps_pca,face_index,vf_vindex,vf_findex)
            ws=self.skinWsNet(torch.cat((gps_pca,vnorms,diss),dim=-1),edge_index,garbatch)
        else:
            ws=self.skinWsNet(torch.cat((gps_pca,diss),dim=-1),edge_index,garbatch)

        if cam_k is None:
            cam_k=torch.Tensor([[3.0375e+03, 0.0000e+00, 2.7000e+02],
                                [0.0000e+00, 3.0375e+03, 2.7000e+02],
                                [0.0000e+00, 0.0000e+00, 1.0000e+00]])
            cam_Rt = torch.tensor([[1, 0, 0, self.tran_mean[0] - self.bcnet_tran_mean[0]], 
                                   [0, 1, 0, self.tran_mean[1] - self.bcnet_tran_mean[1]], 
                                   [0, 0, 1, self.tran_mean[2] - self.bcnet_tran_mean[2]]])
            cam_k = cam_k.matmul(cam_Rt)
            cam_k=cam_k.to(imgs.device)
            
        # deform_rec: deformed pca garment
        # TEST: pca gt
        # gps_pca = pca_verts_gt
        deform_rec,transforms,pose_Rs,Js_transformed=self.skinDeformNet(gps_pca,Js,ws,poses,imgbatch)
        
        # check -------------------------------
        if self.check:
            save_obj(deform_rec, tris_infos['face_index'], 'check/check_deform_pca.obj')
        # check end ---------------------------
        
        
        self.transforms=transforms
        # deform_norms=compute_vnorms(deform_rec,face_index,vf_vindex,vf_findex)
        if 'pro_fs' in kwargs:
            pro_features=kwargs['pro_fs']
        else:
            # rect: rec + t
            deform_rect=deform_rec+trans[imgbatch,:]
            
            # check -----------------------------------
            if self.check:
                save_obj(deform_rect, tris_infos['face_index'], 'check/check_deform_trans_pca.obj')
            # check end -------------------------------
            
            # deform_rect = deform_rec
            deform_rect_pros=self.pro_ps(deform_rect,cam_k)[:,:2]
            # pro_patches=get_patchs_from_imgs(deform_rect_pros,imgs,imgbatch)
            # pro_features=self.patchEncoder(pro_patches.reshape(-1,3*32*32))
        # self.pro_fs=pro_features
        # ordered_datas2,_,ordered_gtypes2=order_data_follow_gartypes([deform_rec,deform_norms,pro_features,transforms[:,:3,:3].reshape(-1,9),transforms[:,:3,3]],batch_num,garbatch,gtypes,self.garmentvnums,self.garments)
        # assert(ordered_gtypes2==ordered_gtypes)

        # displacement_datas=[]
        # for ind,(pca_params,pca_ps),(shapes_gtype,_,latents_gtype),(deform_ps,deform_ns,pro_fs,Rs_gtype,Ts_gtype) in zip(ordered_gtypes2,pca_datas,ordered_datas,ordered_datas2):
        #     garvnum=self.garmentvnums[ind]
        #     size=pca_ps.shape[0]
        #     displacement_gtype=self.garDisplacementLayers[ind](torch.cat((pca_ps,deform_ps,deform_ns,Rs_gtype,Ts_gtype,shapes_gtype[:,None,:].expand(size,garvnum,10),pca_params[:,None,:].expand(size,garvnum,self.pca_dim),pro_fs,latents_gtype[:,None,:].expand(size,garvnum,-1)),dim=-1))
        #     displacement_datas.append([displacement_gtype])
        # [displacements],_=unorder_data_follow_imgbatch(displacement_datas,ordered_imgbids,ordered_gtypes2,batch_num)
        
        displacements = torch.zeros_like(gps_pca)
        
        # check -----------------------------------
        # if self.check:
        #     params = self.garDisplacementLayers[0].named_parameters()
        #     for k, v in params:
        #         print(k, v)
        # check end -------------------------------
        
        gps_diss=gps_pca+displacements
        
        # check ------------------------------------
        if self.check:
            save_obj(gps_diss, tris_infos['face_index'], 'check/check_dis_pca.obj')
        # check end --------------------------------

        tmps=torch.cat((gps_diss,gps_diss.new_ones(gps_diss.shape[0],1)),dim=-1).unsqueeze(-1)
        gps_rec=torch.matmul(transforms,tmps).squeeze(-1)[:,:3]		
        gps_rec=gps_rec+trans[imgbatch,:]
        
        # hard stitch mid line
        if self.hard_stitch:
            for i in range(len(gtypes)):
                types = gtypes[i]
                if types[0] == 1 or types[2] == 1:
                        # tshirt or shirt, hard stitch
                        select = garbatch == i * 2
                        garverts = gps_rec[select]
                        gar_type = 'T-shirt' if types[0] == 1 else 'Shirt'
                        for pair in self.midpairs[gar_type]:
                            mid_pos = (garverts[pair[0]] + garverts[pair[1]]) / 2
                            garverts[pair[0]] = mid_pos
                            garverts[pair[1]] = mid_pos
                            gps_rec[select] = garverts
        
        # check ------------------------------------
        if self.check:
            save_obj(gps_rec, tris_infos['face_index'], 'check/check_result.obj')
        # check end --------------------------------
        
        Js_transformed=Js_transformed+trans.unsqueeze(1)
        body_faces = torch.tensor(self.smpl.faces)
        body_tpose_ps, _, _, _ = self.smpl(shapes, torch.zeros_like(pose_Rs), True, False)
        body_ps,_,_,_=self.smpl(shapes,pose_Rs,True,False)
        
        # check ----------------------------------------
        if self.check:
            save_obj(body_ps[0], body_faces, 'check/check_shape_pose_body.obj')
        # check end ------------------------------------
        
        body_ps=body_ps+trans.unsqueeze(1)
        
        # check -----------------------------------------
        if self.check:
            save_obj(body_ps[0], body_faces, 'check/check_shape_pose_trans_body.obj')
        # check end -------------------------------------
        
        
        # project joints ----------------------------------------------------------------------------
        renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs, T=cam_Ts, lights=None, device=imgs.device)
        Js_2d = renderer.transform_points(Js_transformed)
        Js_2d = Js_2d[:,:,:2]
        # -------------------------------------------------------------------------------------------
     
        # 上衣和裤子各对应一个RT及相机
        cam_Rs_per_gar = torch.cat((cam_Rs.view(-1, 1, 3*3), cam_Rs.view(-1, 1, 3*3)), 1).view(-1, 3, 3)
        cam_Ts_per_gar = torch.cat((cam_Ts.view(-1, 1, 3), cam_Ts.view(-1, 1, 3)), 1).view(-1, 3)
        renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs_per_gar, T=cam_Ts_per_gar, lights=None, device=imgs.device)
        uv_rasterizer = Pytorch3dRasterizer(self.tex_height)
        dense_triangles = []
        uv_masks = torch.zeros(0, 1, self.tex_height, self.tex_width, device=self.device)
        for g in gtypes:
            inds = torch.nonzero(g)
            dense_triangles.append(self.texPcapsLayers[inds[0][0]].dense_triangles)
            dense_triangles.append(self.texPcapsLayers[inds[1][0]].dense_triangles)
            uv_masks = torch.cat((uv_masks, self.texPcapsLayers[inds[0][0]].uv_mask_erosion.unsqueeze(0).unsqueeze(0)), dim=0)
            uv_masks = torch.cat((uv_masks, self.texPcapsLayers[inds[1][0]].uv_mask_erosion.unsqueeze(0).unsqueeze(0)), dim=0)
        max_tri_len = max([len(tri) for tri in dense_triangles])
        dense_faces = torch.zeros(0, max_tri_len, 3, device=self.device)
        for i in range(len(dense_triangles)):
            tri = dense_triangles[i]
            dense_faces = torch.cat((dense_faces, pad(tri, [0, 0, 0, max_tri_len - len(tri)], mode='constant', value=-1).unsqueeze(0)), dim=0)

        if img_names is not None:
            img_name = img_names[0]
            sp = img_name.split('_')
            sid = sp[0] + '_' + sp[1]
            if img_name.startswith('s'):
                sid += '_' + sp[2]
        
        # # TEST: gt texture
        # tex_gt_path = '/workspace/data/Datasets/D3G_and_SIZER/garments/%s/aligned_cloth1_colored_256_filtered.png' % sid
        # tex_gt = torch.tensor(cv2.imread(tex_gt_path) / 255., device=textures.device, dtype=textures.dtype)
        # textures[0] = tex_gt
        
        # render coarse meshes -------------------------------------------------------------------
        coarse_meshes, texUVs = create_meshes(deform_rect, face_index, textures.permute(0, 2, 3, 1), garbatch, torch.where(gtypes == 1)[1], self.garPcapsLayers, self.texPcapsLayers)
        coarse_verts = coarse_meshes.verts_padded()
        coarse_faces = coarse_meshes.faces_padded()
        uvcoords = texUVs.verts_uvs_padded()
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1   # [-1, 1]
        uvcoords[...,1] = -uvcoords[...,1]
        uvfaces = texUVs.faces_uvs_padded()
        
        # # test
        # a = load_obj('bbb.obj')
        # b = a[1]
        # c = a[3]
        # texUVs._verts_uvs_padded = texUVs._verts_uvs_padded[:, :32496, :]
        # texUVs._verts_uvs_padded[0] = b
        # uvcoords = texUVs._verts_uvs_padded
        # uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        # uvcoords = uvcoords*2 - 1   # [-1, 1]
        # uvcoords[...,1] = -uvcoords[...,1]
        # texUVs._faces_uvs_padded[0] = c
        # uvfaces = texUVs._faces_uvs_padded
        
        # # TODO: TEST: FIXED DISPLACEMENT
        # a = load_objs_as_meshes(['garment_tmp_uv.obj'])
        # coarse_verts[0] = a.verts_packed()
        # uv_coarse_vertices = world2uv(coarse_verts, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()
        # coarse_normals = coarse_meshes.verts_normals_padded()
        # uv_coarse_normals = world2uv(coarse_normals, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()
        # b = load_objs_as_meshes(['untitled1.obj'])
        # detail_verts = b.verts_padded().repeat(2, 1, 1)
        # detail_faces = b.faces_padded().repeat(2, 1, 1)
        # dense_uv = b.textures.verts_uvs_padded()
        # dense_uv = torch.cat([dense_uv, dense_uv[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        # dense_uv = dense_uv*2 - 1   # [-1, 1]
        # dense_uv[...,1] = -dense_uv[...,1]
        # dense_uvfaces = b.textures.faces_uvs_padded()
        # uv_detail_vertices = world2uv(detail_verts, detail_faces, dense_uv, dense_uvfaces, uv_rasterizer).detach()

        # C = uv_coarse_vertices.cpu()
        # D = uv_detail_vertices.cpu()
        # N = uv_coarse_normals.cpu()
        
        # H = D - C
        
        # A = (H / (N + 1e-7)).to(self.device)
        
        # for x in range(H.shape[-2]):
        #     for y in range(H.shape[-1]):
        #         h = H[0, :, x, y]
        #         n = N[0, :, x, y]
                
        
        
        
        uv_coarse_vertices = world2uv(coarse_verts, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()
        uv_coarse_vertices_masks = get_imgs_masks(uv_coarse_vertices)
    
        # # test
        # z = uv_masks[0].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()
        # vt = (uvcoords[0] + 1) / 2
        # for i in range(len(vt)):
        #     x = int(vt[i, 0] * 255)
        #     y = int(vt[i, 1] * 255)
        #     z[y, x, :] = [1, 0, 0]
        # cv2.imwrite('test.png', z * 255)
    
        # # test
        # save_path = 'data/texture_data_Shirt_256.npy'
        # make_dense_template(uvcoords[0], uvfaces[0], coarse_faces[0], uv_coarse_vertices_masks[0,0], self.tex_height, save_path)
        
        # # test
        # batch_size = 1
        # obj_filename = "/workspace/data/Datasets/D3G_and_SIZER/tmps/Shirt/garment_tmp_subdivide_uv.obj"
        # from pytorch3d.io import load_obj
        # verts, faces, aux = load_obj(obj_filename)
        # coarse_verts = verts.expand(batch_size, -1, -1)
        # coarse_faces = faces.verts_idx[None,...].expand(batch_size, -1, -1)
        # uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        # uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        # uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        # uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        # uvcoords = uvcoords.expand(batch_size, -1, -1)
        # uv_coarse_vertices = world2uv(coarse_verts, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()

        # # test
        # obj_filename = '/workspace/BCNet/data/head_template.obj'
        # from pytorch3d.io import load_obj
        # verts, faces, aux = load_obj(obj_filename)
        # uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        # uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        # faces = faces.verts_idx[None,...]
        # uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        # uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        # batch_size = 1
        # face_vertices = get_face_vertices(verts.expand(batch_size, -1, -1), faces.expand(batch_size, -1, -1))
        # uv_vertices = uv_rasterizer(uvcoords.expand(batch_size, -1, -1), uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        
        
        # # TEST
        # vv, vt, ff, uvf = load_obj('/workspace/data/Datasets/D3G_and_SIZER/tmps/Shirt/garment_tmp_subdivide_uv.obj')
        # vv, ff, vt, uvf = vv.unsqueeze(0), ff.unsqueeze(0), vt.unsqueeze(0), uvf.unsqueeze(0)
        # vt = torch.cat([vt, vt[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        # vt = vt*2 - 1   # [-1, 1]
        # vt[...,1] = -vt[...,1]
        # uv_coarse_vertices = world2uv(vv.repeat(2,1,1), ff.repeat(2,1,1), vt.repeat(2,1,1), uvf.repeat(2,1,1), uv_rasterizer).detach()
        # uv_coarse_vertices = uv_coarse_vertices.to(self.device)
        
        # TEST
        uv_coarse_vertices_masks = get_imgs_masks(uv_coarse_vertices)
        
        # # TODO: FILL BY EDGES / FACES
        # for i in range(self.tex_height):
        #     for j in range(self.tex_width):
        #         if uv_coarse_vertices_masks[0, 0, i, j] == 1:
        #             continue
        #         nearest_pixel = find_nearest_pixel(uv_coarse_vertices_masks[0,0], (i, j))
        #         uv_coarse_vertices[0, :, i, j] = uv_coarse_vertices[0, :, nearest_pixel[0], nearest_pixel[1]]
        # # nearest_pixels = find_batch_nearest_pixels(uv_coarse_vertices_masks)
        # uv_coarse_vertices *= uv_masks
        # # cv2.imwrite('test.png', uv_coarse_vertices[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        
        # dense_coarse_vertices = uv_coarse_vertices.permute(0,2,3,1).reshape([batch_size * 2, -1, 3])
        # uv_coarse_normals = get_vertex_normals(dense_coarse_vertices, dense_faces)
        # uv_coarse_normals = uv_coarse_normals.reshape([batch_size * 2, self.tex_height, self.tex_width, 3]).permute(0,3,1,2)
        
        coarse_normals = coarse_meshes.verts_normals_padded()
        uv_coarse_normals = world2uv(coarse_normals, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()
        uv_coarse_normals_masks = get_imgs_masks(uv_coarse_normals).detach()
        
        # test
        # cam_R_gt_per_gar = torch.cat((cam_R_gt.view(-1, 1, 3*3), cam_R_gt.view(-1, 1, 3*3)), 1).view(-1, 3, 3)
        
        coarse_render_masks = None
        # coarse_render_masks = get_imgs_masks(renderer(coarse_meshes).permute(0, 3, 1, 2))
        # coarse_render_masks[coarse_render_masks>0] = 1
        
        # coarse_textures = textures * coarse_shading_images * uv_masks.permute(0, 2, 3, 1) + textures * (1 - uv_masks.permute(0, 2, 3, 1))
        
        if True or self.inferring or not self.use_detail:
            # # point light
            # coarse_shading = add_pointlight(uv_coarse_vertices.permute(0,2,3,1).reshape([batch_size*2, -1, 3]), uv_coarse_normals.permute(0,2,3,1).reshape([batch_size*2, -1, 3]), lights=lights_per_gar)
            # coarse_shading_images = coarse_shading.reshape([batch_size*2, self.tex_height, self.tex_width, 3]).permute(0, 3, 1, 2)
            
            # sh light
            coarse_shading = add_SHlight(uv_coarse_normals, lights_per_gar, self.constant_factor)
            coarse_shading_images = coarse_shading
            
            coarse_shading_images_masks = get_imgs_masks(coarse_shading_images)
            # coarse_shading_images_masks = erode(coarse_shading_images_masks)
            # coarse_shading_images_dilated = dilate(coarse_shading_images)
            # coarse_shading_images = (coarse_shading_images * coarse_shading_images_masks) + (coarse_shading_images_dilated * (1 - coarse_shading_images_masks))
            coarse_textures = textures * coarse_shading_images
            texUVs._maps_padded = coarse_textures.permute(0, 2, 3, 1)
        # coarse_meshes._verts_normals_packed *= -1
        rendered_coarse_imgs = renderer(coarse_meshes)


        
        # # vis
        # cv2.imwrite('test_rendered_coarse.png', rendered_coarse_imgs[1].detach().cpu().numpy() * 255)
        # cv2.imwrite('test_coarse_textures.png', coarse_textures[1].detach().cpu().numpy() * 255)
        # cv2.imwrite('test_coarse_vertices.png', uv_coarse_vertices[1].detach().cpu().numpy().transpose(1, 2, 0) * 255)
        # cv2.imwrite('test_coarse_shading_images.png', coarse_shading_images[1].detach().cpu().numpy() * 255)
        # # ---------------------------------------------------------------------------------------


        # displacement map -> normal map -----------------------------------------------------------
        if self.use_detail:
            # TODO: DO WE NEED MASK?
            displacement_maps = displacement_maps.reshape(-1, 1, self.tex_height, self.tex_width) * uv_masks
            # TODO: FIXED DIS
            # displacement_maps *= 0
            uv_detail_vertices = uv_coarse_vertices + displacement_maps * uv_coarse_normals # + A * uv_coarse_normals
            dense_detail_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size * 2, -1, 3])
            uv_detail_normals = get_vertex_normals(dense_detail_vertices, dense_faces)
            uv_detail_normals = uv_detail_normals.reshape([batch_size * 2, self.tex_height, self.tex_width, 3]).permute(0,3,1,2).contiguous()
            uv_detail_normals = uv_detail_normals * uv_coarse_normals_masks + uv_coarse_normals*(1. - uv_coarse_normals_masks)
        # ---------------------------------------------------------------------------------------------------------------


        # render detail meshes -------------------------------------------------------------
        if self.use_detail:
            # point light
            # detail_shading = add_pointlight(dense_detail_vertices, uv_detail_normals.permute(0,2,3,1).reshape([batch_size*2, -1, 3]), lights=lights_per_gar)
            # detail_shading_images = detail_shading.reshape([batch_size*2, self.tex_height, self.tex_width, 3]).permute(0, 3, 1, 2)
            
            # sh
            detail_shading = add_SHlight(uv_detail_normals, lights_per_gar, self.constant_factor)
            detail_shading_images = detail_shading
            
            detail_shading_images_masks = get_imgs_masks(detail_shading_images)
            # detail_shading_images_masks = erode(detail_shading_images_masks)
            # detail_shading_images_dilated = dilate(detail_shading_images)
            # detail_shading_images = (detail_shading_images * detail_shading_images_masks) + (detail_shading_images_dilated * (1 - detail_shading_images_masks))
            # detail_textures = textures * shading_images * masks.permute(0, 2, 3, 1) + textures * (1 - masks.permute(0, 2, 3, 1))
            detail_textures = textures * detail_shading_images
            # detail_meshes, _ = create_meshes(deform_rect, face_index, detail_textures, garbatch, torch.where(gtypes == 1)[1], self.garPcapsLayers, self.texPcapsLayers)
            
            # detail without texture
            texUVs._maps_padded = (torch.ones_like(textures) * 0.9 * detail_shading_images).permute(0, 2, 3, 1)
            rendered_detail_imgs_no_tex = renderer(coarse_meshes)
            
            # detail with texture
            texUVs._maps_padded = detail_textures.permute(0, 2, 3, 1)
            rendered_detail_imgs = renderer(coarse_meshes)
            detail_verts = coarse_meshes.verts_packed()
            dis_packed = displacement_maps
            
            
            # TODO: ID-MRF
            #--- extract texture  
            trans_uv_vertices = renderer.transform_points(uv_coarse_vertices.permute(0, 2, 3, 1).reshape(-1, 256*256, 3), target='screen_space')
            trans_uv_vertices = trans_uv_vertices.reshape(-1, 256, 256, 3)           
            trans_uv_vertices = trans_uv_vertices / imgs.shape[-1] * 2 - 1
            uv_gt = F.grid_sample(torch.cat([imgs_perg, img_masks], dim=1), trans_uv_vertices[:,:,:,:2], mode='bilinear', align_corners=False)
            extracted_textures = uv_gt[:,:3,:,:].detach()
            extracted_uv_masks = uv_gt[:,3:4,:,:].detach()
            # cv2.imwrite('test_tex_gt.jpg', (uv_texture_gt.permute(0, 2, 3, 1) * coarse_shading_images)[0].detach().cpu().numpy() * 255)   
            # cv2.imwrite('test_tex.jpg', (textures * coarse_shading_images)[0].detach().cpu().numpy() * 255 )
            # cv2.imwrite('test_img.jpg', imgs[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
            
            # camera space
            coarse_verts_cam_space = renderer.transform_points(coarse_verts, target='camera_space')
            coarse_normals_cam_space = get_vertex_normals(coarse_verts_cam_space, coarse_faces.clamp(min=0))
            uv_coarse_normals_cam_space = world2uv(coarse_normals_cam_space, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()  
            self_occlusion_masks = (uv_coarse_normals_cam_space[:,[-1],:,:] < -0.05).float().detach()
            uv_vis_masks = (uv_masks * extracted_uv_masks * self_occlusion_masks).detach()
            # extracted_textures *= uv_vis_masks
            
            # # test
            # local_mask = load_local_mask()
            # self.face_attr_mask = local_mask
            # uv_texture = detail_textures
            # pi = 0
            # new_size = 256
            # uv_texture_patch = F.interpolate(uv_texture[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            
        else:
            rendered_detail_imgs = None
            detail_verts = None
            dis_packed = None
            detail_textures = None
            extracted_textures = None
            uv_vis_masks = None
        
        # # # vis
        if False and self.inferring and self.tot % 1 == 0:
            # cv2.imwrite('test_coarse_vertices.png', uv_coarse_vertices[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
            # cv2.imwrite('test_coarse_normals.png', uv_coarse_normals[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
            # cv2.imwrite('test_detail_vertices.png', uv_detail_vertices[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
            # cv2.imwrite('test_detail_normals.png', uv_detail_normals[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
            # cv2.imwrite('test_coarse_shading_images.png', coarse_shading_images[0].detach().cpu().numpy() * 255)
            # cv2.imwrite('test_coarse_tex.png', textures[0].detach().cpu().numpy() * 255)
            # cv2.imwrite('test_detail_tex.png', detail_textures[0].detach().cpu().numpy() * 255)
            # cv2.imwrite('test_uv_mask.png', uv_masks[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
            i = 0
            rendered_coarse_upper = rendered_coarse_imgs[i].detach().clone()
            rendered_coarse_lower = rendered_coarse_imgs[i + 1].detach().clone()
            rendered_coarse_all = rendered_coarse_lower
            rendered_coarse_all[rendered_coarse_upper > 0] = rendered_coarse_upper[rendered_coarse_upper > 0]
            rendered_coarse_with_bg = imgs[i].clone()
            rendered_coarse_with_bg[rendered_coarse_all > 0] = rendered_coarse_all[rendered_coarse_all > 0]
            
            if self.use_detail:
                rendered_detail_upper = rendered_detail_imgs[i].detach().clone()
                rendered_detail_lower = rendered_detail_imgs[i + 1].detach().clone()
                rendered_detail_all = rendered_detail_lower
                rendered_detail_all[rendered_detail_upper > 0] = rendered_detail_upper[rendered_detail_upper > 0]
                rendered_detail_with_bg = imgs[i].clone()
                rendered_detail_with_bg[rendered_detail_all > 0] = rendered_detail_all[rendered_detail_all > 0]
            
            for k in Js_2d[i]:
                x = int(k[0])
                y = int(k[1])
                d = [-3, -2, -1, 0, 1, 2, 3]
                for dx in d:
                    if x + dx < 0 or x + dx >= 540:
                        continue
                    for dy in d:
                        if y + dy < 0 or y + dy >= 540:
                            continue
                        rendered_coarse_all[0][y + dy][x + dx] = 0
                        rendered_coarse_all[1][y + dy][x + dx] = 0
                        rendered_coarse_all[2][y + dy][x + dx] = 1
                        if self.use_detail:
                            rendered_detail_all[0][y + dy][x + dx] = 0
                            rendered_detail_all[1][y + dy][x + dx] = 0
                            rendered_detail_all[2][y + dy][x + dx] = 1
            for k in Js_2d_gt.view(-1, 24, 2)[i]:
                x = int(k[0])
                y = int(k[1])
                d = [-3, -2, -1, 0, 1, 2, 3]
                for dx in d:
                    for dy in d:
                        if x + dx < 0 or x + dx >= 540:
                            continue
                        if y + dy < 0 or y + dy >= 540:
                            continue
                        rendered_coarse_all[0][y + dy][x + dx] = 0
                        rendered_coarse_all[1][y + dy][x + dx] = 1
                        rendered_coarse_all[2][y + dy][x + dx] = 0
                        if self.use_detail:
                            rendered_detail_all[0][y + dy][x + dx] = 0
                            rendered_detail_all[1][y + dy][x + dx] = 1
                            rendered_detail_all[2][y + dy][x + dx] = 0

        
            cv2.imwrite('test_input.jpg', imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255)
            cv2.imwrite('test_coarse_all.jpg', rendered_coarse_all.permute(1, 2, 0).detach().cpu().numpy() * 255)
            cv2.imwrite('test_coarse_all_with_bg.jpg', rendered_coarse_with_bg.permute(1, 2, 0).detach().cpu().numpy() * 255)
            if self.use_detail:
                cv2.imwrite('test_detail_all.jpg', rendered_detail_all.permute(1, 2, 0).detach().cpu().numpy() * 255)
                cv2.imwrite('test_detail_all_with_bg.jpg', rendered_detail_with_bg.permute(1, 2, 0).detach().cpu().numpy() * 255)
                dis_vis = displacement_maps[i][0].detach()
                dis_vis = (dis_vis - dis_vis.min()) / (dis_vis.max() - dis_vis.min() + 1e-7)
                cv2.imwrite('test_dis_map.jpg', dis_vis.cpu().numpy() * 255)
                cv2.imwrite('test_detail_texture.jpg', (detail_textures * uv_vis_masks)[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
                cv2.imwrite('test_extracted_texture.jpg', (extracted_textures * uv_vis_masks)[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        # -----------------------------------------------------------------------------------------
    
        # infer vis ---------------------------------------------------------------------------------------
        if self.inferring or self.tot % 1000 == 0:
        # if self.inferring and self.vis_save_folder is not None:
            coarse_meshes_no_tex = coarse_meshes.clone()
            dummy_textures = torch.ones_like(textures) * coarse_shading_images
            coarse_meshes_no_tex.textures._maps_padded = dummy_textures.permute(0, 2, 3, 1)
            rendered_coarse_imgs_no_tex = renderer(coarse_meshes_no_tex)
            types = torch.where(gtypes == 1)[1]
            vis_imgs = []
            for i in range(0, len(imgs_perg), 2):
                vis_imgs.append([])
                rendered_coarse_upper_no_tex = rendered_coarse_imgs_no_tex[i].detach().clone()
                rendered_coarse_lower_no_tex = rendered_coarse_imgs_no_tex[i + 1].detach().clone()
                rendered_coarse_all_no_tex = rendered_coarse_upper_no_tex
                rendered_coarse_all_no_tex[rendered_coarse_lower_no_tex > 0] = rendered_coarse_lower_no_tex[rendered_coarse_lower_no_tex > 0]
                rendered_coarse_all_no_tex_with_bg = imgs_perg[i].clone()
                rendered_coarse_all_no_tex_with_bg[rendered_coarse_all_no_tex > 0] = rendered_coarse_all_no_tex[rendered_coarse_all_no_tex > 0]
                
                rendered_coarse_upper = rendered_coarse_imgs[i].detach().clone()
                rendered_coarse_lower = rendered_coarse_imgs[i + 1].detach().clone()
                rendered_coarse_all = rendered_coarse_upper
                rendered_coarse_all[rendered_coarse_lower > 0] = rendered_coarse_lower[rendered_coarse_lower > 0]
                rendered_coarse_with_bg = imgs_perg[i].clone()
                rendered_coarse_with_bg[rendered_coarse_all > 0] = rendered_coarse_all[rendered_coarse_all > 0]
                
                if self.use_detail:
                    dis_vis_upper = displacement_maps[i].repeat(3, 1, 1).detach()
                    dis_vis_lower = displacement_maps[i + 1].repeat(3, 1, 1).detach()
                    dis_vis_upper = (dis_vis_upper - dis_vis_upper.min()) / (dis_vis_upper.max() - dis_vis_upper.min() + 1e-7)
                    dis_vis_lower = (dis_vis_lower - dis_vis_lower.min()) / (dis_vis_lower.max() - dis_vis_lower.min() + 1e-7)
                
                
                # upper_vnum = self.garPcapsLayers[types[i]].vnum
                # lower_vnum = self.garPcapsLayers[types[i + 1]].vnum
                # dense_verts_upper, dense_colors_upper, dense_faces_upper = upsample_mesh(coarse_verts[i,:upper_vnum].detach().cpu().numpy(), 
                #                                                                          coarse_normals[i].detach().cpu().numpy(), 
                #                                                                          coarse_faces[i].detach().cpu().numpy(), 
                #                                                                          displacement_maps[i, 0].detach().cpu().numpy(), 
                #                                                                          textures[0].permute(1, 2, 0).detach().cpu().numpy(), 
                #                                                                          self.garPcapsLayers[types[i]].dense_template)

                # dense_verts_lower, dense_colors_lower, dense_faces_lower = upsample_mesh(coarse_verts[i + 1,:lower_vnum].detach().cpu().numpy(), 
                #                                                                          coarse_normals[i + 1].detach().cpu().numpy(), 
                #                                                                          coarse_faces[i + 1].detach().cpu().numpy(), 
                #                                                                          displacement_maps[i + 1, 0].detach().cpu().numpy(), 
                #                                                                          textures[i + 1].permute(1, 2, 0).detach().cpu().numpy(), 
                #                                                                          self.garPcapsLayers[types[i + 1]].dense_template)
                
                # from utils import save_obj
                # save_obj(dense_verts_upper, dense_faces_upper, os.path.join(self.mesh_save_folder, img_name.replace('.jpg', '_up_dense.obj')))
                # save_obj(dense_verts_lower, dense_faces_lower, os.path.join(self.mesh_save_folder, img_name.replace('.jpg', '_bottom_dense.obj')))
                
                if self.use_detail:
                    rendered_detail_upper = rendered_detail_imgs[i].detach().clone()
                    rendered_detail_lower = rendered_detail_imgs[i + 1].detach().clone()
                    rendered_detail_all = rendered_detail_upper
                    rendered_detail_all[rendered_detail_lower > 0] = rendered_detail_lower[rendered_detail_lower > 0]
                    rendered_detail_with_bg = imgs_perg[i].clone()
                    rendered_detail_with_bg[rendered_detail_all > 0] = rendered_detail_all[rendered_detail_all > 0]

                    rendered_detail_upper_no_tex = rendered_detail_imgs_no_tex[i].detach().clone()
                    rendered_detail_lower_no_tex = rendered_detail_imgs_no_tex[i + 1].detach().clone()
                    rendered_detail_all_no_tex = rendered_detail_upper_no_tex
                    rendered_detail_all_no_tex[rendered_detail_lower_no_tex > 0] = rendered_detail_lower_no_tex[rendered_detail_lower_no_tex > 0]
                    rendered_detail_with_bg_no_tex = imgs_perg[i].clone()
                    rendered_detail_with_bg_no_tex[rendered_detail_all_no_tex > 0] = rendered_detail_all_no_tex[rendered_detail_all_no_tex > 0]

                    
                
                # each row of vis
                # ----------------------------------------------------------------------------------------------------------------------------------
                # | input | coarse | coarse(bg) | coarse (tex) | coarse(tex,bg) | Ds | detail | detail(bg) | detail(tex) | detail(tex, bg) | input |
                # ----------------------------------------------------------------------------------------------------------------------------------
                
                # input
                vis_imgs[-1].append(imgs_perg[i].permute(1, 2, 0).detach().cpu().numpy())
                
                # coarse
                vis_imgs[-1].append(rendered_coarse_all_no_tex.permute(1, 2, 0).detach().cpu().numpy())
                vis_imgs[-1].append(rendered_coarse_all_no_tex_with_bg.permute(1, 2, 0).detach().cpu().numpy())
                vis_imgs[-1].append(rendered_coarse_all.permute(1, 2, 0).detach().cpu().numpy())
                vis_imgs[-1].append(rendered_coarse_with_bg.permute(1, 2, 0).detach().cpu().numpy())
                
                # texture
                vis_imgs[-1].append(cv2.resize(textures[i].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                vis_imgs[-1].append(cv2.resize(textures[i + 1].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                
                # D
                if self.use_detail:
                    vis_imgs[-1].append(cv2.resize(dis_vis_upper.permute(1, 2, 0).cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                    vis_imgs[-1].append(cv2.resize(dis_vis_lower.permute(1, 2, 0).cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                
                # detail
                if self.use_detail:
                    vis_imgs[-1].append(rendered_detail_all_no_tex.permute(1, 2, 0).detach().cpu().numpy())
                    vis_imgs[-1].append(rendered_detail_with_bg_no_tex.permute(1, 2, 0).detach().cpu().numpy())
                    vis_imgs[-1].append(rendered_detail_all.permute(1, 2, 0).detach().cpu().numpy())
                    vis_imgs[-1].append(rendered_detail_with_bg.permute(1, 2, 0).detach().cpu().numpy())
                # # input
                # vis_imgs[-1].append(imgs_perg[i].permute(1, 2, 0).detach().cpu().numpy())
                # # coarse_shading
                vis_imgs[-1].append(cv2.resize(coarse_shading[i].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                vis_imgs[-1].append(cv2.resize(coarse_shading[i + 1].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                # vised_textures
                if self.use_detail:
                    vis_imgs[-1].append(cv2.resize((detail_textures * uv_vis_masks)[i].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                    vis_imgs[-1].append(cv2.resize((extracted_textures * uv_vis_masks)[i].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                    vis_imgs[-1].append(cv2.resize((detail_textures * uv_vis_masks)[i + 1].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))
                    vis_imgs[-1].append(cv2.resize((extracted_textures * uv_vis_masks)[i + 1].permute(1, 2, 0).detach().cpu().numpy(), (imgs.shape[-2], imgs.shape[-1])))                  
                    
                # TODO: VIS DENSE DETAIL
                
                
                # cv2.imwrite(os.path.join(self.vis_save_folder, img_name.replace('.jpg', '_input.jpg')), imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255)
                # cv2.imwrite(os.path.join(self.vis_save_folder, img_name.replace('.jpg', '_coarse_all.jpg')), rendered_coarse_all.permute(1, 2, 0).detach().cpu().numpy() * 255)
                # cv2.imwrite(os.path.join(self.vis_save_folder, img_name.replace('.jpg', '_coarse_all_with_bg.jpg')), rendered_coarse_with_bg.permute(1, 2, 0).detach().cpu().numpy() * 255)
                # cv2.imwrite(os.path.join(self.vis_save_folder, img_name.replace('.jpg', '_detail_all.jpg')), rendered_detail_all.permute(1, 2, 0).detach().cpu().numpy() * 255)
                # cv2.imwrite(os.path.join(self.vis_save_folder, img_name.replace('.jpg', '_detail_all_with_bg.jpg')), rendered_detail_with_bg.permute(1, 2, 0).detach().cpu().numpy() * 255)
                # cv2.imwrite(os.path.join(self.vis_save_folder, img_name.replace('.jpg', '_dis.jpg')), dis_vis.cpu().numpy() * 255)
            vis_imgs = np.array(vis_imgs)
            vis_imgs = vis_imgs[:,:,:,:,::-1]
            # fig = vis(vis_imgs, os.path.join(self.vis_save_folder, '%d.png' % self.tot), img_names)
            if self.vis_save_folder is not None:
                # temp_vis_folder = self.vis_save_folder.replace('vis', 'vis1')
                fig = vis(vis_imgs, os.path.join(self.vis_save_folder, '%s.png' % img_name.replace('.jpg', '')))
            else:
                fig = vis(vis_imgs, 'test.png')
            if self.tensorboard_logging:
                self.writer.add_figure('vis/overview', fig, self.tot)
        # ------------------------------------------------------------------------------------------------
        
        # # test
        # cv2.imwrite('test1.png', coarse_textures[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        # cv2.imwrite('test2.png', rendered_coarse_imgs[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        # cv2.imwrite('test3.png', coarse_shading_images[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        
        # cv2.imwrite('test4.png', (displacement_maps[0].permute(1, 2, 0).detach().cpu().numpy() * 100 + 1) / 2 * 255)
        # cv2.imwrite('test5.png', (detail_textures * uv_vis_masks)[0].permute(1, 2, 0).detach().cpu().numpy()*255)
        # cv2.imwrite('test6.png', (extracted_textures * uv_vis_masks)[0].permute(1, 2, 0).detach().cpu().numpy()*255)
        
        
        # coarse_verts_list = coarse_meshes.verts_list()
        # # coarse_verts_normals = coarse_meshes.verts_normals_padded()
        # coarse_verts_uvs = texUVs.verts_uvs_padded()
        # dis_packed = torch.zeros(0, 1, device=imgs.device)
        # for i in range(len(coarse_meshes)):
        #     vlen = len(coarse_verts_list[i])
        #     # n = coarse_verts_normals[i][:vlen]
        #     d = displacement_maps[i]
        #     x = torch.tensor(coarse_verts_uvs[i][:, 0][:vlen] * self.tex_width, dtype=torch.int64, device=imgs.device)
        #     y = torch.tensor(coarse_verts_uvs[i][:, 1][:vlen] * self.tex_height, dtype=torch.int64, device=imgs.device)
        #     dis_packed = torch.cat((dis_packed, d[:, x, y]), dim=0)
        # detail_meshes = coarse_meshes.offset_verts(dis_packed)
        # detail_verts = detail_meshes.verts_packed()
        
        # # TEST: gt RT
        # R = cam_R_gt.view(-1, 3, 3)[0][None]
        # T = cam_T_gt[0][None]

        
        # if R[0][0][0] > 0:
        #     # inverse normal
        #     mesh._set_verts_normals(mesh.verts_normals_padded() * -1)
        
        # # vis coarse and detail mesh -------------------------------------------------------------
        # rendered_img_coarse = rendered_coarse_imgs[0]
        # rendered_img_coarse = rendered_img_coarse.detach().cpu().numpy()
        # rendered_img_coarse *= 255
        
        # rendered_img_detail = rendered_detail_imgs[0]
        # rendered_img_detail = rendered_img_detail.detach().cpu().numpy()
        # rendered_img_detail *= 255
        
        # # TEST: vis gt mesh
        # upper_gt_path = '/workspace/data/Datasets/D3G_and_SIZER/garments/%s/algined_cloth1_repose.obj' % sid       
        # gt_mesh = load_textured_mesh(upper_gt_path, device=imgs.device)
        # gt_mesh.textures = texUVs[0]
        # renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs[0][None], T=cam_Ts[0][None], device=imgs.device)
        # rendered_gt_imgs = renderer(gt_mesh)
        # rendered_img_gt = rendered_gt_imgs[0]
        # rendered_img_gt = rendered_img_gt.detach().cpu().numpy()
        # rendered_img_gt *= 255
        
        # img = imgs[0].cpu().numpy()
        # img *= 255.0
        # img = img.transpose((1, 2, 0))
        # # test: replace background
        # # img = cv2.resize(img, (540, 540))
        # # for i in range(len(rendered_img_coarse)):
        # #     for j in range(len(rendered_img_coarse[i])):
        # #         if sum(rendered_img_coarse[i][j]) == 255 * 3:
        # #             rendered_img_coarse[i][j][0] = img[i][j][0]
        # #             rendered_img_coarse[i][j][1] = img[i][j][1]
        # #             rendered_img_coarse[i][j][2] = img[i][j][2]
            
        # cv2.imwrite('ori_img.jpg', img)
        # cv2.imwrite('rendered_img_coarse.jpg', rendered_img_coarse.detach().cpu().numpy())
        # cv2.imwrite('rendered_img_detail.jpg', rendered_img_detail)
        # cv2.imwrite('rendered_img_gt.jpg', rendered_img_gt)
        # ---------------------------------------------------------------
        
        
        if self.create_detail_meshes:
            
            for i in range(len(coarse_meshes)):
                dense_faces = self.garPcapsLayers[types[i]].dense_faces
                dense_verts_uvs = self.garPcapsLayers[types[i]].dense_verts_uvs
                x = (dense_verts_uvs[:, 0] * self.tex_height).long()
                y = (dense_verts_uvs[:, 1] * self.tex_width).long()
                dense_verts = uv_coarse_vertices.permute(0, 2, 3, 1)[i, x, y, :]
                
                i = 0
                vt = (uvcoords[i] + 1) / 2
                face_verts_order = coarse_faces[i].reshape(-1)
                face_uv_order = uvfaces[i].reshape(-1)
                v = coarse_verts[i].clone()
                x = (vt[face_uv_order, 0] * 255).long()
                y = (vt[face_uv_order, 1] * 255).long()
                v[face_verts_order, :] = uv_coarse_vertices[i].permute(1, 2, 0)[y, x, :]
                save_obj(v, coarse_faces[i], 'test.obj')
                
                
                i = 0
                types = torch.where(gtypes == 1)[1]
                vt = (uvcoords[i] + 1) / 2
                dense_faces = self.garPcapsLayers[types[i]].dense_faces
                face_verts_order = dense_faces.reshape(-1)
                dense_uvfaces = self.garPcapsLayers[types[i]].dense_faces_uvs
                face_uv_order = dense_uvfaces.reshape(-1)
                vnum = self.garPcapsLayers[types[i]].dense_vnum
                v = torch.zeros(vnum, 3, device=self.device)
                x = (vt[face_uv_order, 0] * 255).long()
                y = (vt[face_uv_order, 1] * 255).long()
                v[face_verts_order, :] = uv_detail_vertices[i].permute(1, 2, 0)[y, x, :]
                save_obj(v, dense_faces, 'test.obj')
                
                a = np.zeros((256, 256, 3))
                a[1 - y.cpu(), x.cpu(), 0] = 255
                cv2.imwrite('test.jpg', a)
                
                m = uv_masks[i,:,:,:].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
                m[255 - x.cpu(), y.cpu()] = [0,0,1]
                cv2.imwrite('test.png', m * 255)
                
                h = uv_detail_vertices[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
                h = (h - h.min()) / (h.max() - h.min())
                h[y.cpu(), x.cpu(), 2] += 0.5
                cv2.imwrite('test.png', h * 255)
                
                vt = texUVs.verts_uvs_padded()[i]
                face_verts_order = coarse_faces[i].reshape(-1)
                face_uv_order = uvfaces[i].reshape(-1)
                x = (vt[face_uv_order, 1] * self.tex_height - 1).long()
                y = (vt[face_uv_order, 0] * self.tex_width - 1).long()
                off = torch.zeros(coarse_verts[i].shape[0], device=self.device)
                d = displacement_maps.detach()
                off[face_verts_order] = d[i, :, self.tex_height - 1 - x, y]
                n = coarse_normals[i].detach()
                off = off.unsqueeze(1) * n
                vlen = coarse_meshes[i].verts_packed().shape[0]
                detail_mesh = coarse_meshes[i].offset_verts(off[:vlen])
                detail_verts = detail_mesh.verts_packed()
                detail_faces = detail_mesh.faces_packed()
                save_obj(detail_verts, detail_faces, 'test.obj')
                
                # texture
                z = textures[0].permute(1, 2, 0).detach().cpu().numpy()   
                cv2.imwrite('test_tex.png', z * 255)
                
                # uv_mask
                z = uv_masks[0].permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite('test_uv_mask.png', z * 255)
                
                # uv_mask == texture
                
                # uv_coarse_vertices
                z = uv_coarse_vertices[0].permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite('test_uv_coarse_verts.png', z * 255 + 50)
                
                # uv_detail_vertices
                z = uv_detail_vertices[0].permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite('test_uv_detail_verts.png', z * 255 + 50)
                
                # uv_coarse_vertices of dense template
                z = uv_coarse_vertices[0].permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite('test_uv_dense_coarse_verts.png', z * 255 + 50)
                
            pass
        
        # if self.create_detail_meshes:
        #     # subdivided_meshes = self.subdivider(coarse_meshes)
        #     for i in range(len(coarse_meshes)):
        #         vertices = coarse_meshes[i].verts_packed()
        #         faces = coarse_meshes[i].faces_packed()
        #         # save_obj(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), 'test.obj')
        #         # v_subdivided, f_subdivided = subdivide_mesh('/workspace/data/Datasets/D3G_and_SIZER/tmps/Shirt/garment_tmp.obj', times=2)
        #         # dense_meshes = Meshes([torch.tensor(v_subdivided)], [torch.tensor(f_subdivided)])
        #         # dense_verts_list = dense_meshes.verts_list()
        #         # dense_verts_normals = dense_meshes.verts_normals_padded()
        #         types = torch.where(gtypes == 1)[1]
        #         dense_vf_vindex = self.garPcapsLayers[types[i]].dense_vf_vindex
        #         dense_vf_findex = self.garPcapsLayers[types[i]].dense_vf_findex
        #         dense_faces_uvs = self.texPcapsLayers[types[i]].dense_faces_uvs
		#         dense_verts_uvs = self.texPcapsLayers[types[i]].dense_verts_uvs
        #         dense_texUVs = TexturesUV(maps=textures[i].unsqueeze(0).permute(0, 2, 3, 1), faces_uvs=[dense_faces_uvs], verts_uvs=[dense_verts_uvs])  
        #         dense_verts_uvs = texUVs.verts_uvs_padded()
        #         vlen = self.garPcapsLayers[types[i]].dense_vnum
        #         d = displacement_maps[i].detach()
        #         x = torch.tensor(dense_verts_uvs[:, 0][:vlen] * self.tex_height, dtype=torch.int64, device=imgs.device)
        #         y = torch.tensor(dense_verts_uvs[:, 1][:vlen] * self.tex_width, dtype=torch.int64, device=imgs.device)
                
        #         dense_vertices = uv_detail_vertices[i, :, x, y].permute(1, 0)
        #         dense_faces = self.garPcapsLayers[types[i]].dense_face_index
        #         save_obj(dense_vertices, dense_faces, 'test_dense.obj')
                
        #         dense_meshes = dense_meshes.offset_verts(dis)
        #         detail_verts = detail_meshes.verts_packed()

        # from pytorch3d.io import save_obj
        
        # save_obj('test.obj', coarse_verts[0], coarse_faces[0], verts_uvs=texUVs.verts_uvs_padded()[0], faces_uvs=uvfaces[0], texture_map=textures[0].permute(1, 2, 0))
        
        
        # test
        # erode_displacement_maps = displacement_maps.detach().clone() * erode(uv_vis_masks)
        # displacement_maps = dilate(displacement_maps, ksize=11)
        
        # TEST: save coarse mesh and detail mesh ---------------------------------------------------------
        if self.inferring and self.mesh_save_folder is not None:
            from pytorch3d.io import save_obj
            temp_displacement_maps = displacement_maps # * uv_vis_masks
            types = torch.where(gtypes == 1)[1]
            for i in range(len(types)):
                t = 'up' if i % 2 == 0 else 'bottom'
                obj_name = img_names[i // 2].replace('.jpg', '')
                coarse_save_path = os.path.join(self.mesh_save_folder, obj_name + '_%s_coarse.obj' % t)
                # test
                # coarse_save_path = 'test_%s_coarse.obj' % t
                # if not os.path.exists(coarse_save_path):
                save_obj(coarse_save_path, coarse_verts[i][:self.garmentvnums[types[i]]], coarse_meshes[i].faces_packed(), 
                        verts_uvs=self.texPcapsLayers[types[i]].verts_uvs, 
                        faces_uvs=self.texPcapsLayers[types[i]].faces_uvs, 
                        texture_map=torch.as_tensor(textures[i].permute(1, 2, 0).detach().cpu().numpy()[:,:,::-1].copy(), device=self.device))
                cv2.imwrite(coarse_save_path.replace('.obj', '.png'), textures.permute(0, 2, 3, 1)[i].detach().cpu().numpy() * 255)
                dense_save_path = coarse_save_path.replace('_coarse.obj', '.obj')
                # if not os.path.exists(dense_save_path):
                if self.upsample_dismap:
                    dense_mesh = subdivide_mesh_by_meshlab(coarse_save_path, iter=2, thres=0, uv_dis=F.interpolate(temp_displacement_maps, (4096, 4096), mode='bilinear')[i], use_neighbor=self.use_neighbor)
                else:
                    dense_mesh = subdivide_mesh_by_meshlab(coarse_save_path, iter=2, thres=0, uv_dis=temp_displacement_maps[i], use_neighbor=self.use_neighbor)
                save_obj(dense_save_path, torch.as_tensor(dense_mesh['v']), torch.as_tensor(dense_mesh['f']), 
                        verts_uvs=self.garPcapsLayers[types[i]].dense_vt, 
                        faces_uvs=self.garPcapsLayers[types[i]].dense_faces_uvs, 
                        texture_map=torch.as_tensor(textures[i].permute(1, 2, 0).detach().cpu().numpy()[:,:,::-1].copy(), device=self.device))
                cv2.imwrite(dense_save_path.replace('.obj', '.png'), textures.permute(0, 2, 3, 1)[i].detach().cpu().numpy() * 255)
        # -------------------------------------------------------------------------------------------------
        
        
        # TEST: SAVE DENSE MESH ---------------------------------------------------------------------------------------------------------
        # save_folder = '/workspace/data/Results/infer_results/detail_new1/detail_new1_pca64_ep40_bth0/detail_new1_pca64_ep40_bth0_classify_by_net_without_icp_without_stitch/mesh/'
        # save_folder = 'test/'
        # types = torch.where(gtypes == 1)[1]
        # gps_rec_dense = torch.Tensor(0, 3)
        # garbatch_dense = torch.Tensor(0)
        # face_index_dense = torch.Tensor(0, 3)
        # from pytorch3d.io import save_obj
        # voffset = 0
        # for ind in range(len(types)):
        #     type_name = self.garments[types[ind]]
        #     vnum = self.garmentvnums[types[ind]]
        #     isupper = (type_name != 'Shorts' and type_name != 'Pants')
        #     dense_template_path = '/workspace/BCNet/data/texture_data_%s_256.npy' % type_name
        #     dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1')
        #     dense_vertices, dense_colors, dense_faces = upsample_mesh(coarse_verts[ind][:vnum].detach().cpu().numpy(), coarse_normals[ind].detach().cpu().numpy(), coarse_faces[ind].detach().cpu().numpy(), displacement_maps[ind, 0].detach().cpu().numpy(), textures[ind].permute(1, 2, 0).detach().cpu().numpy(), dense_template)
        #     from utils import save_obj
        #     # save_obj(dense_vertices, dense_faces, os.path.join(save_folder, '%d_%d_%s_dense.obj' % (self.tot, ind // 2, 'up' if isupper else 'bottom')))
        #     gps_rec_dense = torch.cat((gps_rec_dense, torch.as_tensor(dense_vertices)), dim=0)
        #     garbatch_dense = torch.cat((garbatch_dense, torch.as_tensor([ind] * len(dense_vertices))), dim=0)
        #     face_index_dense = torch.cat((face_index_dense, torch.as_tensor(dense_faces + voffset)), dim=0)
        #     voffset = len(gps_rec_dense)
        # gps_rec_dense = gps_rec_dense.to(self.device)
        # garbatch_dense = garbatch_dense.to(self.device)
        # face_index_dense = face_index_dense.to(self.device)
        # ----------------------------------------------------------------------------------------------------------------------------------

            # dense_template_path = '/workspace/BCNet/data/texture_data_Pants_256.npy'
            # dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1')
            # dense_vertices, dense_colors, dense_faces = upsample_mesh(coarse_verts[1][:vnum].detach().cpu().numpy(), coarse_normals[1][:1180].detach().cpu().numpy(), coarse_faces[1].detach().cpu().numpy(), displacement_maps[1, 0].detach().cpu().numpy(), textures[1].permute(1, 2, 0).detach().cpu().numpy(), dense_template)
            # from utils import save_obj
            # save_obj(dense_vertices, dense_faces, 'test_lower.obj')

        # # TEST: 31587
        # dense_template_path = '/workspace/BCNet/data/texture_data_Shirt_256.npy'
        # dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1')
        # dense_faces = dense_template['f']
        # valid_pixel_ids = dense_template['valid_pixel_ids']
        # fid = 27648
        # fvals = dense_faces[fid]
        # vids = valid_pixel_ids[fvals]
        # h = dis_vis.repeat(3, 1, 1).clone()
        # for v in vids:
        #     x = v.item() // self.tex_height
        #     y = v.item() % self.tex_width
        #     h[0, y, x] = 0
        #     h[1, y, x] = 0
        #     h[2, y, x] = 1
        # cv2.imwrite('test.png', h.permute(1, 2, 0).cpu().numpy() * 255)
            

        # # TEST: make dense template
        # save_path = 'data/texture_data_Shirt_256.npy'
        # a = load_obj("/workspace/data/Datasets/D3G_and_SIZER/tmps/T-shirt/dense_template.obj")
        # make_dense_template(uvcoords[0], uvfaces[0], coarse_faces[0], a[2], uv_coarse_vertices_masks[0,0], self.tex_height, save_path)

        # b = load_obj('untitled1.obj')
        # dense_vt = b[1]
        # dense_f = b[2]
        # dense_fuv = b[3]
        # dense_v = b[0]
        
        # face_verts_order = dense_f.reshape(-1)
        # face_uv_order = dense_fuv.reshape(-1)
        # vt = (uvcoords[i] + 1) / 2
        # x = (dense_vt[face_uv_order, 1] * (self.tex_height - 1)).long()
        # y = (dense_vt[face_uv_order, 0] * (self.tex_width - 1)).long()
        
        # a = uv_coarse_vertices.clone()
        
        # a[0][0, 255 - x, y] = 1
        # a[0][1, 255 - x, y] = 1
        # a[0][2, 255 - x, y] = 1
        
        # cv2.imwrite('test4.png', a[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        

        # coarse_verts[0][face_verts_order, :] = uv_coarse_vertices[0].permute(1, 2, 0)[x, y, :]
        # save_obj(coarse_verts[0], coarse_faces[0], 'test.obj')

        # make dense template --------------------------------------------------------------
        # save_path = 'data/texture_data_Shirt_256_new.npy'
        # # a = load_obj("/workspace/data/Datasets/D3G_and_SIZER/tmps/Shirt/dense_template.obj")
        # # make_dense_template(uvcoords[1], uvfaces[1], coarse_faces[1], a[2], uv_coarse_vertices_masks[1,0], self.tex_height, save_path)
        # make_dense_template(uvcoords[0], uvfaces[0], coarse_faces[0], coarse_faces[0], uv_coarse_vertices_masks[0,0], self.tex_height, save_path)
        # ----------------------------------------------------------------------------------

        # # TEST: MAKE DENSE TEMPLATE VT
        # dense_template_path = '/workspace/BCNet/data/texture_data_Shirt_256.npy'
        # dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1')
        # dense_faces = dense_template['f']
        # valid_pixel_ids = dense_template['valid_pixel_ids']
        # x_coords = dense_template['x_coords']
        # y_coords = dense_template['y_coords']
        # vt_x = y_coords[valid_pixel_ids] / self.tex_height
        # vt_y = x_coords[valid_pixel_ids] / self.tex_width
        # for f in dense_faces:
        #     fvals = valid_pixel_ids[f]
        #     vids = valid_pixel_ids[vids]
        #     print(vids)
        #     break

        # TODO: uv displacements to verts displacements (for lap loss)
        # TRY COARSE
        # bz = uvcoords.shape[0]
        # vt = (uvcoords + 1) / 2
        # face_verts_order = coarse_faces.reshape(bz, -1)
        # face_uv_order = uvfaces.reshape(bz, -1)
        # verts_displacments = torch.zeros_like(coarse_verts)
        # for i in range(len(vt)):
        #     x = (vt[i, face_uv_order[i], 0] * 255).long()
        #     y = (vt[i, face_uv_order[i], 1] * 255).long()
        #     verts_displacments[i, face_verts_order[i], :] = displacement_maps[i].permute(1, 2, 0)[y, x, :]
        verts_displacments = None
        
        self.tot += 1
        
        
        # VIS: GARMENT PCA PARAMS ------------------------------------
        # ind = 5
        # t = self.garments[ind]
        # for i in range(0, 10):
        #     for j in np.arange(-5.0, 5.5, 0.5):
        #         range_pca_params = torch.zeros_like(pcas_perg[0].unsqueeze(0))
        #         range_pca_params[0, i] = j
        #         random_pca_ps=self.garPcapsLayers[ind](range_pca_params).reshape(shapes_gtype.shape[0],self.garmentvnums[ind],3)
        #         save_folder = 'check/garment_pca_params/%s/%d' % (t, i)
        #         if not os.path.exists(save_folder):
        #             os.mkdir(save_folder)
        #         save_path = os.path.join(save_folder, '%.1f.obj' % j)     
        #         save_obj(random_pca_ps[0], coarse_faces[0 if ind < 4 else 1], save_path)
        # ------------------------------------------------------------
        
        # VIS: GARMENT POSE --------------------------------------------
        # ind = 5
        # t = self.garments[ind]
        # for i in range(46, 47):
        #     for j in np.arange(-10.0, 10.5, 0.5):
        #         range_poses = poses.clone()
        #         range_poses[0, i] = j
        #         deform_rec,transforms,pose_Rs,Js_transformed=self.skinDeformNet(gps_pca,Js,ws,range_poses,imgbatch)
        #         v = deform_rec[self.garPcapsLayers[2].vnum:]
        #         gfolder = 'check/garment_poses/%s' % t
        #         if not os.path.exists(gfolder):
        #             os.mkdir(gfolder)
        #         save_folder = os.path.join(gfolder, str(i))
        #         if not os.path.exists(save_folder):
        #             os.mkdir(save_folder)
        #         save_path = os.path.join(save_folder, '%.1f.obj' % j)
        #         save_obj(v, coarse_faces[0 if ind < 4 else 1], save_path)
        # --------------------------------------------------------------
        self.tot = self.tot + 1
        if True:
            gps_rec_dense, garbatch_dense, face_index_dense = None, None, None
        if not self.use_detail:
            detail_shading_images = None
        if self.gar_classification:
            return garbatch,gps_pca,deform_rec,deform_rect,gps_diss,gps_rec,ws,shapes,poses,trans,pcas_perg,displacements,Js_transformed,body_faces,body_ns,body_tpose_ps, body_ps,up_gar_prob, bottom_gar_prob, cam_Rs, cam_Ts, tex_pcas_perg, Js_2d, rendered_coarse_imgs, rendered_detail_imgs, detail_verts, dis_packed, coarse_render_masks, detail_textures, extracted_textures, uv_vis_masks, lights, verts_displacments, gps_rec_dense, garbatch_dense, face_index_dense, detail_shading_images
        else:
            return garbatch,gps_pca,deform_rec,deform_rect,gps_diss,gps_rec,ws,shapes,poses,trans,pcas_perg,displacements,Js_transformed,body_faces,body_ns,body_tpose_ps, body_ps, cam_Rs, cam_Ts, tex_pcas_perg, Js_2d, rendered_coarse_imgs, rendered_detail_imgs, detail_verts, dis_packed, coarse_render_masks, detail_textures, extracted_textures, uv_vis_masks, lights, verts_displacments, gps_rec_dense, garbatch_dense, face_index_dense, detail_shading_images
