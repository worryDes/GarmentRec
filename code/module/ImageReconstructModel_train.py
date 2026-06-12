import os.path

import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from module.GCNs import ResidualAdd, MultiPerceptro, SpiralConv
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
                 device='cuda:1'):
        super(ImageReconstructModel, self).__init__()
        if displacement_scale is None:
            raise NotImplementedError
        self.device = device
        self.pca_dim = pca_dim
        self.pca_folder = pca_folder
        self.imgEncoder = ImageEncoder(tran_mean=tran_mean, infer_camera=infer_camera,
                                       light_instance_scale=light_instance_scale)
        self.use_detail = use_detail
        self.n_detail = 128
        if use_detail:
            self.detailEncoder = DetailEncoder(n_detail=self.n_detail)
        self.inferring = inferring
        self.vis_save_folder = vis_save_folder
        self.mesh_save_folder = mesh_save_folder

        self.bcnet_tran_mean = bcnet_tran_mean
        self.tran_mean = tran_mean
        self.garments = garments
        self.garmentvnums = garmentvnums
        self.upper_type_num = upper_type_num
        self.lower_type_num = len(garments) - upper_type_num
        assert (self.upper_type_num > 0 and self.lower_type_num > 0)
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

        self.tot = 0

        self.garPcaparamLayers = nn.ModuleList(
            [GarmentPcaLayer(gtype, 10 + self.imgEncoder.gar_latent_size, pca_dim) for gtype in self.garments])

        if infer_tex:
            self.texPcaparamLayers = nn.ModuleList(
                [TexturePcaLayer(gtype, 10 + self.imgEncoder.gar_latent_size, tex_pca_dim) for gtype in self.garments])

        # self.detailDecoderLayers = nn.ModuleList([DetailDecoder(gtype, self.n_detail, self.tex_width, self.tex_height) for gtype in self.garments])

        self.detailDecoderLayers = nn.ModuleList([Generator(latent_dim=10 + 216 + self.n_detail, out_channels=1,
                                                            out_scale=displacement_scale, sample_mode='bilinear',
                                                            gtype=gtype) for gtype in self.garments]).to(device)

        # self.garPcapsLayers=nn.ModuleList([GarmentPcaDecodeLayer(osp.join(osp.dirname(__file__),'../../body_garment_dataset/tmps/%s/pca_data.npz'%gtype)) for gtype in self.garments])
        self.garPcapsLayers = nn.ModuleList(
            [GarmentPcaDecodeLayer(
                osp.join(pca_folder, gtype, 'pca_data_%d.npz' % pca_dim),
                osp.join(dense_template_folder, 'texture_data_%s_%d.npy' % (gtype, tex_height)))
                for gtype in self.garments])

        if infer_tex:
            self.texPcapsLayers = nn.ModuleList(
                [TexturePcaDecodeLayer(osp.join(pca_folder, gtype, 'pca_data_%d.npz' % pca_dim),
                                       osp.join(pca_folder, gtype, 'tex_uv.pkl'),
                                       osp.join(pca_folder, gtype, 'uv_mask.png'), gtype)
                 for gtype in self.garments])

        # self.patchEncoder=MultiPerceptro([3*32*32,1024,524,256])
        self.skinWsNet = SkinWeightNet
        # use pretrained model, fix weights
        for param in self.skinWsNet.parameters():
            param.requires_grad = False
        self.smpl = SMPL(smpl_model_path, obj_saveable=True)

        self.skinDeformNet = SkinDeformNet(self.smpl)

        # classification Net
        self.gar_classification = with_classification
        if with_classification:
            self.up_classifier = nn.Linear(self.imgEncoder.gar_latent_size, self.upper_type_num)
            self.up_dropout = nn.Dropout(p=0.2)
            self.bottom_classifier = nn.Linear(self.imgEncoder.gar_latent_size, self.lower_type_num)
            self.bottom_dropout = nn.Dropout(p=0.2)

        self.check = check

        # test
        self.visited = set()

        self.subdivider = SubdivideMeshes()

        self.dense_template_folder = dense_template_folder

        pi = np.pi
        self.constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))], device=self.device).float()

        self.tensorboard_logging = tensorboard_logging
        if tensorboard_logging:
            assert (writer is not None)
            self.writer = writer


    def extract_tris_infos(self, tris_infos):
        edge_index = tris_infos['edge_index']
        garbatch = tris_infos['gar_batch']
        face_index = tris_infos['face_index']
        vf_vindex = tris_infos['vf_vindex']
        vf_findex = tris_infos['vf_findex']
        self.edge_index = edge_index
        self.face_index = face_index
        self.vf_vindex = vf_vindex
        self.vf_findex = vf_findex
        self.garbatch = garbatch
        return garbatch, edge_index, face_index, vf_vindex, vf_findex

    def pro_ps(self, ps, cam_k):
        if cam_k.shape[1] == 4:
            ones = torch.ones(ps.shape[0], device=ps.device).reshape(-1, 1)
            ps = torch.cat((ps, ones), dim=1)
        proPs = ps.matmul(cam_k.transpose(0, 1))
        depth = proPs[:, 2].reshape(-1, 1)
        select = (depth >= -1.0e-4) * (depth <= 1.0e-4)
        signs = depth.sign()
        signs[(signs >= -0.01) * (signs <= 0.01)] = 1.0
        depth[select] = signs[select] * 1.0e-4
        proPs = torch.cat((proPs[:, 0:2] / depth, proPs[:, 2].reshape(-1, 1)), dim=-1)
        return proPs

    def forward(self, imgs, img_names=None, gtypes=None, Rs_T=None, cam_k=None, input_imgbatch=None, cam_R_gt=None,
                cam_T_gt=None, Js_2d_gt=None, tex_pca_perg_gt=None, img_masks=None, imgs_perg=None,
                **kwargs):
        batch_size = imgs.shape[0]
        cam_Rs, cam_Ts = None, None
        if not self.infer_camera:
            shapes, poses, trans, garlatents, lights, _ = self.imgEncoder(imgs)
        else:
            shapes, poses, trans, garlatents, cam_exts, lights, _ = self.imgEncoder(imgs)
        poses_pre = poses.clone()

        # lights_per_gar = torch.cat((lights.view(-1, 3+3), lights.view(-1, 3+3)), 1).view(-1, 3+3)
        lights_per_gar = torch.cat((lights, lights), 1).view(-1, 9, 3)

        if self.use_detail:
            detail_latents = self.detailEncoder(imgs)
        # if self.infer_camera:
        #     cam_exts = self.cameraEncoder(imgs)
        if 'garl' in kwargs:
            garlatents = kwargs['garl']
        self.shapes = shapes
        self.poses = poses
        self.trans = trans
        self.garl = garlatents
        if self.infer_camera:
            cam_Rs, cam_Ts = get_batch_RT(cam_exts, device=cam_exts.device)
        self.cam_Rs, self.cam_Ts = cam_Rs, cam_Ts
        batch_num = shapes.shape[0]
        if gtypes is None:
            assert (self.gar_classification)
        if self.gar_classification:
            tmpfs = F.relu(garlatents)
            up_gar_prob = self.up_classifier(self.up_dropout(tmpfs))
            bottom_gar_prob = self.bottom_classifier(self.bottom_dropout(tmpfs))
            up_index = up_gar_prob.max(1)[1]
            up_gtypes = up_gar_prob.new_zeros(up_gar_prob.shape).scatter(1, up_index.repeat(2, 1).transpose(0, 1),
                                                                         up_gar_prob.new_ones(up_gar_prob.shape))
            bottom_index = bottom_gar_prob.max(1)[1]
            bottom_gtypes = bottom_gar_prob.new_zeros(bottom_gar_prob.shape).scatter(1, bottom_index.repeat(2,
                                                                                                            1).transpose(
                0, 1), bottom_gar_prob.new_ones(bottom_gar_prob.shape))
            if gtypes is not None:
                if type(gtypes) is not torch.Tensor:
                    gtypes = torch.tensor(gtypes, dtype=torch.long, device=imgs.device)
                # nonzero postions [r, c]
                rows, cols = torch.nonzero(gtypes >= 0, as_tuple=False).transpose(0, 1)
            # tgtypes:
            # line1: subject1: Shirt_prob, T-shirt_prob, Pants_prob, Shorts_prob
            # line2: subject2: Shirt_prob, T-shirt_prob, Pants_prob, Shorts_prob
            # ...
            tgtypes = torch.cat((up_gtypes, bottom_gtypes), dim=-1)
            # modify prob result by gt
            if gtypes is not None:
                for r, c in zip(rows, cols):
                    c = gtypes[r, c].item()

                    if self.model_type == 'D3G':
                        # d3g
                        # upper
                        if c < self.upper_type_num:
                            tgtypes[r, 0:4] = 0
                            tgtypes[r, c] = 1
                        # lower
                        elif c < len(self.garments):
                            tgtypes[r, 4:] = 0
                            tgtypes[r, c] = 1
                    elif self.model_type == 'BCNET':
                        # bcnet
                        if c < 2:
                            tgtypes[r, 0:2] = 0
                            tgtypes[r, c] = 1
                        elif c < 6:
                            tgtypes[r, 2:] = 0
                            tgtypes[r, c] = 1
                    else:
                        raise NotImplementedError
            gtypes = tgtypes
        self.gtypes = gtypes
        # 同一类组合到一起
        if self.use_detail:
            ordered_datas, ordered_imgbids, ordered_gtypes = order_data_follow_gartypes(
                [shapes, poses_pre, garlatents, detail_latents], batch_num, None, gtypes, self.garmentvnums,
                self.garments)
            pca_datas = []
            for ind, (shapes_gtype, poses_gtype, latents_gtype, detail_latents_gtype) in zip(ordered_gtypes,
                                                                                             ordered_datas):
                pca_params = self.garPcaparamLayers[ind](torch.cat((shapes_gtype, latents_gtype), dim=-1))
                tex_pca_params = self.texPcaparamLayers[ind](torch.cat((shapes_gtype, latents_gtype), dim=-1))
                pca_ps = self.garPcapsLayers[ind](pca_params).reshape(shapes_gtype.shape[0], self.garmentvnums[ind], 3)
                texs = self.texPcapsLayers[ind](tex_pca_params).reshape(shapes_gtype.shape[0],
                                                                        self.tex_height * self.tex_width, 3)
                dis_maps = self.detailDecoderLayers[ind](
                    torch.cat((shapes_gtype, poses_gtype, detail_latents_gtype), dim=-1)).reshape(shapes_gtype.shape[0],
                                                                                                  self.tex_height * self.tex_width)
                pca_datas.append([pca_params, pca_ps, tex_pca_params, texs, dis_maps])
            [pcas_perg, gps_pca, tex_pcas_perg, textures, displacement_maps], tris_infos = unorder_data_follow_imgbatch(
                pca_datas, ordered_imgbids, ordered_gtypes, batch_num, self.garPcapsLayers, self.texPcapsLayers, True,
                True, True, True, True)
        else:
            ordered_datas, ordered_imgbids, ordered_gtypes = order_data_follow_gartypes([shapes, poses, garlatents],
                                                                                        batch_num, None, gtypes,
                                                                                        self.garmentvnums,
                                                                                        self.garments)
            pca_datas = []
            for ind, (shapes_gtype, _, latents_gtype) in zip(ordered_gtypes, ordered_datas):
                pca_params = self.garPcaparamLayers[ind](torch.cat((shapes_gtype, latents_gtype), dim=-1))
                tex_pca_params = self.texPcaparamLayers[ind](torch.cat((shapes_gtype, latents_gtype), dim=-1))
                pca_ps = self.garPcapsLayers[ind](pca_params).reshape(shapes_gtype.shape[0], self.garmentvnums[ind], 3)
                texs = self.texPcapsLayers[ind](tex_pca_params).reshape(shapes_gtype.shape[0],
                                                                        self.tex_height * self.tex_width, 3)
                pca_datas.append([pca_params, pca_ps, tex_pca_params, texs])
            [pcas_perg, gps_pca, tex_pcas_perg, textures], tris_infos = unorder_data_follow_imgbatch(pca_datas,
                                                                                                     ordered_imgbids,
                                                                                                     ordered_gtypes,
                                                                                                     batch_num,
                                                                                                     self.garPcapsLayers,
                                                                                                     self.texPcapsLayers,
                                                                                                     True, True, True,
                                                                                                     True, True)
        textures = textures.view(batch_num * 2, self.tex_height, self.tex_width, 3).permute(0, 3, 1, 2)

        # check --------------------------
        if self.check:
            save_obj(gps_pca, tris_infos['face_index'], 'check/check_pca.obj')
        # check end ----------------------

        garbatch, edge_index, face_index, vf_vindex, vf_findex = self.extract_tris_infos(tris_infos)
        imgbatch = imgBatchFromGarBatch(garbatch, gtypes)
        if input_imgbatch is not None:
            assert ((imgbatch - input_imgbatch).sum() == 0)
        self.imgbatch = imgbatch
        Js, body_ns = self.skinDeformNet.skeleton(shapes, True)

        diss = (gps_pca.unsqueeze(1) - Js[imgbatch, :]).norm(dim=-1)

        if self.skinWsNet.use_normal:
            vnorms = compute_vnorms(gps_pca, face_index, vf_vindex, vf_findex)
            ws = self.skinWsNet(torch.cat((gps_pca, vnorms, diss), dim=-1), edge_index, garbatch)
        else:
            ws = self.skinWsNet(torch.cat((gps_pca, diss), dim=-1), edge_index, garbatch)

        if cam_k is None:
            cam_k = torch.Tensor([[3.0375e+03, 0.0000e+00, 2.7000e+02],
                                  [0.0000e+00, 3.0375e+03, 2.7000e+02],
                                  [0.0000e+00, 0.0000e+00, 1.0000e+00]])
            cam_Rt = torch.tensor([[1, 0, 0, self.tran_mean[0] - self.bcnet_tran_mean[0]],
                                   [0, 1, 0, self.tran_mean[1] - self.bcnet_tran_mean[1]],
                                   [0, 0, 1, self.tran_mean[2] - self.bcnet_tran_mean[2]]])
            cam_k = cam_k.matmul(cam_Rt)
            cam_k = cam_k.to(imgs.device)

        # deform_rec: deformed pca garment
        # TEST: pca gt
        # gps_pca = pca_verts_gt
        deform_rec, transforms, pose_Rs, Js_transformed = self.skinDeformNet(gps_pca, Js, ws, poses, imgbatch)

        # check -------------------------------
        if self.check:
            save_obj(deform_rec, tris_infos['face_index'], 'check/check_deform_pca.obj')
        # check end ---------------------------

        self.transforms = transforms
        # deform_norms=compute_vnorms(deform_rec,face_index,vf_vindex,vf_findex)
        if 'pro_fs' in kwargs:
            pro_features = kwargs['pro_fs']
        else:
            # rect: rec + t
            deform_rect = deform_rec + trans[imgbatch, :]

            # check -----------------------------------
            if self.check:
                save_obj(deform_rect, tris_infos['face_index'], 'check/check_deform_trans_pca.obj')
            # check end -------------------------------

        Js_transformed = Js_transformed + trans.unsqueeze(1)
        body_faces = torch.tensor(self.smpl.faces)
        body_tpose_ps, _, _, _ = self.smpl(shapes, torch.zeros_like(pose_Rs), True, False)
        body_ps, _, _, _ = self.smpl(shapes, pose_Rs, True, False)

        # check ----------------------------------------
        if self.check:
            save_obj(body_ps[0], body_faces, 'check/check_shape_pose_body.obj')
        # check end ------------------------------------

        body_ps = body_ps + trans.unsqueeze(1)

        # check -----------------------------------------
        if self.check:
            save_obj(body_ps[0], body_faces, 'check/check_shape_pose_trans_body.obj')
        # check end -------------------------------------

        # project joints ----------------------------------------------------------------------------
        renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs, T=cam_Ts, lights=None, device=imgs.device)
        Js_2d = renderer.transform_points(Js_transformed)
        Js_2d = Js_2d[:, :, :2]
        # -------------------------------------------------------------------------------------------

        # 上衣和裤子各对应一个RT及相机
        cam_Rs_per_gar = torch.cat((cam_Rs.view(-1, 1, 3 * 3), cam_Rs.view(-1, 1, 3 * 3)), 1).view(-1, 3, 3)
        cam_Ts_per_gar = torch.cat((cam_Ts.view(-1, 1, 3), cam_Ts.view(-1, 1, 3)), 1).view(-1, 3)
        renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs_per_gar, T=cam_Ts_per_gar, lights=None,
                            device=imgs.device)
        uv_rasterizer = Pytorch3dRasterizer(self.tex_height)
        dense_triangles = []
        uv_masks = torch.zeros(0, 1, self.tex_height, self.tex_width, device=self.device)
        for g in gtypes:
            inds = torch.nonzero(g)
            dense_triangles.append(self.texPcapsLayers[inds[0][0]].dense_triangles)
            dense_triangles.append(self.texPcapsLayers[inds[1][0]].dense_triangles)
            uv_masks = torch.cat((uv_masks, self.texPcapsLayers[inds[0][0]].uv_mask_erosion.unsqueeze(0).unsqueeze(0)),
                                 dim=0)
            uv_masks = torch.cat((uv_masks, self.texPcapsLayers[inds[1][0]].uv_mask_erosion.unsqueeze(0).unsqueeze(0)),
                                 dim=0)
        max_tri_len = max([len(tri) for tri in dense_triangles])
        dense_faces = torch.zeros(0, max_tri_len, 3, device=self.device)
        for i in range(len(dense_triangles)):
            tri = dense_triangles[i]
            dense_faces = torch.cat(
                (dense_faces, pad(tri, [0, 0, 0, max_tri_len - len(tri)], mode='constant', value=-1).unsqueeze(0)),
                dim=0)

        if img_names is not None:
            img_name = img_names[0]
            sp = img_name.split('_')
            sid = sp[0] + '_' + sp[1]
            if img_name.startswith('s'):
                sid += '_' + sp[2]

        # render coarse meshes -------------------------------------------------------------------
        coarse_meshes, texUVs = create_meshes(deform_rect, face_index, textures.permute(0, 2, 3, 1), garbatch,
                                              torch.where(gtypes == 1)[1], self.garPcapsLayers, self.texPcapsLayers)
        coarse_verts = coarse_meshes.verts_padded()
        coarse_faces = coarse_meshes.faces_padded()
        uvcoords = texUVs.verts_uvs_padded()
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1  # [-1, 1]
        uvcoords[..., 1] = -uvcoords[..., 1]
        uvfaces = texUVs.faces_uvs_padded()

        uv_coarse_vertices = world2uv(coarse_verts, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()

        coarse_normals = coarse_meshes.verts_normals_padded()
        uv_coarse_normals = world2uv(coarse_normals, coarse_faces, uvcoords, uvfaces, uv_rasterizer).detach()
        uv_coarse_normals_masks = get_imgs_masks(uv_coarse_normals).detach()

        coarse_render_masks = None

        if True or self.inferring or not self.use_detail:
            # # point light
            # coarse_shading = add_pointlight(uv_coarse_vertices.permute(0,2,3,1).reshape([batch_size*2, -1, 3]), uv_coarse_normals.permute(0,2,3,1).reshape([batch_size*2, -1, 3]), lights=lights_per_gar)
            # coarse_shading_images = coarse_shading.reshape([batch_size*2, self.tex_height, self.tex_width, 3]).permute(0, 3, 1, 2)

            # sh light
            coarse_shading = add_SHlight(uv_coarse_normals, lights_per_gar, self.constant_factor)
            coarse_shading_images = coarse_shading

            coarse_textures = textures * coarse_shading_images
            texUVs._maps_padded = coarse_textures.permute(0, 2, 3, 1)
        rendered_coarse_imgs = renderer(coarse_meshes)

        # displacement map -> normal map -----------------------------------------------------------
        if self.use_detail:
            # TODO: DO WE NEED MASK?
            displacement_maps = displacement_maps.reshape(-1, 1, self.tex_height, self.tex_width) * uv_masks
            # TODO: FIXED DIS
            # displacement_maps *= 0
            uv_detail_vertices = uv_coarse_vertices + displacement_maps * uv_coarse_normals  # + A * uv_coarse_normals
            dense_detail_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size * 2, -1, 3])
            uv_detail_normals = get_vertex_normals(dense_detail_vertices, dense_faces)
            uv_detail_normals = uv_detail_normals.reshape([batch_size * 2, self.tex_height, self.tex_width, 3]).permute(
                0, 3, 1, 2).contiguous()
            uv_detail_normals = uv_detail_normals * uv_coarse_normals_masks + uv_coarse_normals * (
                        1. - uv_coarse_normals_masks)
        # ---------------------------------------------------------------------------------------------------------------

        # render detail meshes -------------------------------------------------------------
        if self.use_detail:
            # sh
            detail_shading = add_SHlight(uv_detail_normals, lights_per_gar, self.constant_factor)
            detail_shading_images = detail_shading

            detail_textures = textures * detail_shading_images
            # detail without texture
            texUVs._maps_padded = (torch.ones_like(textures) * 0.9 * detail_shading_images).permute(0, 2, 3, 1)
            rendered_detail_imgs_no_tex = renderer(coarse_meshes)

            # detail with texture
            texUVs._maps_padded = detail_textures.permute(0, 2, 3, 1)
            rendered_detail_imgs = renderer(coarse_meshes)
            detail_verts = coarse_meshes.verts_packed()
            dis_packed = displacement_maps

            # TODO: ID-MRF
            # --- extract texture
            trans_uv_vertices = renderer.transform_points(
                uv_coarse_vertices.permute(0, 2, 3, 1).reshape(-1, 256 * 256, 3), target='screen_space')
            trans_uv_vertices = trans_uv_vertices.reshape(-1, 256, 256, 3)
            trans_uv_vertices = trans_uv_vertices / imgs.shape[-1] * 2 - 1
            uv_gt = F.grid_sample(torch.cat([imgs_perg, img_masks], dim=1), trans_uv_vertices[:, :, :, :2],
                                  mode='bilinear', align_corners=False)
            extracted_textures = uv_gt[:, :3, :, :].detach()
            extracted_uv_masks = uv_gt[:, 3:4, :, :].detach()

            # camera space
            coarse_verts_cam_space = renderer.transform_points(coarse_verts, target='camera_space')
            coarse_normals_cam_space = get_vertex_normals(coarse_verts_cam_space, coarse_faces.clamp(min=0))
            uv_coarse_normals_cam_space = world2uv(coarse_normals_cam_space, coarse_faces, uvcoords, uvfaces,
                                                   uv_rasterizer).detach()
            self_occlusion_masks = (uv_coarse_normals_cam_space[:, [-1], :, :] < -0.05).float().detach()
            uv_vis_masks = (uv_masks * extracted_uv_masks * self_occlusion_masks).detach()
            # extracted_textures *= uv_vis_masks


        else:
            rendered_detail_imgs = None
            detail_verts = None
            dis_packed = None
            detail_textures = None
            extracted_textures = None
            uv_vis_masks = None

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

                m = uv_masks[i, :, :, :].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
                m[255 - x.cpu(), y.cpu()] = [0, 0, 1]
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

        verts_displacments = None

        self.tot += 1

        if True:
            gps_rec_dense, garbatch_dense, face_index_dense, gps_diss, gps_rec = None, None, None, None, None
            displacements = None
        if not self.use_detail:
            detail_shading_images = None

        if self.gar_classification:
            return garbatch, gps_pca, deform_rec, deform_rect, gps_diss, gps_rec, ws, shapes, poses, trans, pcas_perg, displacements, Js_transformed, body_faces, body_ns, body_tpose_ps, body_ps, up_gar_prob, bottom_gar_prob, cam_Rs, cam_Ts, tex_pcas_perg, Js_2d, rendered_coarse_imgs, rendered_detail_imgs, detail_verts, dis_packed, coarse_render_masks, detail_textures, extracted_textures, uv_vis_masks, lights, verts_displacments, gps_rec_dense, garbatch_dense, face_index_dense, detail_shading_images
        else:
            return garbatch, gps_pca, deform_rec, deform_rect, gps_diss, gps_rec, ws, shapes, poses, trans, pcas_perg, displacements, Js_transformed, body_faces, body_ns, body_tpose_ps, body_ps, cam_Rs, cam_Ts, tex_pcas_perg, Js_2d, rendered_coarse_imgs, rendered_detail_imgs, detail_verts, dis_packed, coarse_render_masks, detail_textures, extracted_textures, uv_vis_masks, lights, verts_displacments, gps_rec_dense, garbatch_dense, face_index_dense, detail_shading_images
