import os.path
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from smpl_pytorch.SMPL import SMPL
import numpy as np
import os.path as osp
from renderer import Renderer
import cv2
from pytorch3d.ops import SubdivideMeshes
from module.encoders import *
from module.decoders import *
from module.SkinDeformNet import SkinDeformNet
from module.GarmentDisplacementNet import GarmentDisplacementNet
from utils import *
from renderer_deca import Pytorch3dRasterizer
from torch.nn.functional import pad


class ImageReconstructModel(Module):
    def __init__(self,
                 SkinWeightNet,
                 with_classification=False,
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
                 hard_stitch=False,
                 midpairs=None,
                 infer_camera=False,
                 infer_tex=False,
                 use_detail=True,
                 inferring=False,
                 vis_save_folder=None,
                 mesh_save_folder=None,
                 create_detail_meshes=False,
                 dense_template_folder=None,
                 light_instance_scale=1.0,
                 displacement_scale=None,
                 upsample_dismap=False,
                 use_neighbor=False,
                 device='cuda:0'):
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
        self.hard_stitch = hard_stitch
        self.midpairs = midpairs
        self.infer_camera = infer_camera
        self.infer_tex = infer_tex
        self.create_detail_meshes = create_detail_meshes
        self.upsample_dismap = upsample_dismap
        self.use_neighbor = use_neighbor

        self.garPcaparamLayers = nn.ModuleList(
            [GarmentPcaLayer(gtype, 10 + self.imgEncoder.gar_latent_size, pca_dim) for gtype in self.garments])

        if infer_tex:
            self.texPcaparamLayers = nn.ModuleList(
                [TexturePcaLayer(gtype, 10 + self.imgEncoder.gar_latent_size, tex_pca_dim) for gtype in self.garments])

        self.detailDecoderLayers = nn.ModuleList([Generator(latent_dim=10 + 216 + self.n_detail, out_channels=1,
                                                            out_scale=displacement_scale, sample_mode='bilinear',
                                                            gtype=gtype) for gtype in self.garments]).to(device)

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

        self.garDisplacementLayers = nn.ModuleList(
            [GarmentDisplacementNet(256, self.imgEncoder.gar_latent_size, gtype, pca_folder, pca_dim)
             for gtype in self.garments])

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

        self.subdivider = SubdivideMeshes()

        self.dense_template_folder = dense_template_folder

        pi = np.pi
        self.constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))], device=self.device).float()

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
                cam_T_gt=None, Js_2d_gt=None, pca_verts_gt=None, tex_pca_perg_gt=None, img_masks=None, imgs_perg=None,
                **kwargs):
        batch_size = imgs.shape[0]
        cam_Rs, cam_Ts = None, None
        if not self.infer_camera:
            shapes, poses, trans, garlatents, lights, _ = self.imgEncoder(imgs)
        else:
            shapes, poses, trans, garlatents, cam_exts, lights, _ = self.imgEncoder(imgs)
        poses_pre = poses.clone()

        lights_per_gar = torch.cat((lights, lights), 1).view(-1, 9, 3)

        if self.use_detail:
            detail_latents = self.detailEncoder(imgs)

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

            tgtypes = torch.cat((up_gtypes, bottom_gtypes), dim=-1)
            # modify prob result by gt
            if gtypes is not None:
                for r, c in zip(rows, cols):
                    c = gtypes[r, c].item()
                    if c < self.upper_type_num:
                        tgtypes[r, 0:4] = 0
                        tgtypes[r, c] = 1
                    elif c < len(self.garments):
                        tgtypes[r, 4:] = 0
                        tgtypes[r, c] = 1
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

        garbatch, edge_index, face_index, vf_vindex, vf_findex = self.extract_tris_infos(tris_infos)
        imgbatch = imgBatchFromGarBatch(garbatch, gtypes)

        self.imgbatch = imgbatch
        Js, body_ns = self.skinDeformNet.skeleton(shapes, True)

        diss = (gps_pca.unsqueeze(1) - Js[imgbatch, :]).norm(dim=-1)

        if self.skinWsNet.use_normal:
            vnorms = compute_vnorms(gps_pca, face_index, vf_vindex, vf_findex)
            ws = self.skinWsNet(torch.cat((gps_pca, vnorms, diss), dim=-1), edge_index, garbatch)
        else:
            ws = self.skinWsNet(torch.cat((gps_pca, diss), dim=-1), edge_index, garbatch)

        deform_rec, transforms, pose_Rs, Js_transformed = self.skinDeformNet(gps_pca, Js, ws, poses, imgbatch)
        self.transforms = transforms
        deform_rect = deform_rec + trans[imgbatch, :]

        Js_transformed = Js_transformed + trans.unsqueeze(1)
        # project joints ----------------------------------------------------------------------------
        renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs, T=cam_Ts, lights=None, device=imgs.device)
        Js_2d = renderer.transform_points(Js_transformed)
        Js_2d = Js_2d[:, :, :2]

        # 上衣和裤子各对应一个RT及相机
        cam_Rs_per_gar = torch.cat((cam_Rs.view(-1, 1, 3 * 3), cam_Rs.view(-1, 1, 3 * 3)), 1).view(-1, 3, 3)
        cam_Ts_per_gar = torch.cat((cam_Ts.view(-1, 1, 3), cam_Ts.view(-1, 1, 3)), 1).view(-1, 3)
        renderer = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs_per_gar, T=cam_Ts_per_gar, lights=None,
                            device=imgs.device)
        uv_rasterizer = Pytorch3dRasterizer(self.tex_height)
        dense_triangles = []
        uv_masks_list = []
        for g in gtypes:
            inds = torch.nonzero(g)
            dense_triangles.append(self.texPcapsLayers[inds[0][0]].dense_triangles)
            dense_triangles.append(self.texPcapsLayers[inds[1][0]].dense_triangles)
            uv_masks_list.append(self.texPcapsLayers[inds[0][0]].uv_mask_erosion.unsqueeze(0).unsqueeze(0))
            uv_masks_list.append(self.texPcapsLayers[inds[1][0]].uv_mask_erosion.unsqueeze(0).unsqueeze(0))

        uv_masks = torch.cat(uv_masks_list, dim=0)

        max_tri_len = max([len(tri) for tri in dense_triangles])
        dense_faces_list = []
        for tri in dense_triangles:
            padded_tri = pad(tri, [0, 0, 0, max_tri_len - len(tri)], mode='constant', value=-1).unsqueeze(0)
            dense_faces_list.append(padded_tri)

        dense_faces = torch.cat(dense_faces_list, dim=0)

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

        if not self.use_detail:
            # sh light
            coarse_shading = add_SHlight(uv_coarse_normals, lights_per_gar, self.constant_factor)
            coarse_shading_images = coarse_shading
            coarse_textures = textures * coarse_shading_images
            texUVs._maps_padded = coarse_textures.permute(0, 2, 3, 1)
        rendered_coarse_imgs = renderer(coarse_meshes)

        # displacement map -> normal map
        displacement_maps = displacement_maps.reshape(-1, 1, self.tex_height, self.tex_width) * uv_masks
        if self.use_detail and not self.inferring:
            uv_detail_vertices = uv_coarse_vertices + displacement_maps * uv_coarse_normals
            dense_detail_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size * 2, -1, 3])
            uv_detail_normals = get_vertex_normals(dense_detail_vertices, dense_faces)
            uv_detail_normals = uv_detail_normals.reshape([batch_size * 2, self.tex_height, self.tex_width, 3]).permute(
                0, 3, 1, 2).contiguous()
            uv_detail_normals = uv_detail_normals * uv_coarse_normals_masks + uv_coarse_normals * (
                    1. - uv_coarse_normals_masks)

            # render detail meshes
            # sh
            detail_shading = add_SHlight(uv_detail_normals, lights_per_gar, self.constant_factor)
            detail_shading_images = detail_shading
            detail_textures = textures * detail_shading_images
            # detail without texture
            texUVs._maps_padded = (torch.ones_like(textures) * 0.9 * detail_shading_images).permute(0, 2, 3, 1)
            # detail with texture
            texUVs._maps_padded = detail_textures.permute(0, 2, 3, 1)
            rendered_detail_imgs = renderer(coarse_meshes)
            dis_packed = displacement_maps

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
        else:
            rendered_detail_imgs = None
            dis_packed = None
            detail_textures = None
            uv_vis_masks = None
            extracted_textures = None

        # save coarse mesh and detail mesh
        if self.inferring and self.mesh_save_folder is not None:
            from pytorch3d.io import save_obj
            if not self.use_detail:
                displacement_maps = torch.zeros(batch_size, 3, 256, 256)
            temp_displacement_maps = displacement_maps
            types = torch.where(gtypes == 1)[1]
            for i in range(len(types)):
                t = 'up' if i % 2 == 0 else 'bottom'
                obj_name = os.path.splitext(img_names[i // 2])[0]
                coarse_save_path = os.path.join(self.mesh_save_folder, obj_name + '_%s_coarse.obj' % t)
                save_obj(
                    coarse_save_path,
                    coarse_verts[i][:self.garmentvnums[types[i]]],
                    coarse_meshes[i].faces_packed(),
                    verts_uvs=self.texPcapsLayers[types[i]].verts_uvs,
                    faces_uvs=self.texPcapsLayers[types[i]].faces_uvs,
                    texture_map=torch.from_numpy(
                        textures[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1].copy()
                    )
                )
                dense_save_path = coarse_save_path.replace('_coarse.obj', '.obj')
                # if not os.path.exists(dense_save_path):
                if self.use_detail:
                    if self.upsample_dismap:
                        dense_mesh = subdivide_mesh_by_meshlab(coarse_save_path, iter=2, thres=0, uv_dis=
                        F.interpolate(temp_displacement_maps, (4096, 4096), mode='bilinear')[i],
                                                               use_neighbor=self.use_neighbor)
                    else:
                        dense_mesh = subdivide_mesh_by_meshlab(coarse_save_path, iter=2, thres=0,
                                                               uv_dis=temp_displacement_maps[i],
                                                               use_neighbor=self.use_neighbor)
                    save_obj(dense_save_path, torch.as_tensor(dense_mesh['v']), torch.as_tensor(dense_mesh['f']),
                             verts_uvs=self.garPcapsLayers[types[i]].dense_vt,
                             faces_uvs=self.garPcapsLayers[types[i]].dense_faces_uvs,
                             texture_map=torch.as_tensor(
                                 textures[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1].copy(),
                                 device=self.device))

        if not self.use_detail:
            detail_shading_images = None
        if not self.inferring:
            return garbatch, gps_pca, shapes, poses, trans, pcas_perg, Js_transformed, up_gar_prob, bottom_gar_prob, cam_Rs, cam_Ts, tex_pcas_perg, Js_2d, rendered_coarse_imgs, rendered_detail_imgs, dis_packed, detail_textures, extracted_textures, uv_vis_masks, lights, detail_shading_images
        else:
            return up_gar_prob, bottom_gar_prob, cam_Rs, cam_Ts, displacement_maps