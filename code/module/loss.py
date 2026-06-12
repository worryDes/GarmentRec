import numpy as np
import torch
import torch.nn
from loguru import logger
from pytorch3d.loss.chamfer import knn_points
from pytorch3d.structures import Meshes
import cv2
from pytorch3d.loss import (mesh_laplacian_smoothing, mesh_normal_consistency)

L1Loss = torch.nn.L1Loss()
MSELoss = torch.nn.MSELoss()
CrossEntropyLoss = torch.nn.CrossEntropyLoss()
ReLU = torch.nn.ReLU(inplace=True)

eps = 1e-5

def L1(x_pred, x_gt):
    if (x_pred == None or x_gt == None or len(x_pred) == 0 or len(x_gt) == 0):
        return torch.tensor(0.0, device=x_pred.device)
    return L1Loss(x_pred, x_gt)

def L2(x_pred, x_gt):
    if (x_pred == None or x_gt == None or len(x_pred) == 0 or len(x_gt) == 0):
        return torch.tensor(0.0, device=x_pred.device)
    return MSELoss(x_pred, x_gt)

def L1_Reg(x_pred):
    if (x_pred == None):
        return 0
    return torch.mean(torch.abs(x_pred))

def L2_Reg(x_pred):
    if (x_pred == None):
        return 0
    return torch.mean(x_pred ** 2)

def L_int(P, Q, Nq):
    """interp loss

    Args:
        P (torch.Tensor): verts of P, N * 3
        Q (torch.Tensor): verts of Q, N * 3
        Nq (torch.Tensor): normals of Q ordered by pairs, N * 3
    """
    return torch.sum(ReLU(-Nq * (P - Q))) / P.shape[0]
    pass
            
def weighted_loss(loss, weights, key, return_dict=True, prefix='_w'):
    loss_w = loss * weights[key]
    if not return_dict:
        return loss_w
    d = dict()
    d[key] = loss
    d[key + prefix] = loss_w
    return loss_w, d
    

def calc_shape_params_loss(shapes_pred, poses_pred, trans_pred, alpha_pred, 
                           shapes_gt, poses_gt, trans_gt, alpha_gt, 
                           weights):
    d = dict()
    L_params = 0.0
    if weights['shapes'] > 0:
        L_shapes = L1(shapes_pred, shapes_gt)
        L_shapes_w, t = weighted_loss(L_shapes, weights, 'shapes')
        d.update(t)
        L_params += L_shapes_w
    if weights['poses'] > 0:
        L_poses = L1(poses_pred, poses_gt)
        L_poses_w, t = weighted_loss(L_poses, weights, 'poses')
        d.update(t)
        L_params += L_poses_w
    if weights['trans'] > 0:
        L_trans= L1(trans_pred, trans_gt)
        L_trans_w, t = weighted_loss(L_trans, weights, 'trans')
        d.update(t)
        L_params += L_trans_w
    if weights['alpha'] > 0:
        # 排除没有pca参数的部分
        is_zero_row = (alpha_gt == 0).all(dim=1)
        # 使用 torch.where 替换全零行
        alpha_gt = torch.where(is_zero_row.unsqueeze(1), alpha_pred, alpha_gt)
        L_alpha = L1(alpha_pred, alpha_gt)
        L_alpha_w, t = weighted_loss(L_alpha, weights, 'alpha')
        d.update(t)
        L_params += L_alpha_w
    
    L_params_w, t = weighted_loss(L_params, weights, 'all')
    d.update(t)
    
    return L_params_w, d


def calc_classify_loss(upper_type_pred, lower_type_pred, 
                       upper_type_gt, lower_type_gt, 
                       weights):
    L_classify = 0.0
    d = dict()
    if weights['upper_type'] > 0:
        L_upper_type = CrossEntropyLoss(upper_type_pred, upper_type_gt)
        L_upper_type_w, t = weighted_loss(L_upper_type, weights, 'upper_type')
        d.update(t)
        L_classify += L_upper_type_w
    if weights['lower_type'] > 0:
        L_lower_type = CrossEntropyLoss(lower_type_pred, lower_type_gt)
        L_lower_type_w, t = weighted_loss(L_lower_type, weights, 'lower_type')
        d.update(t)
        L_classify += L_lower_type_w

    L_classify_w, t = weighted_loss(L_classify, weights, 'all')
    d.update(t)
    
    return L_classify_w, d


def calc_camera_loss(cam_R_pred, cam_R_gt, cam_T_pred, cam_T_gt, weights):
    """
        计算相机参数loss (相机为渲染图片的相机)
    Args:
        cam_R_pred (_type_): 预测的相机R, 1*9 or 1*3*3
        cam_R_gt (_type_): 相机R gt
        cam_T_pred (_type_): 预测的相机T, 1*3
        cam_T_gt (_type_): 相机T gt
        weights: 权重

    Returns:
        L_camera_w: weighted loss
        d: dict of sub losses
    """
    L_camera = 0.0
    d = dict()
    if weights['R'] > 0:
        L_R = L1(cam_R_pred.view(-1, 9), cam_R_gt.view(-1, 9))
        L_R_w, t = weighted_loss(L_R, weights, 'R')
        d.update(t)
        L_camera += L_R_w
    if weights['T'] > 0:
        L_T = L1(cam_T_pred, cam_T_gt)
        L_T_w, t = weighted_loss(L_T, weights, 'T')
        d.update(t)
        L_camera += L_T_w

    L_camera_w, t = weighted_loss(L_camera, weights, 'all')
    d.update(t)
    
    return L_camera_w, d


def calc_texture_loss(tex_pca_perg_pred, tex_pca_perg_gt, weights):
    L_tex = 0.0
    d = dict()
    if weights['pca_params'] > 0:
        is_zero_row = (tex_pca_perg_pred == 0).all(dim=1)
        # 使用 torch.where 替换全零行
        tex_pca_perg_gt = torch.where(is_zero_row.unsqueeze(1), tex_pca_perg_pred, tex_pca_perg_gt)
        L_tex_pca = L1(tex_pca_perg_pred, tex_pca_perg_gt)
        L_tex_pca_w, t = weighted_loss(L_tex_pca, weights, 'pca_params')
        d.update(t)
        L_tex += L_tex_pca_w
    L_tex_w, t = weighted_loss(L_tex, weights, 'all')
    d.update(t)
    
    return L_tex_w, d


def calc_landmark_loss(Js_pred, Js_gt, Js_2d_pred, Js_2d_gt, weights):
    L_landmark = 0.0
    d = dict()
    if weights['J_3d'] > 0:
        L_j_3d = L1(Js_pred.reshape(-1, 24 * 3), Js_gt.reshape(-1, 24 * 3))
        L_j_3d_w, t = weighted_loss(L_j_3d, weights, 'J_3d')
        d.update(t)
        L_landmark += L_j_3d_w
    if weights['J_2d'] > 0:
        L_j_2d = L1(Js_2d_pred.reshape(-1, 24 * 2), Js_2d_gt.reshape(-1, 24 * 2))
        L_j_2d_w, t = weighted_loss(L_j_2d, weights, 'J_2d')
        d.update(t)
        L_landmark += L_j_2d_w
        
    L_landmark_w, t = weighted_loss(L_landmark, weights, 'all')
    d.update(t)

    return L_landmark_w, d


def calc_photometric_loss(rendered_coarse_imgs, rendered_detail_imgs, imgs_gt, masks, weights):
    
    L_photometric = 0.0
    d = dict()
    
    masked_imgs_gt = masks * imgs_gt
    
    if weights['coarse'] > 0:
        masked_rendered_coarse_imgs = masks * rendered_coarse_imgs
        L_pho_coarse = L1(masked_rendered_coarse_imgs, masked_imgs_gt)
        L_pho_coarse_w, t = weighted_loss(L_pho_coarse, weights, 'coarse')
        d.update(t)
        L_photometric += L_pho_coarse_w
    if weights['detail'] > 0:
        masked_rendered_detail_imgs = masks * rendered_detail_imgs
        L_pho_detail = L1(masked_rendered_detail_imgs, masked_imgs_gt)
        L_pho_detail_w, t = weighted_loss(L_pho_detail, weights, 'detail')
        d.update(t)
        L_photometric += L_pho_detail_w

    L_photometric_w, t = weighted_loss(L_photometric, weights, 'all')
    d.update(t)

    return L_photometric_w, d


def calc_geo_loss(pca_verts_pred, gar_verts_pred, body_joints_pred, dis_pred, lap_dis_pred, 
                  pca_verts_gt, gar_verts_gt, body_joints_gt, dis_gt, lap_dis_gt, 
                  weights):
    
    L_geo = 0.0
    d = dict()
    
    if weights['pca_verts'] > 0:
        L_pca_verts = L2(pca_verts_pred, pca_verts_gt)
        L_pca_verts_w, t = weighted_loss(L_pca_verts, weights, 'pca_verts')
        d.update(t)
        L_geo += L_pca_verts_w
    if weights['garment_verts'] > 0:
        L_gar_verts = L2(gar_verts_pred, gar_verts_gt)
        L_gar_verts_w, t = weighted_loss(L_gar_verts, weights, 'garment_verts')
        d.update(t)
        L_geo += L_gar_verts_w
    if weights['body_joints'] > 0:
        L_body_joints = L2(body_joints_pred, body_joints_gt)
        L_body_joints_w, t = weighted_loss(L_body_joints, weights, 'body_joints')
        d.update(t)
        L_geo += L_body_joints_w
    if weights['displacement'] > 0: 
        L_dis = L1(dis_pred, dis_gt)
        L_dis_w, t = weighted_loss(L_dis, weights, 'displacement')
        d.update(t)
        L_geo += L_dis_w
    if weights['displacement_laplacian'] > 0:
        L_dis_lap = L2(lap_dis_pred, lap_dis_gt)
        L_dis_lap_w, t = weighted_loss(L_dis_lap, weights, 'displacement_laplacian')
        d.update(t)
        L_geo += L_dis_lap_w
    
    L_geo_w, t = weighted_loss(L_geo, weights, 'all')
    d.update(t)

    return L_geo_w, d


def calc_proj_loss(proj_gar_pred, proj_body_pred, 
                   proj_gar_gt, proj_body_gt, 
                   weights):
    L_proj = 0.0
    d = dict()
    
    if weights['proj_gar'] > 0:
        L_proj_gar = L2(proj_gar_pred, proj_gar_gt)
        L_proj_gar_w, t = weighted_loss(L_proj_gar, weights, 'proj_gar')
        d.update(t)
        L_proj += L_proj_gar_w
    if weights['proj_body'] > 0:
        L_proj_body = L2(proj_body_pred, proj_body_gt)
        L_proj_body_w, t = weighted_loss(L_proj_body, weights, 'proj_body')
        d.update(t)
        L_proj += L_proj_body_w
    
    L_proj_w, t = weighted_loss(L_proj, weights, 'all')
    d.update(t)
    
    return L_proj_w, d


def calc_interp_loss(tpose_gar_pred, tpose_body_pred, gar_pred, body_pred, 
                     garbatch, body_faces, batch_size, weights):
    L_interp = 0.0
    # TODO: d
    d = dict()
    
    # for each subject
    for i in range(batch_size):
        # 1. get pairs
        # tpose garment and body
        tg = torch.unsqueeze(torch.cat((tpose_gar_pred[garbatch == i * 2], tpose_gar_pred[garbatch == i * 2 + 1]), dim=0), dim=0)
        tb = torch.unsqueeze(tpose_body_pred[i], dim=0)
        tpairs = torch.squeeze(knn_points(tg, tb, K=1).idx[0])
        # posed garment and body
        mg = torch.unsqueeze(torch.cat((gar_pred[garbatch == i * 2], gar_pred[garbatch == i * 2 + 1]), dim=0), dim=0)
        mb = torch.unsqueeze(body_pred[i], dim=0)
        mpairs = torch.squeeze(knn_points(mg, mb, K=1).idx[0])
        
        # 2. get normals
        N_tb = Meshes(verts=[tpose_body_pred[i]], faces=[body_faces])[0].verts_normals_packed()
        N_mb = Meshes(verts=[body_pred[i]], faces=[body_faces])[0].verts_normals_packed()
        
        # 3. reorder verts and normals of body by pairs
        tg = torch.squeeze(tg)
        mg = torch.squeeze(mg)
        tb = torch.squeeze(tb)[tpairs]
        mb = torch.squeeze(mb)[tpairs]
        N_tb = N_tb[tpairs]
        N_mb = N_mb[mpairs]
        
        # 4. calc loss
        L_interp += L_int(tg, tb, N_tb) + L_int(mg, mb, N_mb)
        
    L_interp_w, t = weighted_loss(L_interp, weights, 'all')
    d.update(t)
        
    return L_interp_w, d


def calc_midline_loss(midpair_lverts, midpair_rverts, weights):
    d = dict()
    L_midline = L1(midpair_lverts, midpair_rverts)
    L_midline_w, t = weighted_loss(L_midline, weights, 'all')
    d.update(t)
    return L_midline_w, d


def skinng_weight_loss():
    pass


def calc_boundary_loss(boundaries_pred, boundaries_gt, weights):
    """
        计算投影边缘损失
    """
    d = dict()
    L_boundary = L2(boundaries_pred, boundaries_gt)
    L_boundary_w, t = weighted_loss(L_boundary, weights, 'all')
    d.update(t)
    return L_boundary_w, d


def calc_mrf_loss(detail_textures, extracted_textures, mrf_fn, weights):
    d = dict()
    # TODO: why extracted_textures have grad fn ?
    L_mrf = mrf_fn(detail_textures, extracted_textures) / (detail_textures.shape[0] // 2)
    L_mrf_w, t = weighted_loss(L_mrf, weights, 'all')
    d.update(t)
    return L_mrf_w, d


def calc_reg_loss(shapes, poses, trans, R, T, tex, lights, dis, weights):
    L_reg = 0.0
    d = dict()
    if weights['shape'] > 0:
        L_reg_shape = L1_Reg(shapes)
        L_reg_shape_w, t = weighted_loss(L_reg_shape, weights, 'shape')
        d.update(t)
        L_reg += L_reg_shape_w
    if weights['pose'] > 0:
        L_reg_pose = L1_Reg(poses)
        L_reg_pose_w, t = weighted_loss(L_reg_pose, weights, 'pose')
        d.update(t)
        L_reg += L_reg_pose_w
    if weights['trans'] > 0:
        L_reg_trans = L1_Reg(trans)
        L_reg_trans_w, t = weighted_loss(L_reg_trans, weights, 'trans')
        d.update(t)
        L_reg += L_reg_trans_w
    if weights['R'] > 0:
        L_reg_R = L1_Reg(R)
        L_reg_R_w, t = weighted_loss(L_reg_R, weights, 'R')
        d.update(t)
        L_reg += L_reg_R_w
    if weights['T'] > 0:
        L_reg_T = L1_Reg(T)
        L_reg_T_w, t = weighted_loss(L_reg_T, weights, 'T')
        d.update(t)
        L_reg += L_reg_T_w
    if weights['tex'] > 0:
        L_reg_tex = L1_Reg(tex)
        L_reg_tex_w, t = weighted_loss(L_reg_tex, weights, 'tex')
        d.update(t)
        L_reg += L_reg_tex_w
    if weights['light'] > 0:
        L_reg_light = L1_Reg(lights)
        L_reg_light_w, t = weighted_loss(L_reg_light, weights, 'light')
        d.update(t)
        L_reg += L_reg_light_w
    if weights['dis'] > 0:
        L_reg_dis = L1_Reg(dis)
        L_reg_dis_w, t = weighted_loss(L_reg_dis, weights, 'dis')
        d.update(t)
        L_reg += L_reg_dis_w
        
    L_reg_w, t = weighted_loss(L_reg, weights, 'all')
    d.update(t)
    
    return L_reg_w, d


def calc_shading_smooth_loss(shading, weights):
    '''
    assume: shading should be smooth
    ref: Lifting AutoEncoders: Unsupervised Learning of a Fully-Disentangled 3D Morphable Model using Deep Non-Rigid Structure from Motion
    '''
    L_shading = 0.0
    d = dict()
    dx = shading[:,:,1:-1,1:] - shading[:,:,1:-1,:-1]
    dy = shading[:,:,1:,1:-1] - shading[:,:,:-1,1:-1]
    gradient_image = (dx**2).mean() + (dy**2).mean()
    L_shading = gradient_image.mean()
    L_shading_w, t = weighted_loss(L_shading, weights, 'all')
    d.update(t)
    return L_shading_w, d

def bidirectional_nearest_k_loss(pred_pixels, gt_pixels, k=10):
    """
    计算轮廓像素点与GT轮廓像素点的双向最近邻损失

    参数：
        pred_pixels: (Np, 2) tensor，预测轮廓点像素坐标，requires_grad=True
        gt_pixels: (Ng, 2) tensor，GT轮廓点像素坐标
        k: int，每个点考虑最近的k个点

    返回：
        loss: scalar tensor
    """
    Np = pred_pixels.shape[0]
    Ng = gt_pixels.shape[0]

    # 1. 预测轮廓到 GT 最近 k 个点
    # diff_pred_to_gt = pred_pixels[:, None, :] - gt_pixels[None, :, :]  # (Np, Ng, 2)
    # dist2_pred_to_gt = (diff_pred_to_gt ** 2).sum(dim=-1)             # (Np, Ng)
    # topk_dist2_pred_to_gt, _ = torch.topk(dist2_pred_to_gt, k, dim=1, largest=False)
    # loss_pred_to_gt = topk_dist2_pred_to_gt.mean()

    # 2. GT 到预测轮廓最近 k 个点
    k = min(k, Np)
    diff_gt_to_pred = gt_pixels[:, None, :] - pred_pixels[None, :, :]  # (Ng, Np, 2)
    dist2_gt_to_pred = (diff_gt_to_pred ** 2).sum(dim=-1)              # (Ng, Np)
    topk_dist2_gt_to_pred, _ = torch.topk(dist2_gt_to_pred, k, dim=1, largest=False)
    loss_gt_to_pred = topk_dist2_gt_to_pred.float().mean()

    # 3. 双向损失
    loss = loss_gt_to_pred
    return loss

def calc_boundary_loss(pixels_pred, pixels_gt, weights):
    n = len(pixels_pred)
    L_boundary = 0.0
    count = 0
    for i in range(n):
        l = bidirectional_nearest_k_loss(pixels_pred[i], pixels_gt[i])
        if not torch.isnan(l):
            L_boundary += l
            count += 1
    if (count == 0):
        return 0.0
    else:
        return L_boundary / count * weights['all']

def calc_smooth_loss(meshes, weights):
    L_smooth = 0.0
    if weights['laplacian'] > 0:
        L_laplacian = mesh_laplacian_smoothing(meshes) * weights['laplacian']
        L_smooth += L_laplacian
    if weights['normal'] > 0:
        L_normal = mesh_normal_consistency(meshes) * weights['normal']
        L_smooth += L_normal
    L_smooth_w = L_smooth * weights['all']
    return L_smooth_w

    



