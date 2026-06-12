import os

os.sys.path.append(os.getcwd())
os.sys.path.append(os.path.join(os.getcwd(), 'module'))

import torch
import os.path as osp
import numpy as np
import torch
import pickle
from module.SkinWeightModel import SkinWeightNet
import module.ImageReconstructModel as M
import os
from glob import glob
import argparse
import cv2
from utils import *
from pytorch3d.structures import Meshes
from config.config import *
from tqdm import tqdm
from PIL import Image
from renderer import *
from pytorch3d.io import load_obj, save_obj
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='img rec comparing')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--displacement_scale', type=float, default=0.005)
    parser.add_argument('--upsample_dismap', type=int, default=0)
    parser.add_argument('--use_neighbor', type=int, default=1)
    parser.add_argument('--use_detail', type=int, default=1)
    parser.add_argument('--normal_refine', type=int, default=0)
    parser.add_argument('--gpu', default=0, type=int, metavar='ID')
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--smpl_model_path', type=str, default="./smpl_pytorch/model/neutral_smpl_with_cocoplus_reg.txt")
    parser.add_argument('--midpair_path', type=str, default="./data/midpairs.pkl")
    parser.add_argument('--dense_midpair_path', type=str, default="./data/dense_midpairs.pkl")
    parser.add_argument('--dense_template_folder', type=str, default="./data")
    args = parser.parse_args()

    smpl_model_path = args.smpl_model_path
    midpair_path = args.midpair_path
    dense_midpair_path = args.dense_midpair_path
    dense_template_folder = args.dense_template_folder
    displacement_scale = args.displacement_scale

    if args.gpu == None:
        device = torch.device('cpu')
    else:
        device = torch.device(args.gpu)

    model_path = args.model_path
    upsample_dismap = True if args.upsample_dismap == 1 else False
    use_neighbor = True if args.use_neighbor == 1 else False
    use_detail = True if args.use_detail == 1 else False
    normal_refine = True if args.normal_refine == 1 else False
    save_folder = args.output_folder
    image_folder = args.input_folder
    os.makedirs(save_folder, exist_ok=True)

    logger.info("infer option: ")
    logger.info("model: " + model_path)

    bcnet_tran_mean = [-1.0962e-02, 2.8778e-01, 1.2973e+01]
    tran_mean = [0.0, 0.0, 0.0]

    # projection camera
    cam_k = torch.Tensor([[3.0375e+03, 0.0000e+00, 2.7000e+02],
                          [0.0000e+00, 3.0375e+03, 2.7000e+02],
                          [0.0000e+00, 0.0000e+00, 1.0000e+00]])
    cam_Rt = torch.tensor([[1, 0, 0, tran_mean[0] - bcnet_tran_mean[0]],
                           [0, 1, 0, tran_mean[1] - bcnet_tran_mean[1]],
                           [0, 0, 1, tran_mean[2] - bcnet_tran_mean[2]]])
    cam_k = cam_k.matmul(cam_Rt)
    cam_k = cam_k.to(device)

    img_names = [
        x for x in os.listdir(image_folder)
        if (x.endswith('.jpg') or x.endswith('.png'))
           and 'mask' not in x
           and 'normal' not in x
    ]
    img_files = []
    for img in img_names:
        img_files.append(cv2.imread(os.path.join(image_folder, img)))

    if len(img_files) == 0:
        print('zeros img files, exit.')
        exit()

    skinWsNet = SkinWeightNet(4, True)

    # D3G model set
    garments = ['T-shirt', 'front_open_T-shirt', 'Shirt', 'front_open_Shirt', 'Shorts', 'Pants']
    garmentvnums = [1954, 1954, 2468, 2468, 678, 1180]
    upper_type_num = 4
    pca_folder = '/data/gz/D3G/Datasets/D3G_and_SIZER/tmps'
    pca_dim = 64
    gar_vnums_gt = [1954, 1954, 2468, 2468, 678, 1180]

    # load midpairs
    if not os.path.exists(midpair_path):
        raise FileNotFoundError
    with open(midpair_path, 'rb') as f:
        midpairs = pickle.load(f)

    if not os.path.exists(dense_midpair_path):
        raise FileNotFoundError
    with open(dense_midpair_path, 'rb') as f:
        dense_midpairs = pickle.load(f)

    fcl_x = 4500
    fcl_y = 4500
    img_w = 540
    img_h = 540
    rx, ry, rz = 0, 0, 0
    tx = tran_mean[0] - bcnet_tran_mean[0]
    ty = tran_mean[1] - bcnet_tran_mean[1]
    tz = tran_mean[2] - bcnet_tran_mean[2]

    mesh_save_folder = os.path.join(save_folder, 'mesh')
    vis_save_folder = os.path.join(save_folder, 'vis')
    if not os.path.exists(mesh_save_folder):
        os.mkdir(mesh_save_folder)
    if not os.path.exists(vis_save_folder):
        os.mkdir(vis_save_folder)
    print('total %d imgfiles' % len(img_files))

    net = M.ImageReconstructModel(skinWsNet,
                                  with_classification=True,
                                  tran_mean=tran_mean,
                                  garments=garments,
                                  garmentvnums=garmentvnums,
                                  upper_type_num=upper_type_num,
                                  pca_folder=pca_folder,
                                  pca_dim=pca_dim,
                                  smpl_model_path=smpl_model_path,
                                  midpairs=midpairs,
                                  infer_camera=True,
                                  infer_tex=True,
                                  inferring=True,
                                  use_detail=use_detail,
                                  vis_save_folder=vis_save_folder,
                                  mesh_save_folder=mesh_save_folder,
                                  dense_template_folder=dense_template_folder,
                                  displacement_scale=displacement_scale,
                                  upsample_dismap=upsample_dismap,
                                  use_neighbor=use_neighbor,
                                  device=device)

    state_dict = torch.load(model_path, map_location=torch.device(device))
    net.load_state_dict(state_dict, strict=False)

    net = net.to(device)
    net.eval()

    data = {}
    for batch_idx, [names, imgs] in tqdm(enumerate(zip(img_names, img_files))):
        # all to gpu
        imgs = cv2.resize(imgs, (540, 540))
        imgs = imgs / 255.
        imgs = torch.tensor(imgs, dtype=torch.float32).to(device)
        imgs = imgs.permute(2, 0, 1)
        imgs = imgs.unsqueeze(0)
        names = np.array([names])
        imgs_perg = torch.cat((imgs, imgs), 1).reshape(-1, 3, 540, 540)
        input_gtypes = np.array([[-1, -1]])

        up_gar_prob, bottom_gar_prob, cam_Rs, cam_Ts, displacement_maps = net(imgs, names, gtypes=input_gtypes, cam_k=cam_k, imgs_perg=imgs_perg)

        # 后处理
        up_index = up_gar_prob.argmax(dim=1).item()
        bottom_index = bottom_gar_prob.argmax(dim=1).item()
        predicted_up = garments[up_index]
        predicted_bottom = garments[bottom_index + 4]

        tgt_name = os.path.splitext(names[0])[0]
        up_mesh_path = os.path.join(mesh_save_folder, tgt_name + '_up.obj')
        bottom_mesh_path = os.path.join(mesh_save_folder, tgt_name + '_bottom.obj')
        up_verts, up_faces, _ = load_obj(up_mesh_path)

        up_texture = Image.open(up_mesh_path[:-3] + 'png').convert("RGB")
        up_texture = torch.from_numpy(np.array(up_texture)).float() / 255.0
        bottom_texture = Image.open(bottom_mesh_path[:-3] + 'png').convert("RGB")
        bottom_texture = torch.from_numpy(np.array(bottom_texture)).float() / 255.0

        up_verts_uvs = net.garPcapsLayers[up_index].dense_vt.cpu().clone()
        up_faces_uvs = net.garPcapsLayers[up_index].dense_faces_uvs.cpu().clone()
        bottom_verts_uvs = net.garPcapsLayers[bottom_index].dense_vt.cpu()
        bottom_faces_uvs = net.garPcapsLayers[bottom_index].dense_faces_uvs.cpu()

        if predicted_up in ['Shirt', 'T-shirt']:
            pairs = dense_midpairs[predicted_up]
            lverts_index, rverts_index = pairs[..., 0], pairs[..., 1]
            up_verts, up_faces, up_verts_uvs, up_faces_uvs = stitch_seam(up_verts, up_faces.verts_idx, up_verts_uvs, up_faces_uvs, lverts_index, rverts_index)

            up_texture = Image.open(up_mesh_path[:-3] + 'png').convert("RGB")
            up_texture = torch.from_numpy(np.array(up_texture)).float() / 255.0
            save_obj(up_mesh_path, up_verts, up_faces, verts_uvs=up_verts_uvs, faces_uvs=up_faces_uvs, texture_map=up_texture)

        if normal_refine:
            up_mask_path = os.path.join(image_folder, tgt_name + '_mask_up.png')
            bottom_mask_path = os.path.join(image_folder, tgt_name + '_mask_bottom.png')
            normal_path = os.path.join(image_folder, tgt_name + '_normal.png')
            if not os.path.exists(up_mask_path) or not os.path.exists(bottom_mask_path) or not os.path.exists(normal_path):
                print(f"[{tgt_name}] files not complete.")
                continue

            up_mask_cv = cv2.resize(cv2.imread(up_mask_path, cv2.IMREAD_GRAYSCALE), (540, 540))
            up_mask = torch.tensor(up_mask_cv, device=device).unsqueeze(0).unsqueeze(0) / 255.0
            bottom_mask_cv = cv2.resize(cv2.imread(bottom_mask_path, cv2.IMREAD_GRAYSCALE), (540, 540))
            bottom_mask = torch.tensor(bottom_mask_cv, device=device).unsqueeze(0).unsqueeze(0) / 255.0

            normal_cv = cv2.resize(cv2.cvtColor(cv2.imread(normal_path), cv2.COLOR_BGR2RGB), (540, 540)) / 255.0
            normal = torch.from_numpy(normal_cv).permute(2, 0, 1).unsqueeze(0).float().to(device)
            up_normal = normal * up_mask
            bottom_normal = normal * bottom_mask

            render = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_Rs, T=cam_Ts, lights=None, device=device)

            # --- 上衣网格初始化 ---
            up_verts, up_faces, _ = load_obj(up_mesh_path)
            up_verts = up_verts.to(device)
            up_faces = up_faces.verts_idx.to(device)
            up_meshes = Meshes(verts=[up_verts], faces=[up_faces])

            # --- 下衣网格初始化 ---
            bottom_verts, bottom_faces, _ = load_obj(bottom_mesh_path)
            bottom_verts = bottom_verts.to(device)
            bottom_faces = bottom_faces.verts_idx.to(device)
            bottom_meshes = Meshes(verts=[bottom_verts], faces=[bottom_faces])

            up_local_affine = LocalAffine(
                up_meshes.verts_padded().shape[1],
                up_meshes.verts_padded().shape[0],
                up_meshes.edges_packed()
            ).to(device)

            bottom_local_affine = LocalAffine(
                bottom_meshes.verts_padded().shape[1],
                bottom_meshes.verts_padded().shape[0],
                bottom_meshes.edges_packed()
            ).to(device)

            optimizer_cloth = torch.optim.Adam([
                {'params': up_local_affine.parameters(), 'lr': 1e-4},
                {'params': bottom_local_affine.parameters(), 'lr': 1e-4}
            ], amsgrad=True)

            scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_cloth, mode="min", factor=0.1, verbose=0, min_lr=1e-5, patience=5
            )

            loop_cloth = tqdm(range(200), dynamic_ncols=True)
            for i in loop_cloth:
                optimizer_cloth.zero_grad()

                up_deformed_verts, up_stiffness, up_rigid = up_local_affine(up_verts.unsqueeze(0), return_stiff=True)
                up_meshes = up_meshes.update_padded(up_deformed_verts)

                up_colors = compute_normal_color_in_camera(up_meshes, cam_Rs[0])
                up_meshes.textures = TexturesVertex(verts_features=[up_colors])

                pred_up_normal = render.render_normal(up_meshes)[..., :3].permute(0, 3, 1, 2) * up_mask

                # 计算上衣各项 Loss
                loss_normal_up = ((pred_up_normal - up_normal) ** 2).mean() * 1e0
                loss_mesh_up = mesh_normal_consistency(up_meshes) * 1e-2

                bottom_deformed_verts, bottom_stiffness, bottom_rigid = bottom_local_affine(bottom_verts.unsqueeze(0), return_stiff=True)
                bottom_meshes = bottom_meshes.update_padded(bottom_deformed_verts)

                bottom_colors = compute_normal_color_in_camera(bottom_meshes, cam_Rs[0])
                bottom_meshes.textures = TexturesVertex(verts_features=[bottom_colors])

                pred_bottom_normal = render.render_normal(bottom_meshes)[..., :3].permute(0, 3, 1, 2) * bottom_mask

                # 计算下衣各项 Loss
                loss_normal_bottom = ((pred_bottom_normal - bottom_normal) ** 2).mean() * 1e0
                loss_mesh_bottom = mesh_normal_consistency(bottom_meshes) * 1e-2

                total_loss = loss_normal_up + loss_normal_bottom + loss_mesh_up + loss_mesh_bottom

                total_loss.backward(retain_graph=True)
                optimizer_cloth.step()
                scheduler_cloth.step(total_loss.item())

                # 更新进度条
                loop_cloth.set_postfix_str(
                    f"loss={total_loss.item():.6f}"
                )

            up_final_path = os.path.join(mesh_save_folder, tgt_name + '_final_up.obj')
            bottom_final_path = os.path.join(mesh_save_folder, tgt_name + '_final_bottom.obj')
            save_obj(up_final_path, up_deformed_verts[0].detach().cpu(), up_faces, )
            save_obj(bottom_final_path, bottom_deformed_verts[0].detach().cpu(), bottom_faces)

    logger.info('infer done.')

