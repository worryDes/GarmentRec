import os
import sys
import pickle
import argparse
import re
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'module'))

from module.SkinWeightModel import SkinWeightNet
import module.ImageReconstructModel as M
from module.loss import *
from module.basic import Sobel, IDMRFLoss
from load_data import get_dataloader
from utils import *
from config.config import *
from loguru import logger
from renderer import Renderer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='train_config2', type=str)
    args = parser.parse_args()

    # 加载配置
    config, _ = load_config(args.config)
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    num_workers = config['num_workers']
    device = config['device']
    epoch_num = config['epoch_num']
    learning_rate = config['learning_rate']
    smpl_model_path = config['smpl_model_path']
    bcnet_tran_mean = config['bcnet_tran_mean']
    log_path = config['log_path']
    garments = config['garments']
    garmentvnums = config['garmentvnums']
    upper_type_num = config['upper_type_num']
    tran_mean = config['tran_mean']
    model_name = config['model_name']
    model_folder = config['model_folder']
    model_path = config['model_path']
    img_folder = config['img_folder']
    mask_folder = config['mask_folder']
    pca_folder = config['pca_folder']
    pca_dim = config['pca_dim']
    tex_pca_dim = config['tex_pca_dim']
    tex_width = config['tex_width']
    tex_height = config['tex_height']
    train_item_path = config['train_item_path']
    val_item_path = config['val_item_path']
    trunc = config['trunc']
    save_title = config['save_title']
    freeze = config['freeze']
    lap_path = config['lap_path']
    pre_epoch = config['pre_epoch']
    pre_batch = config['pre_batch']
    weights = config['weights']['freeze_%s' % freeze]
    saving_model = config['save_model']
    train_garment_types = config['train_garment_types']
    midpair_path = config['midpair_path']
    train_exclude_real = config['train_exclude_real']
    train_exclude_standard_side = config['train_exclude_standard_side']
    add_val_in_train = config['add_val_in_train']
    infer_camera = config['infer_camera']
    infer_tex = config['infer_tex']
    create_detail_meshes = config['create_detail_meshes']
    subdivide_template_folder = config['subdivide_template_folder']
    subdivide_template_name = config['subdivide_template_name']
    dense_template_folder = config['dense_template_folder']
    light_instance_scale = config['light_instance_scale']
    displacement_scale = config['displacement_scale']

    print('--- Train Session Started ---')
    print(f'Model: {model_name} | LR: {learning_rate} | Batch Size: {batch_size} | Freeze: {freeze}')
    logger.add(sink=log_path, encoding='utf-8')

    # 计算相机投影矩阵
    cam_k = torch.as_tensor([[3.0375e+03, 0.0000e+00, 2.7000e+02],
                             [0.0000e+00, 3.0375e+03, 2.7000e+02],
                             [0.0000e+00, 0.0000e+00, 1.0000e+00]])
    cam_Rt = torch.as_tensor([[1, 0, 0, tran_mean[0] - bcnet_tran_mean[0]],
                              [0, 1, 0, tran_mean[1] - bcnet_tran_mean[1]],
                              [0, 0, 1, tran_mean[2] - bcnet_tran_mean[2]]])
    cam_k = cam_k.matmul(cam_Rt).to(device)

    # 数据集加载
    with open(train_item_path, 'rb') as f:
        train_items = pickle.load(f, encoding='latin1')
    with open(val_item_path, 'rb') as f:
        val_items = pickle.load(f, encoding='latin1')

    if add_val_in_train:
        train_items = {**train_items, **val_items}

    if not os.path.exists(midpair_path):
        raise FileNotFoundError
    with open(midpair_path, 'rb') as f:
        midpairs = pickle.load(f)

    train_data_loader = get_dataloader('train', img_folder, mask_folder, train_items,
                                       batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       train_garment_types=train_garment_types,
                                       exclude_real=train_exclude_real,
                                       exclude_standard_side=train_exclude_standard_side)

    with open(lap_path, 'rb') as f:
        lap_matrices = pickle.load(f)

    # 搭建网络
    skinWsNet = SkinWeightNet(4, True)
    use_detail = True
    net = M.ImageReconstructModel(skinWsNet, with_classification=True,
                                  tran_mean=tran_mean, garments=garments, garmentvnums=garmentvnums,
                                  upper_type_num=upper_type_num, pca_folder=pca_folder, pca_dim=pca_dim,
                                  tex_pca_dim=tex_pca_dim, tex_witdh=tex_width, tex_height=tex_height,
                                  smpl_model_path=smpl_model_path, infer_camera=infer_camera, infer_tex=infer_tex,
                                  use_detail=use_detail, create_detail_meshes=create_detail_meshes,
                                  dense_template_folder=dense_template_folder,
                                  light_instance_scale=light_instance_scale,
                                  displacement_scale=displacement_scale, device=device)

    pretrain_models = ['skinWsNet', 'imgEncoder']

    # 权重恢复与载入
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device(device))
        if pre_epoch == 0 and pre_batch == 0:
            pop_list = ['garPcapsLayers', 'texPcapsLayers', 'detailDecoderLayers']
            pop_params = [k for k in state_dict.keys() if any(k.startswith(item) for item in pop_list)]
            for p in pop_params:
                state_dict.pop(p, None)
                print('pop params: ' + p)
        net.load_state_dict(state_dict, strict=False)
    else:
        if not pre_epoch == 0 or not pre_batch == 0:
            logger.error('model path did not exist, check the model path: %s' % model_path)
            exit(-1)
        logger.warning('model path did not exist, load default model.')
        pretrained_params = torch.load(os.path.join(model_folder, 'garNet.pth'), map_location='cpu')
        state_params = {k: v for k, v in pretrained_params.items() if k.split('.')[0] in pretrain_models}
        net.load_state_dict(state_params, strict=False)
        net.imgEncoder.tran_mean = torch.as_tensor([0.0, 0.0, 0.0])

        for k, v in net.named_parameters():
            if 'garDisplacementLayers' in k and v.dtype == torch.float32:
                v.data.normal_(mean=0, std=0.02)

    # 梯度冻结与网络训练模式切换更新
    coarse_layers = ['imgEncoder', 'garPcaparamLayers', 'up_classifier', 'bottom_classifier', 'texPcaparamLayers']
    detail_layers = ['patchEncoder', 'garDisplacementLayers', 'detailEncoder', 'detailDecoderLayers']

    if freeze == 'nothing':
        for k, v in net.named_parameters():
            if k.split('.')[0] in coarse_layers or k.split('.')[0] in detail_layers:
                v.requires_grad = True
    elif freeze == 'detail':
        for k, v in net.named_parameters():
            if k.split('.')[0] in coarse_layers:
                v.requires_grad = True
            elif k.split('.')[0] in detail_layers:
                v.requires_grad = False
    elif freeze == 'coarse':
        for k, v in net.named_parameters():
            if k.split('.')[0] in coarse_layers:
                v.requires_grad = False
            elif k.split('.')[0] in detail_layers:
                v.requires_grad = True

        net.imgEncoder.eval()
        net.garPcaparamLayers.eval()
        net.up_classifier.eval()
        net.bottom_classifier.eval()
        net.texPcaparamLayers.eval()
    else:
        raise NotImplementedError

    net = net.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    mrf_fn = IDMRFLoss(device=device)
    total_batches = len(train_data_loader)

    # 核心训练循环
    for epoch in range(pre_epoch, epoch_num):
        logger.info(f'--- Start Epoch {epoch}/{epoch_num} ---')
        if epoch > pre_epoch:
            pre_batch = 0

        for batch_idx, batch_data in enumerate(train_data_loader):
            batch = pre_batch + batch_idx

            (imgs, img_names, masks, Rs, pca_perg_gt, gtypes,
             upper_type_gt, lower_type_gt, gps_gt, shapes_gt, poses_gt, trans_gt,
             cam_R_gt, cam_T_gt, tex_pca_perg_gt) = batch_data

            imgs = imgs.to(device)
            imgs_perg = torch.cat((imgs, imgs), 1).reshape(-1, 3, 540, 540)
            masks = masks.to(device)
            Rs = Rs.to(device)
            pca_perg_gt = pca_perg_gt.to(device)
            upper_type_gt = upper_type_gt.to(device)
            lower_type_gt = lower_type_gt.to(device)
            gps_gt = gps_gt.to(device)
            shapes_gt = shapes_gt.to(device)
            poses_gt = poses_gt.to(device)
            trans_gt = trans_gt.to(device)
            cam_R_gt = cam_R_gt.to(device)
            cam_T_gt = cam_T_gt.to(device)
            tex_pca_perg_gt = tex_pca_perg_gt.to(device)

            smpl = net.smpl
            body_tpose_ps_gt, _, _, _ = smpl(shapes_gt, torch.zeros_like(poses_gt), True, True)
            body_ps_gt, _, _, Js_gt = smpl(shapes_gt, Rs, True, False)
            body_ps_gt += trans_gt.unsqueeze(1)
            Js_gt += trans_gt.unsqueeze(1)

            renderer_gt = Renderer(540, 540, 1.5, 1.5, 0, 0, R=cam_R_gt.view(-1, 3, 3), T=cam_T_gt, device=device)
            Js_2d_gt = renderer_gt.transform_points(Js_gt)[:, :, :2]

            # 周期性保存模型
            if saving_model and epoch > pre_epoch and epoch % 5 == 0 and batch == 0:
                sub_model_folder = os.path.join(model_folder, save_title)
                os.makedirs(sub_model_folder, exist_ok=True)
                cur_model_path = f'{sub_model_folder}/{save_title}_pca{pca_dim}_ep{epoch}_bth{batch}.pth'
                torch.save(net.state_dict(), cur_model_path)
                logger.info('model saved in: ' + cur_model_path)

            # 前向传播
            outputs = net(imgs, img_names, gtypes=gtypes, check=False, cam_k=cam_k,
                          cam_R_gt=cam_R_gt, cam_T_gt=cam_T_gt, Js_2d_gt=Js_2d_gt,
                          tex_pca_perg_gt=tex_pca_perg_gt,
                          img_masks=masks, imgs_perg=imgs_perg)

            (garbatch, gps_pca, shapes_pred, poses_pred, trans_pred, pca_perg_pred,
             Js_pred, up_gar_prob, bottom_gar_prob, cam_R_pred, cam_T_pred, tex_pca_perg_pred,
             Js_2d_pred, rendered_coarse_imgs, rendered_detail_imgs, dis_pred, detail_textures,
             extracted_textures, uv_vis_masks, lights, detail_shading_images) = outputs

            all_loss = 0.0
            loss_str_list = []

            # 各种 Loss 函数链式计算与收集
            if weights['shape_param_loss']['all'] > 0:
                shape_param_loss, _ = calc_shape_params_loss(shapes_pred, poses_pred, trans_pred, pca_perg_pred,
                                                             shapes_gt, Rs, trans_gt, pca_perg_gt,
                                                             weights['shape_param_loss'])
                all_loss += shape_param_loss
                loss_str_list.append(f"shape: {shape_param_loss.item():.4f}")

            if weights['classify_loss']['all'] > 0:
                classify_loss, _ = calc_classify_loss(up_gar_prob, bottom_gar_prob, upper_type_gt, lower_type_gt,
                                                      weights['classify_loss'])
                all_loss += classify_loss
                loss_str_list.append(f"class: {classify_loss.item():.4f}")

            if weights['camera_loss']['all'] > 0:
                camera_loss, _ = calc_camera_loss(cam_R_pred, cam_R_gt, cam_T_pred, cam_T_gt, weights['camera_loss'])
                all_loss += camera_loss
                loss_str_list.append(f"cam: {camera_loss.item():.4f}")

            if weights['texture_loss']['all'] > 0:
                tex_loss, _ = calc_texture_loss(tex_pca_perg_pred, tex_pca_perg_gt, weights['texture_loss'])
                all_loss += tex_loss
                loss_str_list.append(f"tex: {tex_loss.item():.4f}")

            if weights['landmark_loss']['all'] > 0:
                landmark_loss, _ = calc_landmark_loss(Js_pred, Js_gt, Js_2d_pred, Js_2d_gt, weights['landmark_loss'])
                all_loss += landmark_loss
                loss_str_list.append(f"landmark: {landmark_loss.item():.4f}")

            if weights['photometric_loss']['all'] > 0:
                photometric_loss, _ = calc_photometric_loss(rendered_coarse_imgs, rendered_detail_imgs,
                                                            imgs_perg, masks, weights['photometric_loss'])
                all_loss += photometric_loss
                loss_str_list.append(f"photo: {photometric_loss.item():.4f}")

            if weights['mrf_loss']['all'] > 0:
                mrf_loss, _ = calc_mrf_loss(detail_textures * uv_vis_masks, extracted_textures * uv_vis_masks, mrf_fn, weights['mrf_loss'])
                all_loss += mrf_loss
                loss_str_list.append(f"mrf: {mrf_loss.item():.4f}")

            if weights['midline_loss']['all'] > 0:
                midpair_lverts = torch.zeros((0, 3), device=device)
                midpair_rverts = torch.zeros((0, 3), device=device)
                for i in range(len(upper_type_gt)):
                    utype = upper_type_gt[i]
                    if utype == 0 or utype == 2:
                        select = (garbatch == i * 2)
                        garverts = gps_pca[select]
                        gar_type = 'T-shirt' if utype == 0 else 'Shirt'
                        for pair in midpairs[gar_type]:
                            midpair_lverts = torch.cat((midpair_lverts, torch.unsqueeze(garverts[pair[0]], dim=0)),
                                                       dim=0)
                            midpair_rverts = torch.cat((midpair_rverts, torch.unsqueeze(garverts[pair[1]], dim=0)),
                                                       dim=0)
                midline_loss, _ = calc_midline_loss(midpair_lverts, midpair_rverts, weights['midline_loss'])
                all_loss += midline_loss
                loss_str_list.append(f"midline: {midline_loss.item():.4f}")

            if weights['shading_smooth_loss']['all'] > 0:
                shading_smooth_loss, _ = calc_shading_smooth_loss(detail_shading_images, weights['shading_smooth_loss'])
                all_loss += shading_smooth_loss
                loss_str_list.append(f"shading: {shading_smooth_loss.item():.4f}")

            if weights['reg_loss']['all'] > 0:
                reg_loss, _ = calc_reg_loss(shapes_pred, poses_pred, trans_pred, cam_R_pred, cam_T_pred,
                                            tex_pca_perg_pred, lights, dis_pred, weights['reg_loss'])
                all_loss += reg_loss
                loss_str_list.append(f"reg: {reg_loss.item():.4f}")

            assert not torch.isnan(all_loss), "Loss became NaN!"
            loss_str_list.append(f"Total: {all_loss.item():.4f}")

            # 终端单行动态覆盖打印
            print(f"\rBatch [{batch_idx + 1}/{total_batches}] " + " | ".join(loss_str_list), end="", flush=True)

            # 反向传播
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        scheduler.step()

    print('\nOptimized Training Session Finished.')