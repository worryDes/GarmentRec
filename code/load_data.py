from torch.utils.data import Dataset, DataLoader
import os
from utils import *
import numpy as np
from smpl_pytorch.util import batch_rodrigues
import torch

# 将映射关系定义为常量，避免在函数内部重复创建
UPPER_TYPE_MAP = {'T-shirt': 0, 'front_open_T-shirt': 1, 'Shirt': 2}
LOWER_TYPE_MAP = {'Shorts': 0}


class BCNetDataset(Dataset):
    def __init__(self, type, img_folder, mask_folder, items, train_garment_types, exclude_real, exclude_standard_side,
                 load_mask=True):
        super(BCNetDataset, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.items = items
        self.load_mask = load_mask

        # 优化1：只读取一次目录
        img_names = os.listdir(img_folder)
        self.img_list = []  # 存储元组: (img_name, sid)
        tot = 0

        for img_name in img_names:
            if img_name.startswith('s'):
                continue
            if exclude_real and '_syn_' not in img_name:
                continue

            img_id = img_name[img_name.rfind('_') + 1: img_name.find('.')]
            if exclude_standard_side and int(img_id) < 20:
                continue

            upper_mask_path = os.path.join(mask_folder, img_name.replace('.jpg', '_mask_upper.png'))
            lower_mask_path = os.path.join(mask_folder, img_name.replace('.jpg', '_mask_lower.png'))
            if load_mask and (not os.path.exists(upper_mask_path) or not os.path.exists(lower_mask_path)):
                tot += 1
                continue

            sp = img_name.split('_')
            sid = sp[0] + '_' + sp[1]
            if sp[0].startswith('s'):
                sid += '_' + sp[2]

            if sid not in items:
                continue
            data = items[sid]
            if None in data['tex_pca_params']:
                continue
            upper_type = data['detail_upper_type']
            lower_type = data['lower_type']
            if upper_type not in train_garment_types and lower_type not in train_garment_types:
                continue

            # 优化2：在初始化时就绑定好 sid，避免 getitem 里面重复计算字符串
            self.img_list.append((img_name, sid))

        if load_mask:
            print('remove no mask:' + str(tot))

        logger.info('%s data prepared.' % type)

    def __getitem__(self, idx):
        d = dict()
        img_name, sid = self.img_list[idx]  # 直接解包

        img_path = os.path.join(self.img_folder, img_name)
        img = read_img(img_path)
        d['img'] = img
        d['img_name'] = img_name

        if self.load_mask:
            # 优化3：更高效的字符串替换，减少 find 操作
            base_name = img_name.rsplit('.', 1)[0]
            upper_mask_path = os.path.join(self.mask_folder, base_name + '_mask_upper.png')
            lower_mask_path = os.path.join(self.mask_folder, base_name + '_mask_lower.png')
            upper_mask = read_img(upper_mask_path, color=False)
            lower_mask = read_img(lower_mask_path, color=False)
            assert (upper_mask is not None and lower_mask is not None)
            d['upper_mask'] = upper_mask
            d['lower_mask'] = lower_mask

        # 优化4：直接使用预存的 sid
        for k, v in self.items[sid].items():
            d[k] = v
        return d

    def __len__(self):
        return len(self.img_list)


def collate_fn(batch, load_mask=True):
    batch_data = dict()
    for i in range(len(batch)):
        for k, v in batch[i].items():
            if k not in batch_data:
                batch_data[k] = []
            batch_data[k].append(v)

    # 优化5：建议保持在 cpu 上，后续通过 data loader 的 pin_memory 并在 train loop 移至 GPU
    device = 'cpu'

    # img
    imgs = torch.from_numpy(np.stack(batch_data['img'], axis=0)).to(device)
    img_names = batch_data['img_name']

    # mask
    if load_mask:
        upper_masks = torch.from_numpy(np.stack(batch_data['upper_mask'], axis=0)).to(device)
        lower_masks = torch.from_numpy(np.stack(batch_data['lower_mask'], axis=0)).to(device)
        h, w = upper_masks.shape[1], upper_masks.shape[2]
        assert (h == lower_masks.shape[1] and w == lower_masks.shape[2])
        masks = torch.cat((upper_masks.permute(0, 3, 1, 2), lower_masks.permute(0, 3, 1, 2)), 1).reshape(-1, 1, h, w)

    # pose rodrigues matrix
    poses = torch.as_tensor(np.array(batch_data['poses_repose']), dtype=torch.float32, device=device)
    Rs = batch_rodrigues(poses.view(-1, 3)).view(-1, 24 * 3 * 3)

    # pca params
    pca_params = batch_data['pca_params']
    pca_params = np.array([[x if x is not None else np.zeros(64) for x in param] for param in pca_params])
    pca_perg_gt = torch.as_tensor(pca_params, dtype=torch.float32, device=device).view(-1, pca_params[0].shape[1])

    # garment types
    # 优化6：使用 dict.get() 代替 if-else 链
    upper_type_gt = [UPPER_TYPE_MAP.get(t, 3) for t in batch_data['detail_upper_type']]
    lower_type_gt = [LOWER_TYPE_MAP.get(t, 1) for t in batch_data['lower_type']]

    upper_type_gt = torch.as_tensor(upper_type_gt, dtype=torch.long, device=device)
    lower_type_gt = torch.as_tensor(lower_type_gt, dtype=torch.long, device=device)
    assert (upper_type_gt.shape == lower_type_gt.shape)

    # 优化7：列表推导式简化 gtypes 组合
    gtypes = [[int(u), int(l) + 4] for u, l in zip(upper_type_gt, lower_type_gt)]

    # garments
    uppers = batch_data['upper_repose']
    lowers = batch_data['lower_repose']
    gps_gt = []
    assert (len(uppers) == len(lowers))
    for i in range(len(uppers)):
        for j in range(len(uppers[i])):
            gps_gt.append(np.array(uppers[i][j]))
        for j in range(len(lowers[i])):
            gps_gt.append(np.array(lowers[i][j]))
    gps_gt = torch.as_tensor(np.array(gps_gt), dtype=torch.float32, device=device)

    # smpl params
    shapes_gt = torch.as_tensor(np.array(batch_data['shapes']), dtype=torch.float32, device=device)
    poses_gt = torch.as_tensor(np.array(batch_data['poses_repose']), dtype=torch.float32, device=device)
    trans_gt = torch.as_tensor(np.array(batch_data['trans_repose']), dtype=torch.float32, device=device)

    # camera params
    # 优化8：告别循环中的 torch.cat，改用列表收集后一次性 stack/cat
    cam_R_list = []
    cam_T_list = []
    for i in range(len(img_names)):
        k = img_names[i]
        if k not in batch_data['camera_params'][i]:
            R = torch.eye(3, 3).view(1, -1)
            T = torch.zeros(1, 3).view(1, -1)
        else:
            d = batch_data['camera_params'][i][k]
            R = torch.as_tensor(d['R'], dtype=torch.float32, device=device).view(1, -1)
            T = torch.as_tensor(d['T'], dtype=torch.float32, device=device).view(1, -1)
        cam_R_list.append(R)
        cam_T_list.append(T)

    cam_R_gt = torch.cat(cam_R_list, dim=0)
    cam_T_gt = torch.cat(cam_T_list, dim=0)

    # tex pca params
    tex_pca_params = batch_data['tex_pca_params']
    tex_pca_params = np.array([[x if x is not None else np.zeros(64) for x in param] for param in tex_pca_params])
    tex_pca_perg_gt = torch.as_tensor(tex_pca_params, dtype=torch.float32, device=device).view(-1,
                                                                                               tex_pca_params[0].shape[
                                                                                                   1])

    if not load_mask:
        masks = torch.cat((torch.ones_like(imgs), torch.ones_like(imgs)), dim=0)

    return imgs, img_names, masks, Rs, pca_perg_gt, gtypes, upper_type_gt, lower_type_gt, gps_gt, shapes_gt, poses_gt, trans_gt, cam_R_gt, cam_T_gt, tex_pca_perg_gt


def get_dataloader(type, img_folder, mask_folder, items, batch_size, shuffle, num_workers,
                   train_garment_types=['T-shirt', 'Shirt', 'front_open_T-shirt', 'front_open_Shirt', 'Shorts',
                                        'Pants'],
                   exclude_real=False, exclude_standard_side=False, load_mask=True):
    if train_garment_types == ['all']:
        train_garment_types = ['T-shirt', 'Shirt', 'front_open_T-shirt', 'front_open_Shirt', 'Shorts', 'Pants']
    dataset = BCNetDataset(type, img_folder, mask_folder, items, train_garment_types, exclude_real,
                           exclude_standard_side, load_mask)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,  # 别忘了把传入的 num_workers 加上
                            pin_memory=True if num_workers > 0 else False,  # 开启 pin_memory 加快 CPU->GPU 拷贝
                            collate_fn=lambda batch: collate_fn(batch, load_mask=load_mask))
    return dataloader