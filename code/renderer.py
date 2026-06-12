import torch.nn as nn
import torch
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import *
from pytorch3d.structures import Meshes
import torch.nn.functional as F
import numpy as np

class Renderer(nn.Module):
    def __init__(self, img_w, img_h, fx_ndc, fy_ndc, px_ndc, py_ndc, R, T, lights=None, device='cpu'):
        super().__init__()
        # camera settings
        s = min(img_h, img_w)
        fx_screen = fx_ndc * s / 2
        fy_screen = fy_ndc * s / 2
        px_screen = img_w / 2 - px_ndc * s / 2
        py_screen = img_h / 2 - py_ndc * s / 2
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = R.shape[0]

        fcl_screen = torch.tensor([fx_screen, fy_screen], dtype=torch.float32).reshape(1, 2).to(device)
        prp_screen = torch.tensor([px_screen, py_screen], dtype=torch.float32).reshape(1, 2).to(device)

        cameras_screen = PerspectiveCameras(
            device=device,
            R=R,
            T=T,
            focal_length=fcl_screen,
            principal_point=prp_screen,
            in_ndc=False,
            image_size=[(img_h, img_w)])

        self.cameras = cameras_screen

        # raster settings
        raster_settings = RasterizationSettings(
            image_size=(img_h, img_w),
            blur_radius=0,
            faces_per_pixel=1,
        )

        if lights is None:
            # default light settings
            pointlights = PointLights(
                device=device,
                ambient_color=((1.0, 1.0, 1.0),),
                diffuse_color=((0.0, 0.0, 0.0),),
                specular_color=((0.0, 0.0, 0.0),),
                location=((0.0, 0.0, 1e5),)
            )
        else:
            light_positions = lights[:, :3]
            light_intensities = lights[:, 3:]
            pointlights = PointLights(
                device=device,
                ambient_color=light_intensities,
                diffuse_color=((0.0, 0.0, 0.0),),
                specular_color=((0.0, 0.0, 0.0),),
                location=light_positions
            )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras_screen,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras_screen,
                lights=pointlights,
                blend_params=BlendParams(background_color=[0, 0, 0])
            )
        )

        raster_settings_mask = RasterizationSettings(
            image_size=(img_h, img_w),
            blur_radius=0,
            faces_per_pixel=1,
            cull_backfaces=True
        )
        renderer_mask = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras_screen,
                raster_settings=raster_settings_mask
            ),
            shader=SoftSilhouetteShader(
                blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
            )
        )

        self.raster_settings_mesh = RasterizationSettings(
            image_size=(img_h, img_w),  # 输出图像尺寸，比如256或512
            blur_radius=np.log(1.0 / 1e-4) * 1e-7,  # 细节模糊参数，接近于0
            faces_per_pixel=1,  # 每像素考虑的三角形数
            cull_backfaces=True,
        )
        self.meshRas = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings_mesh
        )
        self.renderer_normal = MeshRenderer(
            rasterizer=self.meshRas,
            shader=cleanShader(device=device, blend_params=BlendParams(1e-4, 1e-8, (0.5, 0.5, 0.5))),
        )

        self.renderer_mask = renderer_mask
        self.renderer = renderer

    def transform_points(self, points, target='screen_space'):
        if target == 'screen_space':
            return self.cameras.transform_points_screen(points)
        elif target == 'camera_space':
            return self.cameras.get_world_to_view_transform().transform_points(points)
        elif target == 'NDC_space':
            return self.cameras.get_ndc_camera_transform().transform_points(points)
        else:
            raise NotImplementedError

    def get_matrix(self):
        return self.cameras.get_world_to_view_transform().get_matrix()

    def forward(self, mesh):
        imgs = self.renderer(mesh)
        return imgs[:, :, :, 0:3].permute(0, 3, 1, 2)

    def get_mask(self, mesh):
        silhouette = self.renderer_mask(mesh)
        return (silhouette[..., 3] > 0.1).float()

    def get_pixel_coords(self, verts):
        if verts.shape[0] == 0:
            return torch.tensor([[0,0]], device=verts.device)
        # 转屏幕坐标
        verts_screen = self.transform_points(verts, target='screen_space')  # (N,3)
        # 返回像素坐标 (x, y)，保留梯度
        pixel_coords = verts_screen[:, [1, 0]]
        return pixel_coords

    def get_contour_vertices_index(self, verts, contour_pixels):
        device = verts.device
        contour_pixels = torch.tensor(contour_pixels, device=device, dtype=torch.long)

        # Step 1: 投影到屏幕空间 ([-1,1] -> 像素坐标)
        verts_screen = self.transform_points(verts, target='screen_space')  # (V,3)
        verts_pix = verts_screen[:, :2].round().long()  # (V,2)，取整数像素位置

        # Step 2: 构造查找表，找到哪些顶点落在 contour 上
        H, W = self.img_h, self.img_w
        mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        rows = contour_pixels[:, 0]
        cols = contour_pixels[:, 1]
        mask[rows, cols] = True
        mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mask_dilated = F.max_pool2d(mask_f, kernel_size=11, stride=1, padding=5)
        mask_dilated = mask_dilated.squeeze().bool()
        mask = mask_dilated

        # Step 3: 过滤出在轮廓上的顶点
        row = verts_pix[:, 1].clamp(0, H - 1)
        col = verts_pix[:, 0].clamp(0, W - 1)
        inside_mask = mask[row, col]

        contour_vertex_idx = torch.where(inside_mask)[0]  # (Nidx,)

        return contour_vertex_idx

    def render_normal(self, mesh):
        """
        渲染 mesh（不受光照影响，仅显示自身 Texture 颜色）。
        要求 mesh 已经包含 TexturesVertex 或 TexturesAtlas。
        """

        return self.renderer_normal(mesh)

class cleanShader(torch.nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)

        # 采样纹理颜色（这里应该是法线颜色映射后的顶点颜色）
        texels = meshes.sample_textures(fragments)

        # 基于深度和权重做softmax加权融合，得到无光照纯颜色渲染结果
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images

class LocalAffine(nn.Module):
    def __init__(self, num_points, batch_size=1, edges=None):
        '''
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        '''
        super(LocalAffine, self).__init__()
        self.A = nn.Parameter(
            torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1)
        )
        self.b = nn.Parameter(
            torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(
                batch_size, num_points, 1, 1
            )
        )
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        '''
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix,
        '''
        if self.edges is None:
            raise Exception("edges cannot be none when calculate stiff")
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=3)
        w1 = torch.index_select(affine_weight, dim=1, index=idx1)
        w2 = torch.index_select(affine_weight, dim=1, index=idx2)
        w_diff = (w1 - w2)**2
        w_rigid = (torch.linalg.det(self.A) - 1.0)**2
        return w_diff, w_rigid

    def forward(self, x, return_stiff=False):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        if return_stiff:
            stiffness, rigid = self.stiffness()
            return out_x, stiffness, rigid
        else:
            return out_x