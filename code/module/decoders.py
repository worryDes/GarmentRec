import torch
import torch.nn as nn
from torch.nn import Module
import os.path as osp
import numpy as np
import openmesh as om
import pickle
from module.GCNs import MultiPerceptro
import cv2
from utils import generate_triangles, load_obj

class GarmentPcaDecodeLayer(Module):
    def __init__(self, pca_npz, dense_template_path):
        super(GarmentPcaDecodeLayer,self).__init__()
        datas=np.load(pca_npz)
        self.register_buffer('mean',torch.from_numpy(datas['mean'].astype(np.float32)))
        self.register_buffer('components',torch.from_numpy(datas['components'].astype(np.float32)))
        self.register_buffer('std',torch.from_numpy(datas['singular_values'].astype(np.float32)))
        self.type=osp.splitext(osp.basename(pca_npz))[0]
        # load standard template
        mesh=om.read_trimesh(osp.join(osp.dirname(pca_npz),'garment_tmp.obj'))
        self.register_buffer('edge_index',torch.from_numpy(mesh.hv_indices().transpose()).to(torch.long))
        self.register_buffer('face_index',torch.from_numpy(mesh.face_vertex_indices()).to(torch.long))
        # load dense template
        dense_template = None
        if osp.exists(dense_template_path):
            dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1')
        self.dense_template = dense_template

        dense_verts, dense_uvcoords, dense_faces, dense_uv_faces = load_obj(osp.join(osp.dirname(pca_npz),'garment_tmp_subdivide_uv_new.obj'))
        dense_verts_uvs = torch.zeros(dense_verts.shape[0], 2)
        dense_verts_uvs[dense_faces.reshape(-1)] = dense_uvcoords[dense_uv_faces.reshape(-1)]
        self.register_buffer('dense_faces', dense_faces)
        self.register_buffer('dense_faces_uvs', dense_uv_faces)
        self.register_buffer('dense_vt', dense_uvcoords)
        
        # dense_mesh = om.read_trimesh(osp.join(osp.dirname(pca_npz),'garment_tmp_subdivide_uv.obj'))
        # self.register_buffer('dense_edge_index',torch.from_numpy(dense_mesh.hv_indices().transpose()).to(torch.long))
        # self.register_buffer('dense_face_index',torch.from_numpy(dense_mesh.face_vertex_indices()).to(torch.long))

        vf_fid=torch.zeros(0,dtype=torch.long)
        vf_vid=torch.zeros(0,dtype=torch.long)
        for vid,fids in enumerate(mesh.vertex_face_indices()):
            fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
            vf_fid=torch.cat((vf_fid,fids),dim=0)
            vf_vid=torch.cat((vf_vid,fids.new_ones(fids.shape)*vid),dim=0)
        self.register_buffer('vf_findex',vf_fid)
        self.register_buffer('vf_vindex',vf_vid)
        self.vnum=mesh.n_vertices()
        self.fnum=mesh.n_faces()
        
        
        
        # vf_fid=torch.zeros(0,dtype=torch.long)
        # vf_vid=torch.zeros(0,dtype=torch.long)
        # for vid,fids in enumerate(dense_mesh.vertex_face_indices()):
        #     fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
        #     vf_fid=torch.cat((vf_fid,fids),dim=0)
        #     vf_vid=torch.cat((vf_vid,fids.new_ones(fids.shape)*vid),dim=0)
        # self.register_buffer('dense_vf_findex',vf_fid)
        # self.register_buffer('dense_vf_vindex',vf_vid)
        self.dense_vnum=dense_verts.shape[0]
        self.dense_fnum=dense_faces.shape[0]
        
    def unregular_pcas(self,pcas):
        return pcas*self.std.view(1,-1)
    def forward(self,pcas):
        return torch.matmul(pcas,self.components)+self.mean.view(1,-1)


class TexturePcaDecodeLayer(Module):
    def __init__(self, pca_npz, tex_uv_path, uv_mask_path, gtype):
        super(TexturePcaDecodeLayer,self).__init__()
        datas=np.load(pca_npz)
        tex_uvs = pickle.load(open(tex_uv_path, 'rb'))
        uv_mask = cv2.imread(uv_mask_path, cv2.IMREAD_GRAYSCALE)
        # erode mask
        kernel = np.ones((3, 3), np.uint8)
        uv_mask_erosion = uv_mask
        # uv_mask_erosion = cv2.erode(uv_mask, kernel, iterations=1)
        
        tex_height = uv_mask.shape[0]
        tex_width = uv_mask.shape[1]
        
        dense_triangles = generate_triangles(tex_height, tex_width, mask=uv_mask_erosion)
        dense_triangles = torch.from_numpy(dense_triangles)
        
        # # vis
        # a = np.zeros(tex_height * tex_width)
        # for i in range(len(dense_triangles)):
        #     for j in range(len(dense_triangles[i])):
        #         ind = dense_triangles[i][j]
        #         a[ind] = 1
        # cv2.imwrite('test_tris.png', a.reshape(tex_height, tex_width) * 255)
        
        # # test: load dense template
        # dense_template_path = 'data/texture_data_%s_256.npy' % gtype
        # if osp.exists(dense_template_path):
        #     dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1')
        #     img_size = dense_template['img_size']
        #     dense_faces = dense_template['f']
        #     x_coords = dense_template['x_coords']
        #     y_coords = dense_template['y_coords']
            
        #     valid_pixel_ids = dense_template['valid_pixel_ids']
        #     valid_pixel_ids_t = valid_pixel_ids % img_size * img_size + valid_pixel_ids // img_size
        #     dense_triangles = torch.from_numpy(valid_pixel_ids_t[dense_faces])
        
        # tri_mask = None
        
        # dense_triangles = generate_triangles(tex_height, tex_width, mask=None, valid_pixels=valid_pixel_ids_t)
        # dense_triangles = torch.from_numpy(dense_triangles)
        
        self.register_buffer('mean',torch.from_numpy(datas['tex_mean'].astype(np.float32)))
        self.register_buffer('components',torch.from_numpy(datas['tex_components'].astype(np.float32)))
        self.register_buffer('std',torch.from_numpy(datas['tex_singular_values'].astype(np.float32)))
        self.register_buffer('faces_uvs', tex_uvs['faces_uvs'])
        self.register_buffer('verts_uvs', tex_uvs['verts_uvs'])
        # self.register_buffer('dense_faces_uvs', tex_uvs['dense_faces_uvs'])
        # self.register_buffer('dense_verts_uvs', self.verts_uvs)
        self.register_buffer('uv_mask', torch.from_numpy(uv_mask))
        self.register_buffer('uv_mask_erosion', torch.from_numpy(uv_mask_erosion))
        self.register_buffer('dense_triangles', dense_triangles)
    def unregular_pcas(self,pcas):
        return pcas*self.std.view(1,-1)
    def forward(self,pcas):
        return torch.matmul(pcas,self.components)+self.mean.view(1,-1)  


class GarmentPcaLayer(Module):
    def __init__(self, gtype, latent_size, pca_dim=64):
        super(GarmentPcaLayer,self).__init__()
        self.gtype=gtype	
        if type(latent_size)==list:
            self.decoder=MultiPerceptro(latent_size)	
        else:
            self.decoder=MultiPerceptro([latent_size, 128, pca_dim])
    def forward(self,xs):
        return self.decoder(xs)


class TexturePcaLayer(Module):
    def __init__(self, gtype, latent_size, pca_dim=64):
        super(TexturePcaLayer,self).__init__()
        self.gtype = gtype
        if type(latent_size)==list:
            self.decoder=MultiPerceptro(latent_size)	
        else:
            self.decoder=MultiPerceptro([latent_size, 128, pca_dim])
    
    def forward(self,xs):
        return self.decoder(xs)


# detail decoder
class DetailDecoder(Module):
    def __init__(self, gtype, latent_size=128, tex_w=256, tex_h=256):
        super(DetailDecoder,self).__init__()
        self.gtype = gtype
        if type(latent_size)==list:
            self.decoder=MultiPerceptro(latent_size)	
        else:
            self.decoder=MultiPerceptro([latent_size, tex_w * tex_h * 3])
    
    def forward(self,xs):
        return self.decoder(xs)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, out_scale=0.01, sample_mode = 'bilinear', gtype=None):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        
        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode), #16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode), #256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img*self.out_scale