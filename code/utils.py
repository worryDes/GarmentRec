import torch
import torch_scatter
import numpy as np
import cv2
from loguru import logger
from typing import Union
import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer import *
import yaml
from functools import cmp_to_key
from trimesh.registration import icp, procrustes, transform_points
import math
from torchvision.ops import roi_align
import torch.nn.functional as F
from pytorch3d.ops import SubdivideMeshes
from sklearn.neighbors import NearestNeighbors
from pytorch3d.io import load_objs_as_meshes
import pickle
from random import random
import pymeshlab

def read_img(file, color=True):
	# logger.warning(file)
	if color:
		img=cv2.imread(file)
	else:
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	if img is None:
		logger.error('no exist img: ' + file)
		return img
	try:
		h=img.shape[0]
		w=img.shape[1]
	except:
		logger.error('error file: %s' % file)
		raise AttributeError

	if h!=w:
		l=max(h,w)
		nimg=np.zeros((l,l,3),np.uint8)
		hs=max(int((l-h)/2.),0)
		he=min(int((l+h)/2.),l)
		he=min(he,hs+h)
		ws=max(int((l-w)/2.),0)
		we=min(int((l+w)/2.),l)
		we=min(we,ws+w)
		nimg[hs:he,ws:we]=img[:he-hs,:we-ws]
	else:
		nimg=img
	nimg=cv2.resize(nimg,(540,540))
	if nimg.shape[-1] == 3:
		nimg=nimg.transpose(2,0,1)
		nimg=nimg.astype(np.float32)/255.
	else:
		nimg = np.expand_dims(nimg, 2)
	return nimg

def cmp(x, y):
		a = int(x[:x.find('.')])
		b = int(y[:y.find('.')])
		if a == b:
				return 0
		elif a < b:
				return -1
		else:
				return 1

def read_all_imgs(folder:str) -> list:
	img_names = os.listdir(folder)
	imgs = []
	img_names = sorted(img_names)
	for i in range(len(img_names)):
		imgs.append(read_img(os.path.join(folder, img_names[i])))
	return imgs


def batch_icp(ps_pred, ps_gt, Js_pred, Js_gt, garbatch_pred, garbatch_gt):
	voffset_pred = 0
	voffset_gt = 0
	ps_pred_icp = torch.zeros(0, device=ps_pred.device)
	for i in range(len(Js_pred) * 2):
		ind = i // 2
		select_pred = garbatch_pred == i
		vnum_pred = select_pred.sum()
		ps_one_pred = ps_pred[select_pred]
		voffset_pred += vnum_pred

		select_gt = garbatch_gt == i
		vnum_gt = select_gt.sum()
		ps_one_gt = ps_gt[select_gt]
		voffset_gt += vnum_gt
		ps_one_pred_icp, _ = one_icp(ps_one_pred, ps_one_gt, Js_pred[ind], Js_gt[ind])
		ps_pred_icp = torch.cat((ps_pred_icp, ps_one_pred_icp))
	return ps_pred_icp


def one_icp(ps_pred, ps_gt, Js_pred, Js_gt):
	Rt_init = procrustes(Js_pred.cpu(), Js_gt.cpu(), return_cost=False)
	Rt, _, _ = icp(ps_pred.cpu(), ps_gt.cpu(), initial=Rt_init)
	ps_pred_icp = torch.tensor(transform_points(ps_pred.cpu(), Rt), device=ps_gt.device)
	assert(ps_pred_icp.shape == ps_pred.shape)
	return ps_pred_icp, Rt


def save_obj(ps,tris,name):
	with open(name, 'w') as fp:
		for v in ps:
			fp.write( 'v {:f} {:f} {:f}\n'.format( v[0], v[1], v[2]) )
		if tris is not None:
			for f in tris: # Faces are 1-based, not 0-based in obj files
				fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1) )


def save_batch_objs(bps,face_index,batch,names, R=None, t=None, skip_exist=False):
	if len(bps.shape)==2:
		assert(len(names)==batch.max()+1)
		voffset=0
		for ind in range(len(names)):
			if skip_exist and os.path.exists(names[ind]):
				continue
			select=batch==ind
			vnum=select.sum()
			tris=face_index[(face_index>=voffset) * (face_index<voffset+vnum)].reshape(-1,3)-voffset
			ps=bps[select]
			if R is not None:
				ps = np.transpose(np.matmul(R[ind // 2, :, :], np.transpose(ps)))
			if t is not None:
				ps += t[ind // 2, :]
			save_obj(ps,tris,names[ind])
			voffset+=vnum
	elif len(bps.shape)==3:
		assert(bps.shape[0]==len(names))
		i = 0
		for ps,n in zip(bps,names):
			if skip_exist and os.path.exists(n):
				continue
			if R is not None:
				ps = np.transpose(np.matmul(R[i, :, :], np.transpose(ps)))
			if t is not None:
				ps += t[i, :]
			save_obj(ps,face_index,n)
	else:
		assert(False)
	

def get_batch_meshes(bps, face_index, batch, names, device='cpu'):
	meshes = []
	if len(bps.shape)==2:
		assert(len(names)==batch.max()+1)
		voffset=0
		for ind in range(0, len(names), 2):
			# upper
			select=batch==ind
			vnum_upper=select.sum()
			tris_upper=face_index[(face_index>=voffset) * (face_index<voffset+vnum_upper)].reshape(-1,3)-voffset
			ps_upper = bps[select]
			ps_upper = torch.tensor(ps_upper, device=device)
			tris_upper = torch.tensor(tris_upper, device=device)
			voffset+=vnum_upper
			# lower
			select=batch==(ind + 1)
			vnum_lower=select.sum()
			tris_lower=face_index[(face_index>=voffset) * (face_index<voffset+vnum_lower)].reshape(-1,3)-voffset
			tris_lower += vnum_upper
			ps_lower = bps[select]
			ps_lower = torch.tensor(ps_lower, device=device)
			tris_lower = torch.tensor(tris_lower, device=device)
			voffset+=vnum_lower

			ps = torch.cat((ps_upper, ps_lower), 0)
			tris = torch.cat((tris_upper, tris_lower), 0)
			# create dummy texture
			rgb_verts = torch.ones_like(ps)[None]
			textures = TexturesVertex(verts_features=rgb_verts)
			textured_meshes = Meshes(
				verts=[ps],
				faces=[tris],
				textures=textures
			)
			meshes.append(textured_meshes)
	elif len(bps.shape)==3:
		assert(bps.shape[0]==len(names))
		for ps,n in zip(bps,names):
			meshes.append(Meshes(ps, tris))
			# save_obj(ps,face_index,n)
	else:
		assert(False)
	return meshes


def compute_connectivity_infos_from_mesh(mesh,device=None):
	if type(mesh) is str:
		mesh=om.read_trimesh(mesh)
	face_index=torch.from_numpy(mesh.face_vertex_indices().astype(np.int64))
	vf_fid=torch.zeros(0,dtype=torch.long)
	vf_vid=torch.zeros(0,dtype=torch.long)
	for vid,fids in enumerate(mesh.vertex_face_indices()):
		fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
		vf_fid=torch.cat((vf_fid,fids),dim=0)
		vf_vid=torch.cat((vf_vid,fids.new_ones(fids.shape)*vid),dim=0)
	if device is not None:
		face_index=face_index.to(device)
		vf_fid=vf_fid.to(device)
		vf_vid=vf_vid.to(device)
	return face_index,vf_fid,vf_vid

#verts:(v,3) or (b,v,3), tri_fs:(f,3)
def compute_fnorms(verts,tri_fs):
	v0=verts.index_select(-2,tri_fs[:,0])
	v1=verts.index_select(-2,tri_fs[:,1])
	v2=verts.index_select(-2,tri_fs[:,2])
	e01=v1-v0
	e02=v2-v0
	fnorms=torch.cross(e01,e02,-1)
	diss=fnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	fnorms=fnorms/diss
	return fnorms


def compute_vnorms(verts,tri_fs,vertex_index,face_index):
	fnorms=compute_fnorms(verts,tri_fs)
	vnorms=torch_scatter.scatter(fnorms.index_select(-2,face_index),vertex_index,-2,None,verts.shape[-2])
	diss=vnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	vnorms=vnorms/diss
	return vnorms


def log_all(level: Union[int, str], msgs:list, sep=' ', max_len=0):
	all = ''
	for i in range(len(msgs)):
		x = str(msgs[i])
		if max_len > 0:
			x = x[:max_len]
		all += x
		if i != len(msgs) - 1:
			all += sep
	logger.log(level, all)
	

def Geman_McClure_Loss(x,c):
	return x*x*2.0/c/c/(x*x/c/c + 4.)


def lap_coords(L, V):
		return L @ V
	

def sin(d):
		return torch.sin(torch.deg2rad(d))


def cos(d):
		return torch.cos(torch.deg2rad(d))
	
	
def getRt(rx, ry, rz, tx, ty, tz, device='cpu'):
	"""通过旋转角度和平移组建R和t

	Args:
			rx (_type_): 绕x轴旋转
			ry (_type_): 绕y轴旋转
			rz (_type_): 绕z轴旋转
			tx (_type_): 沿x轴平移
			ty (_type_): 沿y轴平移
			tz (_type_): 沿z轴平移
			device (str, optional): device. Defaults to 'cpu'.

	Returns:
			_type_: R 3*3, and T 1*3
	"""
 
	Rx, Ry, Rz = torch.eye(3, device=device), torch.eye(3, device=device), torch.eye(3, device=device)
	
	Rx[1][1] = cos(rx)
	Rx[1][2] = -sin(rx)
	Rx[2][1] = sin(rx)
	Rx[2][2] = cos(rx)
 
	Ry[0][0] = cos(ry)
	Ry[0][2] = sin(ry)
	Ry[2][0] = -sin(ry)
	Ry[2][2] = cos(ry)
 
	Rz[0][0] = cos(rz)
	Rz[0][1] = -sin(rz)
	Rz[1][0] = sin(rz)
	Rz[1][1] = cos(rz)
 
	# Rx = torch.tensor([[1, 			 0, 			 0],
	# 									 [0, cos(rx), -sin(rx)],
	# 									 [0, sin(rx),  cos(rx)]], dtype=torch.float32, device=device)

	# Ry = torch.tensor([[cos(ry),  0, sin(ry)],
	# 									 [0, 			  1, 			 0],
	# 									 [-sin(ry), 0, cos(ry)]], dtype=torch.float32, device=device)

	# Rz = torch.tensor([[cos(rz), -sin(rz), 0],
	# 									 [sin(rz),  cos(rz), 0],
	# 									 [0, 							0, 1]], dtype=torch.float32, device=device)

	R = torch.mm(Rz, torch.mm(Ry, Rx))
	T = torch.zeros(3, device=device)
	T[0], T[1], T[2] = tx, ty, tz
	# T = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device)
	return R, T


def get_batch_RT(cam_exts, device='cpu'):
	"""通过旋转角度和平移组建R和t, for one batch

	Args:
			cam_exts: 相机外参, rx, ry, rz, tx, ty, tz, batch_size*6

	Returns:
			_type_: Rs batch_size*3*3, Ts batch_size*3
	"""
	# TODO: can remove for-loop ?
	batch_size = cam_exts.shape[0]
	Rs = torch.zeros(batch_size, 3, 3, device=device)
	Ts = torch.zeros(batch_size, 3, device=device)
	for i in range(batch_size):
		R, t = getRt(*cam_exts[i], device=device)
		Rs[i] = R
		Ts[i] = t
	return Rs, Ts


def create_meshes(verts, faces, maps, idx, types, garlayers, texlayers):
	num = torch.max(idx).item() + 1
	meshes = []
	vert_list = []
	face_list = []
	verts_normals_list = []
	faces_uvs_list = []
	verts_uvs_list = []
	voffset=0
	for i in range(num):
		select = idx==i
		vnum = select.sum().item()
		v = verts[select]
		f = faces[(faces >= voffset) * (faces < voffset + vnum)].reshape(-1,3) - voffset
		vf_vindex = garlayers[types[i]].vf_vindex
		vf_findex = garlayers[types[i]].vf_findex
		verts_normals = compute_vnorms(v, f, vf_vindex, vf_findex)
		faces_uvs = texlayers[types[i]].faces_uvs
		verts_uvs = texlayers[types[i]].verts_uvs
	
		vert_list.append(v)
		face_list.append(f)
		verts_normals_list.append(verts_normals)
		faces_uvs_list.append(faces_uvs)
		verts_uvs_list.append(verts_uvs)
		voffset += vnum
	assert(len(vert_list) == len(face_list))

	texs = TexturesUV(maps=maps, faces_uvs=faces_uvs_list, verts_uvs=verts_uvs_list)
	# TODO: verts normals
	meshes = Meshes(verts=vert_list, faces=face_list, textures=texs, verts_normals=verts_normals_list)
	assert((meshes.verts_packed_to_mesh_idx() == idx).all())
	assert(len(meshes) == num)
	return meshes, texs

def create_dense_meshes(verts, faces, normals, maps, verts_uvs, faces_uvs):
	texs = TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
	# TODO: verts normals
	meshes = Meshes(verts=verts, faces=faces, textures=texs, verts_normals=normals)
	assert(len(meshes) == len(verts))
	return meshes, texs


def save_obj(ps,tris,name):
	with open(name, 'w') as fp:
		for v in ps:
				fp.write( 'v {:f} {:f} {:f}\n'.format( v[0], v[1], v[2]) )
		if tris is not None:
				for f in tris: # Faces are 1-based, not 0-based in obj files
						fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1) )


def imgBatchFromGarBatch(batch,gtypes):
	garnums_perimg=(gtypes!=0).sum(1)		#per img has one up and one bottom, this is trainset situation
	if (garnums_perimg!=2).sum()==0:
		imgbatch=batch//2
	else:	#general situation
		imgbatch=batch.detach().clone()
		e_ids=garnums_perimg.cumsum(dim=0)
		e_ids=e_ids.detach().cpu().numpy()
		s_id=0
		for ind,e_id in enumerate(e_ids):
				if ind-1>=0:
						s_id=e_ids[ind-1]
				else:
						s_id=0
				imgbatch[(batch>=s_id)*(batch<e_id)]=ind
	return imgbatch
			
#datas is (all_vnums,feature_num) or batch_num,-1 datas
def order_data_follow_gartypes(datas,batch_num,gar_batch,gtypes,garmentvnums=[4248,4258,5327,3721,5404,2818],garments=['shirts','short_shirts','pants','short_pants','skirts','short_skirts']):
	indexs=gtypes.nonzero(as_tuple=False)
	gar_type_ids_per_gar=indexs[:,1]
	img_batch_ids_per_gar=indexs[:,0]
	gar_batch_ids_per_gar=torch.arange(indexs.shape[0],device=indexs.device,dtype=torch.long)
	ordered_datas=[]
	ordered_gtypes=[]
	ordered_select_img_bach_ids=[]
	for ind,garvnum in enumerate(garmentvnums):
		select_mask=(gar_type_ids_per_gar==ind)
		select_img_batch_ids=img_batch_ids_per_gar[select_mask]
		select_gar_batch_ids=gar_batch_ids_per_gar[select_mask]
		select_gars_num=select_gar_batch_ids.numel()
		if select_gars_num>0:
				if gar_batch is not None:
						select_rows=((gar_batch.view(-1,1)-select_gar_batch_ids.view(1,-1))==0).nonzero(as_tuple=False)[:,0]
				tmp=[]
				for data in datas:
						if data.shape[0]==batch_num:
								tmp.append(data[select_img_batch_ids].reshape(select_gars_num,data.shape[-1]))
						else:
								if gar_batch is None:
										assert(False)
								tmp.append(data[select_rows,:].reshape(select_gars_num,garvnum,data.shape[-1]))
				ordered_datas.append(tmp)
				ordered_gtypes.append(ind)
				ordered_select_img_bach_ids.append(select_img_batch_ids)
	return ordered_datas,ordered_select_img_bach_ids,ordered_gtypes
	
	
	# ordered_datas: pca datas
# len(ordered_datas): exist gar num
# len(ordered_datas[0]): 2 (pca, points)
# return datas: datas[0]: pca ((batch_size * 2) * 64), datas[1]: all points (sum_points * 3)
def unorder_data_follow_imgbatch(ordered_datas,ordered_imgbids,ordered_gtypes,batch_num,garlayers=None,texlayers=None, \
	require_gar_batch=False, require_edge_index=False, require_face_index=False,require_vffindex=False,require_vfvindex=False):
	record_offset=False
	if require_gar_batch or require_edge_index or require_face_index or require_vffindex or require_vfvindex:
			assert(garlayers is not None)
			
			assert(6==len(garlayers))
			
			record_offset=True

	datas=[]
	for ordered_data in ordered_datas[0]:
			datas.append(ordered_data.new_zeros(0,ordered_data.shape[-1]))
	if require_gar_batch:
			batch=torch.zeros(0,dtype=torch.long,device=ordered_imgbids[0].device)
	if require_edge_index:
			edge_index=torch.zeros(2,0,dtype=torch.long,device=ordered_imgbids[0].device)
	if require_face_index:
			face_index=torch.zeros(0,3,dtype=torch.long,device=ordered_imgbids[0].device)
	if require_vffindex:
			vf_findex=torch.zeros(0,dtype=torch.long,device=ordered_imgbids[0].device)
	if require_vfvindex:
			vf_vindex=torch.zeros(0,dtype=torch.long,device=ordered_imgbids[0].device)

	gid=0
	voffset=0
	foffset=0
	for bid in range(batch_num):
		for ind,img_bids,tmp_datas in zip(ordered_gtypes,ordered_imgbids,ordered_datas):
				select_mask=img_bids==bid
				if select_mask.sum()==0:
						continue
				if record_offset:
						garlayer=garlayers[ind]
						texlayer=texlayers[ind]
				for tid,(data,ordered_data) in enumerate(zip(datas,tmp_datas)):
						if ordered_data.dim()==2:
								
								data=torch.cat((data,ordered_data[select_mask]),dim=0)
						else:
								data=torch.cat((data,ordered_data[select_mask].reshape(-1,ordered_data.shape[-1])),dim=0)
						datas[tid]=data
				if require_gar_batch:
						batch=torch.cat((batch,batch.new_ones(garlayer.vnum)*gid),dim=0)
				if require_edge_index:
						edge_index=torch.cat((edge_index,garlayer.edge_index+voffset),dim=-1)
				if require_face_index:
						face_index=torch.cat((face_index,garlayer.face_index+voffset),dim=0)
				if require_vffindex:
						vf_findex=torch.cat((vf_findex,garlayer.vf_findex+foffset),dim=0)
				if require_vfvindex:
						vf_vindex=torch.cat((vf_vindex,garlayer.vf_vindex+voffset),dim=0)
				gid+=1
				if record_offset:
						voffset+=garlayer.vnum
						foffset+=garlayer.fnum
	other_out={}
	if require_gar_batch:
			other_out['gar_batch']=batch
	if require_edge_index:
			other_out['edge_index']=edge_index
	if require_face_index:
			other_out['face_index']=face_index
	if require_vffindex:
			other_out['vf_findex']=vf_findex
	if require_vfvindex:
			other_out['vf_vindex']=vf_vindex
	return datas,other_out

def get_patchs_from_imgs(pros,imgs,imgbatch,box_len=32):
	x1=pros[:,0]-box_len/2.
	x2=pros[:,0]+box_len/2.
	y1=pros[:,1]-box_len/2.
	y2=pros[:,1]+box_len/2.
	boxes=torch.stack((imgbatch.to(torch.float),x1,y1,x2,y2),dim=-1)
	return roi_align(imgs,boxes,(box_len,box_len))

def get_face_vertices(vertices, faces):
	""" 
	:param vertices: [batch size, number of vertices, 3]
	:param faces: [batch size, number of faces, 3]
	:return: [batch size, number of faces, 3, 3]
	"""
	assert (vertices.ndimension() == 3)
	assert (faces.ndimension() == 3)
	assert (vertices.shape[0] == faces.shape[0])
	assert (vertices.shape[2] == 3)
	assert (faces.shape[2] == 3)

	bs, nv = vertices.shape[:2]
	bs, nf = faces.shape[:2]
	device = vertices.device
	faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
	vertices = vertices.reshape((bs * nv, 3))
	# pytorch only supports long and byte tensors for indexing
	return vertices[faces.long()]


def world2uv(vertices, faces, uvcoords, uvfaces, uv_rasterizer):
	'''
	warp vertices from world space to uv space
	vertices: [bz, V, 3]
	faces: [bz, F, 3]
	uv_vertices: [bz, 3, h, w]
	'''
	batch_size = vertices.shape[0]
	face_vertices = get_face_vertices(vertices, faces)
	uv_vertices = uv_rasterizer(uvcoords, uvfaces, face_vertices)[:, :3]
	return uv_vertices


def batch_orth_proj(X, R, T):
	''' orthgraphic projection
		X:  3d vertices, [bz, n_point, 3]
		camera: scale and translation, [bz, 3], [scale, tx, ty]
	'''


	camera = camera.clone().view(-1, 1, 3)
	X_trans = X[:, :, :2] + camera[:, :, 1:]
	X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
	shape = X_trans.shape
	Xn = (camera[:, :, 0:1] * X_trans)
	return Xn


def get_vertex_normals(vertices, faces):
	"""
	:param vertices: [batch size, number of vertices, 3]
	:param faces: [batch size, number of faces, 3]
	:return: [batch size, number of vertices, 3]
	"""
	assert (vertices.ndimension() == 3)
	assert (faces.ndimension() == 3)
	assert (vertices.shape[0] == faces.shape[0])
	assert (vertices.shape[2] == 3)
	assert (faces.shape[2] == 3)
	bs, nv = vertices.shape[:2]
	bs, nf = faces.shape[:2]
	device = vertices.device
	normals = torch.zeros(bs * nv, 3).to(device)

	faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
	vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

	faces = faces.reshape(-1, 3)
	vertices_faces = vertices_faces.reshape(-1, 3, 3)

	normals.index_add_(0, faces[:, 1].long(), 
						torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
	normals.index_add_(0, faces[:, 2].long(), 
						torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
	normals.index_add_(0, faces[:, 0].long(),
						torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

	normals = F.normalize(normals, eps=1e-6, dim=1)
	normals = normals.reshape((bs, nv, 3))
	# pytorch only supports long and byte tensors for indexing
	return normals
	
# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=0, margin_y=0, mask=None, valid_pixels=None):
	# quad layout:
	# 0 1 ... w-1
	# w w+1
	#.
	# w*h
	triangles = []
	for x in range(margin_x, w-1-margin_x):
		for y in range(margin_y, h-1-margin_y):
			triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
			triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
			# if x < 0 or x >= w or y < 0 or y >= h or mask[y][x] == 0 or mask[y][x + 1] == 0 or mask[y + 1][x] == 0 or mask[y + 1][x + 1] == 0:
			# 	continue
			if valid_pixels is not None:
				valids0 = []
				valids1 = []
				for i in range(3):
					if triangle0[i] in valid_pixels:
						valids0.append(triangle0[i])
					if triangle1[i] in valid_pixels:
						valids1.append(triangle1[i])
				if not len(valids0) == 0 and not len(valids0) == 3:
					for i in range(3):
						if triangle0[i] in valids0:
							continue
						triangle0[i] = valids0[0]
				if not len(valids1) == 0 and not len(valids1) == 3:
					for i in range(3):
						if triangle1[i] in valids1:
							continue
						triangle1[i] = valids1[0]
			triangles.append(triangle0)
			triangles.append(triangle1)
	triangles = np.array(triangles)
	triangles = triangles[:,[0,2,1]]
	return triangles
	
def generate_batch_triangles(batch_size, h, w, margin_x=2, margin_y=5, masks=None):
	batch_triangles = []
	for i in range(batch_size):
		batch_triangles.append(generate_triangles(h, w, margin_x, margin_y, mask=masks[i]))
	return batch_triangles


def dilate(bin_img, ksize=5):
	pad = (ksize - 1) // 2
	bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
	out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
	return out

def erode(bin_img, ksize=5):
	out = 1 - dilate(1 - bin_img, ksize)
	return out


def add_pointlight(vertices, normals, lights):
	'''
			vertices: [bz, nv, 3]
			lights: [bz, 6]
	returns:
			shading: [bz, nv, 3]
	'''

	light_positions = lights[:,:3].reshape(-1, 1, 3)
	light_intensities = lights[:,3:].reshape(-1, 1, 3)
	directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
	normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
	# normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
	shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
	return shading.mean(1)


def add_SHlight(normal_images, sh_coeff, constant_factor):
	'''
		sh_coeff: [bz, 9, 3]
	'''
	N = normal_images
	sh = torch.stack([
			N[:,0]*0.+1., N[:,0], N[:,1], \
			N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
			N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
			], 
			1) # [bz, 9, h, w]
	sh = sh*constant_factor[None,:,None,None]
	shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
	return shading



def get_imgs_masks(imgs):
	"""
	Args:
		imgs (tensor): [bz, 3, h, w]
	"""
	bz = imgs.shape[0]
	h = imgs.shape[2]
	w = imgs.shape[3]
 
	imgs = imgs.reshape(bz, 3, -1).abs().sum(1)
	imgs[imgs>0] = 1
	return imgs.reshape(bz, 1, h, w)


def knn_padding_imgs(imgs):
	"""
	Args:
		imgs (tensor): [bz, 3, h, w]
	"""	

	bz = imgs.shape[0]
	h = imgs.shape[2]
	w = imgs.shape[3]

	imgs = imgs.reshape(bz, 3, -1).abs().sum(1)
	nonzeros = torch.nonzero(imgs)
	masks = torch.zeros(bz, h * w, device=imgs.device)
	nonzeros = torch.split(nonzeros[:,1], [len(torch.where(nonzeros[:, 0] == i)[0]) for i in range(bz)], dim=0)
	for i in range(len(nonzeros)):
		masks[i, nonzeros[i]] = 1
	masks = masks.reshape(bz, 1, h, w)
	print('.')
 
# # borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
# def write_obj(obj_name,
#               vertices,
#               faces,
#               colors=None,
#               texture=None,
#               uvcoords=None,
#               uvfaces=None,
#               inverse_face_order=False,
#               normal_map=None,
#               ):
#     ''' Save 3D face model with texture. 
#     Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
#     Args:
#         obj_name: str
#         vertices: shape = (nver, 3)
#         colors: shape = (nver, 3)
#         faces: shape = (ntri, 3)
#         texture: shape = (uv_size, uv_size, 3)
#         uvcoords: shape = (nver, 2) max value<=1
#     '''
#     if os.path.splitext(obj_name)[-1] != '.obj':
#         obj_name = obj_name + '.obj'
#     mtl_name = obj_name.replace('.obj', '.mtl')
#     texture_name = obj_name.replace('.obj', '.png')
#     material_name = 'FaceTexture'

#     faces = faces.copy()
#     # mesh lab start with 1, python/c++ start from 0
#     faces += 1
#     if inverse_face_order:
#         faces = faces[:, [2, 1, 0]]
#         if uvfaces is not None:
#             uvfaces = uvfaces[:, [2, 1, 0]]

#     # write obj
#     with open(obj_name, 'w') as f:
#         # first line: write mtlib(material library)
#         # f.write('# %s\n' % os.path.basename(obj_name))
#         # f.write('#\n')
#         # f.write('\n')
#         if texture is not None:
#             f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

#         # write vertices
#         if colors is None:
#             for i in range(vertices.shape[0]):
#                 f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
#         else:
#             for i in range(vertices.shape[0]):
#                 f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

#         # write uv coords
#         if texture is None:
#             for i in range(faces.shape[0]):
#                 f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
#         else:
#             for i in range(uvcoords.shape[0]):
#                 f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
#             f.write('usemtl %s\n' % material_name)
#             # write f: ver ind/ uv ind
#             uvfaces = uvfaces + 1
#             for i in range(faces.shape[0]):
#                 f.write('f {}/{} {}/{} {}/{}\n'.format(
#                     #  faces[i, 2], uvfaces[i, 2],
#                     #  faces[i, 1], uvfaces[i, 1],
#                     #  faces[i, 0], uvfaces[i, 0]
#                     faces[i, 0], uvfaces[i, 0],
#                     faces[i, 1], uvfaces[i, 1],
#                     faces[i, 2], uvfaces[i, 2]
#                 )
#                 )
#             # write mtl
#             with open(mtl_name, 'w') as f:
#                 f.write('newmtl %s\n' % material_name)
#                 s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
#                 f.write(s)

#                 if normal_map is not None:
#                     name, _ = os.path.splitext(obj_name)
#                     normal_name = f'{name}_normals.png'
#                     f.write(f'disp {normal_name}')
#                     # out_normal_map = normal_map / (np.linalg.norm(
#                     #     normal_map, axis=-1, keepdims=True) + 1e-9)
#                     # out_normal_map = (out_normal_map + 1) * 0.5

#                     cv2.imwrite(
#                         normal_name,
#                         # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
#                         normal_map
#                     )
#             cv2.imwrite(texture_name, texture)


# def subdivide_mesh(mesh_path, times=1):
# 	""" 
# 	subdivide mesh

# 	Args:
# 		mesh_path (_type_): _description_
# 		times (int, optional): _description_. Defaults to 1.

# 	Returns:
# 		_type_: verts and faces after subdivide
# 	"""
# 	mesh = read_triangle_mesh(mesh_path)
# 	for _ in range(times):
# 		mesh = mesh.subdivide_loop()
# 	write_triangle_mesh(mesh_path.replace('.obj', '_subdivide.obj'), mesh)
# 	v_subdivide = np.asarray(mesh.vertices)
# 	f_subdivide = np.asarray(mesh.triangles)
# 	return v_subdivide, f_subdivide

 
# def save_detailed_mesh(self, filename, verts, faces, texture, uvcoords, uvfaces):
# 	'''
# 	vertices: [nv, 3], tensor
# 	texture: [3, h, w], tensor
# 	'''
# 	i = 0
# 	vertices = verts.detach().cpu().numpy()
# 	faces = faces.detach().cpu().numpy()
# 	uvcoords = uvcoords.detach().cpu().numpy()
# 	uvfaces = uvfaces.detach()
# 	# save coarse mesh, with texture and normal map
# 	normal_map = (normal_map * 0.5 + 0.5).detach().cpu().numpy()
# 	write_obj(filename, vertices, faces, 
# 					texture=texture, 
# 					uvcoords=uvcoords, 
# 					uvfaces=uvfaces, 
# 					normal_map=normal_map)
# 	# upsample mesh, save detailed mesh
# 	texture = texture[:,:,[2,1,0]]
# 	normals = normals.detach().cpu().numpy()
# 	displacement_map = displacement_map.detach().cpu().numpy().squeeze()
# 	dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
# 	write_obj(filename.replace('.obj', '_detail.obj'), 
# 					dense_vertices, 
# 					dense_faces,
# 					colors = dense_colors,
# 					inverse_face_order=True)
 
 
def normalize(x):
	x = (x - x.min()) / (x.max() - x.min() + 1e-7)
	x = x * 2 - 1
	return x


## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
	""" Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
	Load a mesh from a file-like object.
	"""
	with open(obj_filename, 'r') as f:
		lines = [line.strip() for line in f]

	verts, uvcoords = [], []
	faces, uv_faces = [], []
	# startswith expects each line to be a string. If the file is read in as
	# bytes then first decode to strings.
	if lines and isinstance(lines[0], bytes):
		lines = [el.decode("utf-8") for el in lines]

	for line in lines:
		tokens = line.strip().split()
		if line.startswith("v "):  # Line is a vertex.
			vert = [float(x) for x in tokens[1:4]]
			if len(vert) != 3:
				msg = "Vertex %s does not have 3 values. Line: %s"
				raise ValueError(msg % (str(vert), str(line)))
			verts.append(vert)
		elif line.startswith("vt "):  # Line is a texture.
			tx = [float(x) for x in tokens[1:3]]
			if len(tx) != 2:
				raise ValueError(
					"Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
				)
			uvcoords.append(tx)
		elif line.startswith("f "):  # Line is a face.
			# Update face properties info.
			face = tokens[1:]
			face_list = [f.split("/") for f in face]
			for vert_props in face_list:
				# Vertex index.
				faces.append(int(vert_props[0]))
				if len(vert_props) > 1:
					if vert_props[1] != "":
						# Texture index is present e.g. f 4/1/1.
						uv_faces.append(int(vert_props[1]))

	verts = torch.tensor(verts, dtype=torch.float32)
	uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
	faces = torch.tensor(faces, dtype=torch.long); faces = faces.reshape(-1, 3) - 1
	uv_faces = torch.tensor(uv_faces, dtype=torch.long); uv_faces = uv_faces.reshape(-1, 3) - 1
	return (
		verts,
		uvcoords,
		faces,
		uv_faces
	)


def knnsearch(A, B, n_neighbors=1):
	# knn find neighbors
	knn = NearestNeighbors(n_neighbors=n_neighbors)
	knn.fit(B)
	neigh_dist, neigh_ind = knn.kneighbors(A, return_distance=True)
	assert(len(neigh_ind) == len(neigh_dist) and len(neigh_ind) == len(A))
	return neigh_dist, neigh_ind


def find_nearest_pixel(img, target):
	nonzero = torch.nonzero(img)
	distances = torch.sqrt((nonzero[:,0] - target[0]).abs() + (nonzero[:,1] - target[1]).abs())
	nearest_index = torch.argmin(distances)
	return nonzero[nearest_index]


def find_batch_nearest_pixels(imgs):
	bz = imgs.shape[0]
	h = imgs.shape[-2]
	w = imgs.shape[-1]
	targets_x = torch.arange(0, h).repeat(w)
	targets_y = torch.arange(0, w).repeat(h)
	for i in range(bz):
		nonzero = torch.nonzero(imgs[i])
		distances = torch.zeros(h * w, nonzero.shape[0], 3)
		distances[:, :, :] = torch.sqrt((nonzero[:, 1] - targets_x).abs() + (nonzero[:, 2] - targets_y).abs())
  
  
def upsample_mesh(vertices, normals, faces, displacement_map, texture_map, dense_template):
	''' Credit to Timo
	upsampling coarse mesh (with displacment map)
		vertices: vertices of coarse mesh, [nv, 3]
		normals: vertex normals, [nv, 3]
		faces: faces of coarse mesh, [nf, 3]
		texture_map: texture map, [256, 256, 3]
		displacement_map: displacment map, [256, 256]
		dense_template:
	Returns:
		dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
		dense_colors: vertex color, [number of dense vertices, 3]
		dense_faces: [number of dense faces, 3]
	'''
	img_size = dense_template['img_size']
	dense_faces = dense_template['f']
	x_coords = dense_template['x_coords']
	y_coords = dense_template['y_coords']
	valid_pixel_ids = dense_template['valid_pixel_ids']		# sorted
	valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
	valid_pixel_b_coords = dense_template['valid_pixel_b_coords']	# 质心坐标（权重）

	pixel_3d_points = vertices[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
					vertices[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
					vertices[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
	vertex_normals = normals
	pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
					vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
					vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
	pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
	# TODO: CHECK
	displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
	dense_colors = texture_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
	offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
	dense_vertices = pixel_3d_points + offsets
	return dense_vertices, dense_colors, dense_faces


def make_dense_template(uvcoords, uvfaces, faces, dense_faces, uv_mask, tex_size, save_path):
	fixed_vertices = uvcoords.clone()
	fixed_vertices[...,:2] = -fixed_vertices[...,:2]
	meshes_screen = Meshes(verts=[fixed_vertices.float()], faces=[uvfaces.long()])
	pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
		meshes_screen,
		image_size=tex_size,
		blur_radius=0.00001,
		faces_per_pixel=1,
		bin_size=None,
		max_faces_per_bin=None,
		perspective_correct=False
	)
	pix_to_face = pix_to_face[0, :, :, 0]
	bary_coords = bary_coords[0, :, :, 0, :]
	flame_dense_template_path = 'data/texture_data_256.npy'
	flame_dense_template = np.load(flame_dense_template_path, allow_pickle=True, encoding='latin1').item()

	# img_size = flame_dense_template['img_size']
	# dense_faces = flame_dense_template['f']
	x_coords = flame_dense_template['x_coords']
	y_coords = flame_dense_template['y_coords']
	# valid_pixel_ids = flame_dense_template['valid_pixel_ids']
	# valid_pixel_3d_faces = flame_dense_template['valid_pixel_3d_faces']
	# valid_pixel_b_coords = flame_dense_template['valid_pixel_b_coords']	

	# # vis dense faces
	# a = np.zeros((tex_size, tex_size, 3))
	# for i in range(len(faces)):
	# 	f = dense_faces[i]
	# 	c = [random(), random(), random()]
	# 	for id in f:
	# 		x = int(x_coords[id.item()])
	# 		y = int(y_coords[id.item()])
	# 		a[y, x, :] = c
	# import cv2; cv2.imwrite('test.png', a * 255)

	valid_pixel_ids = []
	for i in range(len(x_coords)):
		x = int(x_coords[i])
		y = int(y_coords[i])
		if uv_mask[y, x] == 1:
			valid_pixel_ids.append(i)
	valid_pixel_ids = np.array(sorted(valid_pixel_ids))
	
	valid_pixel_3d_faces = []
	for id in valid_pixel_ids:
		x = int(x_coords[id])
		y = int(y_coords[id])
		face_id = pix_to_face[y, x]
		face_vals = faces[face_id]
		valid_pixel_3d_faces.append([face_vals[0].item(), face_vals[1].item(), face_vals[2].item()])
	valid_pixel_3d_faces = np.array(valid_pixel_3d_faces)
 
	valid_pixel_b_coords = []
	for id in valid_pixel_ids:
		x = int(x_coords[id])
		y = int(y_coords[id])
		b = bary_coords[y, x, :]
		valid_pixel_b_coords.append([b[0].item(), b[1].item(), b[2].item()])
	valid_pixel_b_coords = np.array(valid_pixel_b_coords)
	
	assert(len(valid_pixel_ids) == len(valid_pixel_3d_faces) == len(valid_pixel_b_coords))
	
	d = dict()
	d['img_size'] = tex_size
	d['f'] = dense_faces
	d['x_coords'] = x_coords
	d['y_coords'] = y_coords
	d['valid_pixel_ids'] = valid_pixel_ids
	d['valid_pixel_3d_faces'] = valid_pixel_3d_faces
	d['valid_pixel_b_coords'] = valid_pixel_b_coords
	
	# save
	with open(save_path, 'wb') as f:
		pickle.dump(d, f)

	print('dense template saved in: ' + save_path)
 

def subdivide_mesh_by_meshlab(mesh_path, iter=1, thres=1, uv_dis=None, use_neighbor=False):
	# TODO: input mesh with vt
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(mesh_path)
	ms.meshing_surface_subdivision_loop(loopweight=0, iterations=iter, threshold=pymeshlab.Percentage(thres))
	m = ms.current_mesh()
	rtn_vals = dict()
	rtn_vals['v'] = m.vertex_matrix()
	rtn_vals['f'] = m.face_matrix()
	rtn_vals['vn'] = np.asarray(F.normalize(torch.as_tensor(m.vertex_normal_matrix()), eps=1e-6, dim=1))
	if m.has_vertex_tex_coord():
		rtn_vals['vt'] = m.vertex_tex_coord_matrix()
	if m.has_wedge_tex_coord():
		rtn_vals['ft'] = m.wedge_tex_coord_matrix()
	if uv_dis is not None:
		if not m.has_vertex_tex_coord() or not m.has_wedge_tex_coord():
			raise AttributeError('Add uv_dis requires the mesh have verts_uvs and faces_uvs.')
		face_verts_order = rtn_vals['f'].reshape(-1).astype(int)
		h, w = uv_dis.shape[-2], uv_dis.shape[-1]
		off = np.zeros((rtn_vals['v'].shape[0], 1))
		d = np.asarray(uv_dis.permute(1, 2, 0).detach().cpu())
		x = h - 1 - (rtn_vals['ft'][:, 1] * (h - 1)).astype(int)
		y = (rtn_vals['ft'][:, 0] * (w - 1)).astype(int)
		off[face_verts_order, :] = d[x, y, :]
		if use_neighbor:
			x_neighs = [-2, -1, 0, 1, 2]
			y_neighs = [-2, -1, 0, 1, 2]
			z = 0
			tot = 0
			for dx in x_neighs:
				for dy in y_neighs:
					z += d[x + dx, y + dy, :]
					tot += 1
			off[face_verts_order, :] = z / tot
		rtn_vals['v'] += off * rtn_vals['vn']

	ms.clear()
	return rtn_vals

def get_mask_contour_pixels(mask):
	"""
	输入:
		mask: torch.Tensor 或 numpy array, dtype=np.uint8 或 bool, 0/255
			  shape: (C,H,W) 或 (H,W)
	输出:
		contour_pixels: numpy array, shape (Nc,2), (row, col)
						每个点严格在 mask 上为 True，并且是轮廓点
	"""
	# 1. 转成 numpy, dtype bool
	if isinstance(mask, torch.Tensor):
		mask = mask.detach().cpu()
		if mask.ndim == 3:  # C,H,W -> H,W
			mask = mask[0]
		mask_np = (mask.numpy() > 0)
	else:
		mask_np = (mask > 0)

	H, W = mask_np.shape

	# 2. 构建 padded mask，方便边界检查
	padded = np.pad(mask_np, pad_width=1, mode='constant', constant_values=0)

	# 3. 上下左右检查邻居是否有 False
	#    对 mask 中为 True 的点，如果其上下左右邻居有 False，则是轮廓
	neighbors = (
			padded[0:H, 1:W + 1] &  # 上
			padded[2:H + 2, 1:W + 1] &  # 下
			padded[1:H + 1, 0:W] &  # 左
			padded[1:H + 1, 2:W + 2]  # 右
	)
	# neighbors 为 True 表示**内部点**，取反就是轮廓
	boundary = mask_np & (~neighbors)

	# 4. 获取轮廓像素坐标
	ys, xs = np.where(boundary)
	contour_pixels = np.stack([ys, xs], axis=1)

	return contour_pixels

def stitch_seam(
	verts,
	faces,
	uvs,
	faces_uv,
	left_idx_list,
	right_idx_list,
	pos_mode='avg',
	uv_mode='avg',
	clean_unused=True,
	device=None
):
	"""
	纯粹：对齐 + 合并 + remap + 清理
	"""

	if device is None:
		device = verts.device

	verts = verts.clone()
	faces = faces.clone()
	uvs = uvs.clone() if uvs is not None else None
	faces_uv = faces_uv.clone() if faces_uv is not None else None

	# ====== idx ======
	if isinstance(left_idx_list, np.ndarray):
		left_idx_list = torch.from_numpy(left_idx_list.astype(np.int64))
	if isinstance(right_idx_list, np.ndarray):
		right_idx_list = torch.from_numpy(right_idx_list.astype(np.int64))

	left_idx_list = torch.as_tensor(left_idx_list, dtype=torch.long, device=device)
	right_idx_list = torch.as_tensor(right_idx_list, dtype=torch.long, device=device)

	# ==================================================
	# 1. Vertex merge
	# ==================================================
	if pos_mode == 'left':
		pass
	elif pos_mode == 'right':
		verts[left_idx_list] = verts[right_idx_list]
	elif pos_mode == 'avg':
		verts[left_idx_list] = (verts[left_idx_list] + verts[right_idx_list]) * 0.5
	else:
		raise ValueError("pos_mode must be left/right/avg")

	# ==================================================
	# 2. UV merge (per vertex UV)
	# ==================================================
	if uvs is not None:
		if uv_mode == 'left':
			pass
		elif uv_mode == 'right':
			uvs[left_idx_list] = uvs[right_idx_list]
		elif uv_mode == 'avg':
			uvs[left_idx_list] = (uvs[left_idx_list] + uvs[right_idx_list]) * 0.5
		else:
			raise ValueError("uv_mode must be left/right/avg")

	# ==================================================
	# 3. Vertex remap (R -> L)
	# ==================================================
	V = verts.shape[0]
	remap_v = torch.arange(V, device=device)
	remap_v[right_idx_list] = left_idx_list

	faces = remap_v[faces]

	# ==================================================
	# 4. UV face remap（关键补上）
	# ==================================================
	if faces_uv is not None:
		U = uvs.shape[0]
		remap_uv = torch.arange(U, device=device)

		faces_uv = remap_uv[faces_uv]

	# ==================================================
	# 5. delete degenerate faces
	# ==================================================
	keep = (
		(faces[:, 0] != faces[:, 1]) &
		(faces[:, 1] != faces[:, 2]) &
		(faces[:, 0] != faces[:, 2])
	)

	faces = faces[keep]
	if faces_uv is not None:
		faces_uv = faces_uv[keep]

	# ==================================================
	# 6. clean unused vertices / uv
	# ==================================================
	if clean_unused:

		used_v = torch.zeros(verts.shape[0], dtype=torch.bool, device=device)
		used_v[faces.reshape(-1)] = True

		new_v = torch.full((verts.shape[0],), -1, dtype=torch.long, device=device)
		new_v[used_v] = torch.arange(used_v.sum(), device=device)

		verts = verts[used_v]
		faces = new_v[faces]

		if uvs is not None and faces_uv is not None:

			used_uv = torch.zeros(uvs.shape[0], dtype=torch.bool, device=device)
			used_uv[faces_uv.reshape(-1)] = True

			new_uv = torch.full((uvs.shape[0],), -1, dtype=torch.long, device=device)
			new_uv[used_uv] = torch.arange(used_uv.sum(), device=device)

			uvs = uvs[used_uv]
			faces_uv = new_uv[faces_uv]

	return verts, faces, uvs, faces_uv

def compute_normal_color_in_camera(meshes, cam_R):
	verts = meshes.verts_packed()  # (V,3)
	faces = meshes.faces_packed()  # (F,3)
	R_y_180 = torch.tensor([
		[-1., 0., 0.],
		[0., 1., 0.],
		[0., 0., -1.]
	], device=verts.device)

	verts_flipped = (R_y_180 @ verts.T).T  # (V,3)

	verts_cam = (cam_R.T @ verts_flipped.T).T   # (V,3)

	mesh_cam = Meshes(verts=[verts_cam], faces=[faces])

	normals_cam = mesh_cam.verts_normals_packed()  # 已归一化单位向量

	# 5. 法线映射到颜色
	colors = (normals_cam + 1) / 2
	return colors

def simplify_mesh_with_uv(
	mesh_path,
	target_faces=50000,
	out_path="simplified.obj"
):
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(mesh_path)

	ms.meshing_decimation_quadric_edge_collapse(
		targetfacenum=target_faces,
		preservenormal=True,
		preservetopology=True,
		preserveboundary=True
	)

	ms.save_current_mesh(
		out_path,
		save_vertex_color=False,
		save_wedge_texcoord=True
	)

	return out_path