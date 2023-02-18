import torch
import torch.nn as nn
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	bce_loss = nn.BCELoss()
	loss = bce_loss(voxel_src, voxel_tgt)
	return loss


def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	device = point_cloud_src.device 	
	dists = torch.cdist(point_cloud_src, point_cloud_tgt)
	loss_src = torch.min(dists, dim=2)[0]
	loss_tgt = torch.min(dists, dim=1)[0]
	loss_chamfer = torch.mean(loss_src) + torch.mean(loss_tgt)
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss

	vertices = mesh_src.verts_packed()
	device = vertices.device
	num_vertices = vertices.shape[0]

    # 计算每个顶点的邻居顶点
	neighbors = mesh_src.verts_packed_to_mesh_idx()
	num_neighbors = neighbors.count_nonzero(dim=1)

    # 对于每个顶点，计算它和其相邻顶点之间的距离
	distances = torch.cdist(vertices, vertices)

    # 计算 Laplacian smoothing loss
	laplacian_loss = torch.tensor(0., device=device)
	for v in range(num_vertices):
		neighbor_indices = neighbors[v, :num_neighbors[v]]
		neighbor_vertices = vertices[neighbor_indices]
		center = vertices[v]
		mean = torch.mean(neighbor_vertices, dim=0)
		laplacian_loss += torch.sum((center - mean) ** 2)

    # 归一化损失
	smoothness_loss /= num_vertices

	return smoothness_loss
