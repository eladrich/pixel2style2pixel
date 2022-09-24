# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import os
import torch
import json
import argparse
import scipy.io
import sys

def compute_rotation(angles, rank):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        device = torch.device(rank)

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(device)
        zeros = torch.zeros([batch_size, 1]).to(device)

        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:]
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = torch.bmm(torch.bmm(rot_z, rot_y), rot_x)
        return rot.permute(0, 2, 1)

def fix_intrinsics(intrinsics):
    assert intrinsics.shape[1:] == (3, 3), intrinsics
    intrinsics[:,0,0] = 1492.645/700
    intrinsics[:,1,1] = 1492.645/700
    intrinsics[:,0,2] = 1/2
    intrinsics[:,1,2] = 1/2
    assert intrinsics[:,0,1].all() == 0
    assert intrinsics[:,2,2].all() == 1
    assert intrinsics[:,1,0].all() == 0
    assert intrinsics[:,2,0].all() == 0
    assert intrinsics[:,2,1].all() == 0
    return intrinsics

def fix_pose_orig(pose):
    location = pose[:, :3, 3].clone()
    radius = torch.sqrt(torch.sum(location**2,dim= 1))
    pose[:, :3, 3] =  location / radius[:,None] * 2.7

    return pose

def make_batch_identity(batch, dim, rank):

    device = torch.device(rank)

    im = torch.eye(dim, device=device, requires_grad=True)
    im = im.reshape((1, dim, dim))
    im = im.repeat(batch, 1, 1)

    return im

def angle_trans_to_cams(angle, trans, rank):

    device = torch.device(rank)

    batch = angle.shape[0]
    # R = compute_rotation(angle)
    R = compute_rotation_matrix_from_ortho6d(angle,rank = rank)
    trans = trans.to(device)
    mean_trans = torch.Tensor([0.01645587, 0.23713592, 2.5715761]).to(device)
    trans = trans + mean_trans

    c = -torch.bmm(R, trans[:,:,None]).squeeze()
   
    pose = make_batch_identity(batch, 4, rank)

    pose[:, :3, :3] = R

    # c *= 0.27 # normalize camera radius
    # c[1] += 0.006 # additional offset used in submission
    # c[2] += 0.161 # additional offset used in submission

    pose[:,:3,3] = c

    focal = 2985.29 # = 1015*1024/224*(300/466.285)#
    pp = 512#112
    w = 1024#224
    h = 1024#224

    count = 0
    K = make_batch_identity(batch, 3, rank)
    K[:,0,0] = focal
    K[:,1,1] = focal
    K[:,0,2] = w/2.0
    K[:,1,2] = h/2.0

    # Rot = make_batch_identity(batch,3)
    # Rot[:, 0, 0] = 1
    # Rot[:, 1, 1] = -1
    # Rot[:, 2, 2] = -1
        
    # pose[:, :3, :3] = torch.bmm(pose[:, :3, :3].clone(), Rot)

    # fixed_pose = fix_pose_orig(pose)

    intrinsics = fix_intrinsics(K)

    cams = torch.cat([pose.reshape(batch,-1), intrinsics.reshape(batch,-1)],dim=1)

    return cams

def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True, rank = 0):

    device = torch.device(rank)

    x_mean = torch.Tensor([ 9.59558767e-01,  4.84514121e-04, -6.47509161e-03]).to(device)
    y_mean = torch.Tensor([1.80135038e-04, -9.87191543e-01, -8.92407249e-02]).to(device)
    x_raw = poses[:,0:3] + x_mean[None,:]#batch*3 
    y_raw = poses[:,3:6] + y_mean[None,:]#batch*3

    x = normalize_vector(x_raw, use_gpu, rank=rank) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z, use_gpu,rank=rank)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def normalize_vector(v, use_gpu=True, rank = 0):

    device = torch.device(rank)
    
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch

    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))

    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out