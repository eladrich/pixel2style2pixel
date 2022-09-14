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

def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        device = 'cuda:0'

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(device)
        zeros = torch.zeros([batch_size, 1]).to(device)

        x, y, z = angles[:, :1].clone(), angles[:, 1:2].clone(), angles[:, 2:].clone(),
        
        rot_x = torch.cat([
            ones.clone(), zeros.clone(), zeros.clone(),
            zeros.clone(), torch.cos(x), -torch.sin(x), 
            zeros.clone(), torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros.clone(), torch.sin(y),
            zeros.clone(), ones.clone(), zeros.clone(),
            -torch.sin(y), zeros.clone(), torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros.clone(),
            torch.sin(z), torch.cos(z), zeros.clone(),
            zeros.clone(), zeros.clone(), ones.clone()
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
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
    pose[:, :3, 3] /= radius[:,None] * 2.7

    return pose

def make_batch_identity(batch, dim):

    im = torch.eye(dim, device='cuda:0', requires_grad=True)
    im = im.reshape((1, dim, dim))
    im = im.repeat(batch, 1, 1)

    return im


def angle_trans_to_cams(angle, trans):

    batch = angle.shape[0]
    R = compute_rotation(angle)
    trans = trans.to('cuda:0')
    trans[:,2] += -10

    c = -torch.bmm(R, trans[:,:,None]).squeeze()
   
    pose = make_batch_identity(batch, 4)

    pose[:, :3, :3] = R

    c *= 0.27 # normalize camera radius
    c[:,1] += 0.006 # additional offset used in submission
    c[:,2] += 0.161 # additional offset used in submission
    pose[:,0,3] = c[:,0]
    pose[:,1,3] = c[:,1]
    pose[:,2,3] = c[:,2]

    focal = 1492.645 # = 1015*512/224*(300/466.285)#
    pp = 256#112
    w = 512#224
    h = 512#224

    count = 0
    K = make_batch_identity(batch, 3)
    K[:,0,0] = focal
    K[:,1,1] = focal
    K[:,0,2] = w/2.0
    K[:,1,2] = h/2.0

    Rot = make_batch_identity(batch,3)
    Rot[:, 0, 0] = 1
    Rot[:, 1, 1] = -1
    Rot[:, 2, 2] = -1
        
    pose[:, :3, :3] = torch.bmm(pose[:, :3, :3].clone(), Rot).squeeze()

    pose = fix_pose_orig(pose)

    intrinsics = fix_intrinsics(K)

    cams = torch.cat([pose.reshape(batch,-1), intrinsics.reshape(batch,-1)],dim=1)

    return cams