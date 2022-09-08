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

def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
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

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0,0] = 1492.645/700
    intrinsics[1,1] = 1492.645/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    return intrinsics

def fix_pose_orig(pose):
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3]/radius * 2.7
    return pose



def angle_trans_to_cams(angle, trans):

    R = compute_rotation(torch.from_numpy(angle))[0].numpy()
    trans[2] += -10
    c = -np.dot(R, trans)
    pose = np.eye(4)
    pose[:3, :3] = R

    c *= 0.27 # normalize camera radius
    c[1] += 0.006 # additional offset used in submission
    c[2] += 0.161 # additional offset used in submission
    pose[0,3] = c[0]
    pose[1,3] = c[1]
    pose[2,3] = c[2]

    focal = 1492.645 # = 1015*512/224*(300/466.285)#
    pp = 256#112
    w = 512#224
    h = 512#224

    count = 0
    K = np.eye(3)
    K[0][0] = focal
    K[1][1] = focal
    K[0][2] = w/2.0
    K[1][2] = h/2.0
    intrinsics = K.tolist()

    Rot = np.eye(3)
    Rot[0, 0] = 1
    Rot[1, 1] = -1
    Rot[2, 2] = -1        
    pose[:3, :3] = np.dot(pose[:3, :3], Rot)

    pose = fix_pose_orig(pose)
    intrinsics = fix_intrinsics(intrinsics)
    cams = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)])

    return cams